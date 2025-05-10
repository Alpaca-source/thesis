from neo4j import GraphDatabase
import os
from datetime import datetime
import pandas as pd
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class Neo4jHandler:
    def __init__(self):
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")  # or your actual URI
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.NEO4J_DATABASE_QNA_METRICS = os.getenv("NEO4J_DATABASE_QNA_METRICS", "qnametrics")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.create_database_if_not_exists(self.NEO4J_DATABASE_QNA_METRICS)

    def create_database_if_not_exists(self, database_name):
        """Create the database if it doesn't exist (Enterprise Edition only)"""
        query = f"CREATE DATABASE {database_name} IF NOT EXISTS"
        with self.driver.session(database="system") as session:
            session.run(query)

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._clear_database)

    def score_experiment(self, variant, variant_response, method, input_tokens=0, intermediate_tokens=0, output_tokens=0, duration=0):
        # Calculate BERTScore
        authoritative_response = self.get_authoritative_response_by_variation(variant)
        bert_score_value, cosine_sim_value = self.test_variant(variant_response, authoritative_response)
        self.add_experiment_to_variation(variant, variant_response, bert_score_value, cosine_sim_value, method, input_tokens, intermediate_tokens, output_tokens, duration)

    def test_variant(self, variant_response, authoritative_response):
        # Handle the case where "3" is a variant response as well as an authoritative response
        if variant_response.strip().lower() == authoritative_response.strip().lower():
            return 1.0, 1.0  # Perfect score for exact matches (case insensitive)

        # Calculate BERTScore
        P, R, F1 = score([variant_response], [authoritative_response], lang="en", verbose=True)
        bert_score_value = F1.mean().item()

        # Calculate cosine similarity
        vectorizer = TfidfVectorizer().fit_transform([variant_response, authoritative_response])
        vectors = vectorizer.toarray()
        cosine_sim_value = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

        return bert_score_value, cosine_sim_value


    def get_all_questions_and_authoritative_responses(self):
        query = """
        MATCH (q:Question)-[:HAS_RESPONSE]->(r:Response)
        RETURN q.text AS question, r.text AS response
        """
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.run(query)
            questions_and_responses = [{"question": record["question"], "response": record["response"]} for record in result]
        return questions_and_responses

    @staticmethod
    def _clear_database(tx):
        query = "MATCH (n) DETACH DELETE n"
        tx.run(query)

    def create_question_with_variations(self, question_id, question_text, response, variations):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._create_and_link_variations, question_id, question_text, response, variations)

    @staticmethod
    def _create_and_link_variations(tx, question_id, question_text, response, variations):
        query = (
            "CREATE (q:Question {id: $question_id, text: $question_text}) "
            "CREATE (r:Response {text: $response}) "
            "CREATE (q)-[:HAS_RESPONSE]->(r) "
            "WITH q, r "
            "UNWIND $variations AS variation "
            "CREATE (v:Variation {text: variation}) "
            "CREATE (q)-[:HAS_VARIATION]->(v) "
            "CREATE (v)-[:HAS_AUTHORITATIVE_RESPONSE]->(r)"
        )
        tx.run(query, question_id=question_id, question_text=question_text, response=response, variations=variations)

    def add_response_to_variation(self, variation_text, response):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._add_response_to_variation, variation_text, response)

    @staticmethod
    def _add_response_to_variation(tx, variation_text, response):
        query = (
            "MATCH (v:Variation {text: $variation_text}) "
            "CREATE (r:Response {text: $response}) "
            "CREATE (v)-[:HAS_RESPONSE]->(r)"
        )
        tx.run(query, variation_text=variation_text, response=response)

    def get_variations_and_responses(self, question_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.execute_read(self._get_variations_and_responses, question_text)
        return result

    @staticmethod
    def _get_variations_and_responses(tx, question_text):
        query = (
            "MATCH (q:Question {text: $question_text})-[:HAS_VARIATION]->(v:Variation)-[:HAS_RESPONSE]->(r:Response) "
            "RETURN v.text AS variation, r.text AS response"
        )
        result = tx.run(query, question_text=question_text)
        return [{"variation": record["variation"], "response": record["response"]} for record in result]

    def get_question_by_variation_text(self, variation_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.execute_read(self._get_question_by_variation_text, variation_text)
        return result

    @staticmethod
    def _get_question_by_variation_text(tx, variation_text):
        query = (
            "MATCH (q:Question)-[:HAS_VARIATION]->(v:Variation {text: $variation_text}) "
            "RETURN q.text AS question"
        )
        result = tx.run(query, variation_text=variation_text)
        return result.single()["question"]

    def get_response_by_variation_text(self, variation_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.execute_read(self._get_response_by_variation_text, variation_text)
        return result

    @staticmethod
    def _get_response_by_variation_text(tx, variation_text):
        query = (
            "MATCH (v:Variation {text: $variation_text})-[:HAS_RESPONSE]->(r:Response) "
            "RETURN r.text AS response"
        )
        result = tx.run(query, variation_text=variation_text)
        return result.single()["response"]

    def create_variation_for_question(self, question_text, new_variation):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._create_variation_for_question, question_text, new_variation)

    @staticmethod
    def _create_variation_for_question(tx, question_text, new_variation):
        query = (
            "MATCH (q:Question {text: $question_text})-[:HAS_RESPONSE]->(r:Response) "
            "CREATE (v:Variation {text: $new_variation}) "
            "CREATE (q)-[:HAS_VARIATION]->(v) "
            "CREATE (v)-[:HAS_AUTHORITATIVE_RESPONSE]->(r)"
        )
        tx.run(query, question_text=question_text, new_variation=new_variation)

    def add_experiment_to_variation(self, variation_text, generated_response, bert_score, cosine_similarity, experiment_type, input_tokens, thinking_tokens, output_tokens, runtime):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._add_experiment_to_variation, variation_text, generated_response, bert_score, cosine_similarity, experiment_type, input_tokens, thinking_tokens, output_tokens, runtime)

    @staticmethod
    def _add_experiment_to_variation(tx, variation_text, generated_response, bert_score, cosine_similarity, experiment_type, input_tokens, thinking_tokens, output_tokens, runtime):
        query = (
            "MATCH (v:Variation {text: $variation_text}) "
            "CREATE (e:Experiment {date_time: datetime(), generated_response: $generated_response, bert_score: $bert_score, cosine_similarity: $cosine_similarity, type: $experiment_type, input_tokens: $input_tokens, thinking_tokens: $thinking_tokens, output_tokens: $output_tokens, runtime: $runtime}) "
            "CREATE (v)-[:HAS_EXPERIMENT]->(e)"
        )
        tx.run(query, variation_text=variation_text, generated_response=generated_response, bert_score=bert_score, cosine_similarity=cosine_similarity, experiment_type=experiment_type, input_tokens=input_tokens, thinking_tokens=thinking_tokens, output_tokens=output_tokens, runtime=runtime)

    def get_experiment_by_variation_text(self, variation_text, experiment_type=None):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.execute_read(self._get_experiment_by_variation_text, variation_text, experiment_type)
        return result

    @staticmethod
    def _get_experiment_by_variation_text(tx, variation_text, experiment_type):
        if experiment_type:
            query = (
                "MATCH (v:Variation {text: $variation_text})-[:HAS_EXPERIMENT]->(e:Experiment {type: $experiment_type}) "
                "RETURN e ORDER BY e.date_time DESC LIMIT 1"
            )
            result = tx.run(query, variation_text=variation_text, experiment_type=experiment_type)
        else:
            query = (
                "MATCH (v:Variation {text: $variation_text})-[:HAS_EXPERIMENT]->(e:Experiment) "
                "RETURN e ORDER BY e.date_time DESC LIMIT 1"
            )
            result = tx.run(query, variation_text=variation_text)
        return result.single()

    def delete_question(self, question_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._delete_question, question_text)

    @staticmethod
    def _delete_question(tx, question_text):
        query = (
            "MATCH (q:Question {text: $question_text}) "
            "OPTIONAL MATCH (q)-[:HAS_VARIATION]->(v:Variation)-[:HAS_RESPONSE]->(r:Response) "
            "OPTIONAL MATCH (v)-[:HAS_EXPERIMENT]->(e:Experiment) "
            "DETACH DELETE q, v, r, e"
        )
        tx.run(query, question_text=question_text)

    def delete_variation(self, variation_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._delete_variation, variation_text)

    @staticmethod
    def _delete_variation(tx, variation_text):
        query = (
            "MATCH (v:Variation {text: $variation_text}) "
            "OPTIONAL MATCH (v)-[:HAS_EXPERIMENT]->(e:Experiment) "
            "DETACH DELETE v, e"
        )
        tx.run(query, variation_text=variation_text)

    def update_question(self, old_question_text, new_question_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._update_question, old_question_text, new_question_text)

    @staticmethod
    def _update_question(tx, old_question_text, new_question_text):
        query = (
            "MATCH (q:Question {text: $old_question_text}) "
            "SET q.text = $new_question_text"
        )
        tx.run(query, old_question_text=old_question_text, new_question_text=new_question_text)

    def update_variation(self, old_variation_text, new_variation_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._update_variation, old_variation_text, new_variation_text)

    @staticmethod
    def _update_variation(tx, old_variation_text, new_variation_text):
        query = (
            "MATCH (v:Variation {text: $old_variation_text}) "
            "SET v.text = $new_variation_text"
        )
        tx.run(query, old_variation_text=old_variation_text, new_variation_text=new_variation_text)

    def update_response(self, old_response_text, new_response_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            session.execute_write(self._update_response, old_response_text, new_response_text)

    @staticmethod
    def _update_response(tx, old_response_text, new_response_text):
        query = (
            "MATCH (r:Response {text: $old_response_text}) "
            "SET r.text = $new_response_text"
        )
        tx.run(query, old_response_text=old_response_text, new_response_text=new_response_text)

    def get_authoritative_response_by_variation(self, variation_text):
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.execute_read(self._get_authoritative_response_by_variation, variation_text)
        return result

    @staticmethod
    def _get_authoritative_response_by_variation(tx, variation_text):
        query = (
            "MATCH (v:Variation {text: $variation_text})-[:HAS_AUTHORITATIVE_RESPONSE]->(r:Response) "
            "RETURN r.text AS authoritative_response"
        )
        result = tx.run(query, variation_text=variation_text)
        return result.single()["authoritative_response"]

    def get_all_variations(self):
        query = """
        MATCH (v:Variation)
        RETURN v.text AS variation
        """
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.run(query)
            variations = [record["variation"] for record in result]
        return variations    
    
    def get_latest_experiment_per_type_per_variation(self):
        query = """
        MATCH (v:Variation)-[:HAS_EXPERIMENT]->(e:Experiment)
        WITH v, e
        ORDER BY e.date_time DESC
        WITH v, e.type AS type, COLLECT(e)[0] AS latest_experiment
        MATCH (v)-[:HAS_AUTHORITATIVE_RESPONSE]->(r:Response)
        MATCH (q:Question)-[:HAS_VARIATION]->(v)
        RETURN v.text AS variation, q.text AS question, r.text AS authoritative_response, latest_experiment.generated_response AS generated_response, latest_experiment.date_time AS date_time, latest_experiment.cosine_similarity AS cosine_similarity, latest_experiment.bert_score AS bert_score, type
        """
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.run(query)
            experiments = [{"variation": record["variation"], "question": record["question"], "authoritative_response": record["authoritative_response"], "generated_response": record["generated_response"], "date_time": record["date_time"], "cosine_similarity": record["cosine_similarity"], "bert_score": record["bert_score"], "type": record["type"]} for record in result]
        return pd.DataFrame(experiments)

    def get_variations_without_experiment_type(self, experiment_type):
        query = """
        MATCH (v:Variation)
        WHERE NOT (v)-[:HAS_EXPERIMENT]->(:Experiment {type: $experiment_type})
        RETURN v.text AS variation
        """
        with self.driver.session(database=self.NEO4J_DATABASE_QNA_METRICS) as session:
            result = session.run(query, experiment_type=experiment_type)
            variations = [record["variation"] for record in result]
        return variations

# Usage example
if __name__ == "__main__":
    handler = Neo4jHandler()
    
    # Example to get the latest experiment per type per variation
    temp = handler.get_latest_experiment_per_type_per_variation()
    temp.to_csv("Exp_1_Test.csv", index=False)

    print(temp.head())
    print(temp['question'])

    # Example to get variations without a specific experiment type
    experiment_type = "type1"
    variations_without_experiment = handler.get_variations_without_experiment_type(experiment_type)
    print(f"Variations without experiment type '{experiment_type}': {variations_without_experiment}")

    handler.close()