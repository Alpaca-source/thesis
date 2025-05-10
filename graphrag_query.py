import subprocess
import pandas as pd
import os
import openai
import time
from neo4j_qna import Neo4jHandler
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Azure OpenAI environment variables
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize GraphRAG instance

def get_graphrag_response(query):
    """
    Fetches a response from GraphRAG.

    Args:
        query (str): The user query.

    Returns:
        tuple: (response_text, duration)
    """
    root = "./graphrag/"
    method = "local"
    #query = "Who had a No 1 in the 80's with Karma Chameleon"
    
    command = [
        "graphrag", "query",
        "--root", root,
        "--method", method,
        "--query", query
    ]

    start_time = time.time()
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        response_text = result.stdout.strip()
        end_time = time.time()
        duration = end_time - start_time
        return response_text, duration
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        return f"Error: {str(e)}", duration
    

def test_variant(variant, authoritative_response):
    # Calculate BERTScore
    P, R, F1 = score([variant], [authoritative_response], lang="en", verbose=True)
    bert_score_value = F1.mean().item()

    # Calculate cosine similarity
    vectorizer = TfidfVectorizer().fit_transform([variant, authoritative_response])
    vectors = vectorizer.toarray()
    cosine_sim_value = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    return bert_score_value, cosine_sim_value, f"{variant}"

# Initialize the Neo4jHandler
handler = Neo4jHandler()

# Get all variations and their corresponding authoritative responses from the database
query = """
MATCH (q:Question)-[:HAS_VARIATION]->(v:Variation)-[:HAS_AUTHORITATIVE_RESPONSE]->(r:Response)
RETURN v.text AS variation, r.text AS authoritative_response
"""
with handler.driver.session(database=handler.NEO4J_DATABASE_QNA_METRICS) as session:
    result = session.run(query)
    variations_and_responses = [(record["variation"], record["authoritative_response"]) for record in result]

# Get all questions and responses from the database
query = """
MATCH (q:Question)-[:HAS_RESPONSE]->(r:Response)
RETURN q.text AS question, r.text AS response
"""
with handler.driver.session(database=handler.NEO4J_DATABASE_QNA_METRICS) as session:
    result = session.run(query)
    questions_and_responses = [{"question": record["question"], "response": record["response"]} for record in result]

# Create a DataFrame of all questions and responses
df_questions_responses = pd.DataFrame(questions_and_responses)

# Combine the Q&A data with the prompt
prompt = "You are an AI assistant that strictly follows the Q&A examples provided. Given a new question, respond only if it closely matches the format, topic, and style of the provided examples. If the question does not align with the given Q&A examples, respond with 'I don't have an answer for that.'\n\n"
for index, row in df_questions_responses.iterrows():
    prompt += f"Q: {row['question']}\nA: {row['response']}\n\n"

# Run each variation through the test_variant function and insert the result as an experiment
for i, variation in enumerate(handler.get_variations_without_experiment_type("graphrag")):
    print(i)
    print(variation)
    response_text, duration = get_graphrag_response(variation)
    print(response_text)
    try:
        handler.score_experiment(variation, response_text, "graphrag", 0, 0, 0, duration)
    except Exception as e:
        print(f"Failed to score experiment for variation: {variation}, Error: {str(e)}")
        #variants_failed = variants_failed.append({"variation": variation, "error": str(e)}, ignore_index=True)
    
    # Add a 10-second delay
    time.sleep(3)

# Print the combined prompt
print(prompt)

# Example question to get a response from GraphRAG
#example_question = "Who had a No 1 in the 80's with Karma Chameleon?"
#response, duration = get_graphrag_response(example_question)
#print(f"Response to '{example_question}': {response}")
#print(f"Time taken: {duration} seconds")

# Close the Neo4jHandler
handler.close()