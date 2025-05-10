import os
import asyncio
import pandas as pd
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
from neo4j import GraphDatabase
from datetime import datetime
from neo4j_qna import Neo4jHandler
import time

logging.basicConfig(level=logging.INFO)
load_dotenv()

class BaseRag:
    def __init__(self):
        # Azure OpenAI settings
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        self.AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

        # Azure Embedding settings
        self.AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
        self.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        self.AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
        self.AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")

        # Embedding dimension
        self.embedding_dimension = 3072

        # Working directory
        self.WORKING_DIR = "./documents"
        if not os.path.exists(self.WORKING_DIR):
            os.mkdir(self.WORKING_DIR)

        # Initialize LightRAG instance
        self.rag = LightRAG(
            working_dir=self.WORKING_DIR,
            llm_model_func=self.llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dimension,
                max_token_size=8192,
                func=self.embedding_func,
            ),
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.90,  # 90% similarity
                "use_llm_check": False        # or True, if you want an extra LLM-based check
            },
            node2vec_params={
                'dimensions': self.embedding_dimension,  # <-- updated here
                'num_walks': 10,
                'walk_length': 40,
                'window_size': 2,
                'iterations': 3,
                'random_seed': 3
            },
            graph_storage="Neo4JStorage",  # <-----------override KG default
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            log_level="DEBUG"  # <-----------override log_level default
        )

        # Initialize Neo4j driver
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.neo4j_database = os.getenv("NEO4J_SCRAPING_DATABASE", "scraping")
        self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

        logging.info("BaseRag initialized with Neo4j and Azure OpenAI settings.")

    async def llm_model_func(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        client = AzureOpenAI(
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        chat_completion = client.chat.completions.create(
            model=self.AZURE_OPENAI_DEPLOYMENT,  # deployment name as model
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            top_p=kwargs.get("top_p", 1),
            n=kwargs.get("n", 1),
        )
        logging.info(f"LLM model function called with prompt: {prompt}")
        return chat_completion.choices[0].message.content

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        client = AzureOpenAI(
            api_key=self.AZURE_EMBEDDING_API_KEY,
            api_version=self.AZURE_EMBEDDING_API_VERSION,
            azure_endpoint=self.AZURE_EMBEDDING_ENDPOINT,
        )
        embedding = client.embeddings.create(model=self.AZURE_EMBEDDING_DEPLOYMENT, input=texts)
        embeddings = [item.embedding for item in embedding.data]
        logging.info(f"Embedding function called with texts: {texts}")
        return np.array(embeddings)

    def insert_text(self, text: str, doc_id: str):
        self.rag.insert(text, ids=[doc_id])
        logging.info(f"Inserted text into RAG with doc_id: {doc_id}")

    def delete_text(self, doc_id: str):
        self.rag.adelete_by_doc_id(doc_id)
        logging.info(f"Deleted text from RAG with doc_id: {doc_id}")

    def update_text(self, doc_id: str, new_text: str):
        self.delete_text(doc_id)
        self.insert_text(new_text, doc_id)
        logging.info(f"Updated text in RAG with doc_id: {doc_id}")

    def query_rag(self, query_text: str, query_param: QueryParam, system_prompt: str = None):
        result = self.rag.query(query_text, param=query_param, system_prompt=system_prompt)
        logging.info(f"Queried RAG with query_text: {query_text}")
        return result
    
    def query_with_separate_keyword_extraction(self, query: str, query_param: QueryParam, system_prompt: str = None) -> tuple:
        """
        Query the RAG with separate keyword extraction.

        Args:
            query (str): The query text.
            query_param (QueryParam): The query parameters.
            system_prompt (str, optional): The system prompt. Defaults to None.

        Returns:
            tuple: (result, duration, input_tokens, retrieved_tokens, generation_tokens)
        """
        start_time = time.time()
        try:
            result = self.rag.query_with_separate_keyword_extraction(
                query=query,
                prompt=system_prompt,
                param=query_param
            )
            end_time = time.time()
            duration = end_time - start_time
            print(result)

            # Extract token usage details
            input_tokens = result['usage']['prompt_tokens'] if 'usage' in result else 0
            retrieved_tokens = result['usage']['retrieved_tokens'] if 'usage' in result else 0
            generation_tokens = result['usage']['completion_tokens'] if 'usage' in result else 0

            logging.info(f"Queried Separate Query RAG with Prompt of:\n {system_prompt} \n\n with query of: \n {query}")
            return result, duration, input_tokens, retrieved_tokens, generation_tokens
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logging.error(f"Error querying Separate Query RAG: {e}")
            return {"error": str(e)}, duration, 0, 0, 0

# Usage example
if __name__ == '__main__':
    base_rag = BaseRag()
    handler = Neo4jHandler()

    df_questions_responses = pd.DataFrame(handler.get_all_questions_and_authoritative_responses())

    # Combine the Q&A data with the prompt
    prompt = "You are a QA agent responsible for answering questions concisely. If the question closely matches a known answer, return only the answer—no explanations, context, or formatting beyond what is necessary for clarity. If you cannot confidently generate a response, return only: 'I do not have a response."
    data = ""
    for index, row in df_questions_responses.iterrows():
        data += f"Q: {row['question']}\nA: {row['response']}\n\n"

    #base_rag.insert_text(data, "1")
    for i, variation in enumerate(handler.get_variations_without_experiment_type("lightrag")):
        print(i)
        print(variation)
        response_text, duration, input_tokens, retrieved_tokens, generation_tokens = base_rag.query_with_separate_keyword_extraction(query=variation, query_param=QueryParam(mode="hybrid"), system_prompt=prompt)
        try:
            handler.score_experiment(variation, response_text, "lightrag", input_tokens, retrieved_tokens, generation_tokens, duration)
        except Exception as e:
            print(f"Failed to score experiment for variation: {variation}, Error: {str(e)}")
            #variants_failed = variants_failed.append({"variation": variation, "error": str(e)}, ignore_index=True)
        
        # Add a 10-second delay
        time.sleep(3)

# Print the combined prompt
    #print(prompt)

    # Example question to get a response from LightRAG
    example_question = "Who was the only fictional character to feature in Time Magazine's 100 most important people of the 20th Century?"
    #print(f"Response to '{example_question}': {response}")
    #print(f"Time taken: {duration} seconds")

    # Close the Neo4jHandler
    handler.close()

