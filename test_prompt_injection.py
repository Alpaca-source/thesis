import pandas as pd
import os
import openai
import time
from openai import AzureOpenAI
from neo4j_qna import Neo4jHandler
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Azure OpenAI environment variables
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def get_openai_response(prompt, question):
    """
    Fetches a response from Azure OpenAI's Chat Completions API.

    Args:
        prompt (str): The system instruction prompt, which includes Q&A examples.
        question (str): The user question.

    Returns:
        tuple: (response_text, input_tokens, output_tokens, duration)
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": (
                    "You are an AI assistant that follows a structured Q&A format. "
                    "Use only the information from the provided Q&A examples to generate responses. "
                    "If the question does not match any example, respond with: 'I don't have an answer for that.' "
                    "Maintain a concise and accurate response style."
                )},
                {"role": "user", "content": f"{prompt}\nQ: {question}\nA:"}
            ],
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=["\n"]
        )

        # Extract response text
        print(response)
        response_text = response.choices[0].message.content.strip() if response.choices else "Error: No response generated"

        end_time = time.time()
        duration = end_time - start_time

        # Extract token usage details
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return response_text, input_tokens, output_tokens, duration
    
    except openai.OpenAIError as e:
        return f"Error: {str(e)}", 0, 0, 0

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


# Create a DataFrame of all questions and responses
df_questions_responses = pd.DataFrame(handler.get_all_questions_and_authoritative_responses())

# Combine the Q&A data with the prompt
prompt = "You are an AI assistant that strictly follows the Q&A examples provided. Given a new question, respond only if it closely matches the format, topic, and style of the provided examples. If the question does not align with the given Q&A examples, respond with 'I don't have an answer for that.'\n\n"
for index, row in df_questions_responses.iterrows():
    prompt += f"Q: {row['question']}\nA: {row['response']}\n\n"

# Run each variation through the test_variant function and insert the result as an experiment

for i, variation in enumerate(handler.get_variations_without_experiment_type("prompt_injection")):
    print(i)
    print(variation)
    response_text, input_tokens, output_tokens, duration = get_openai_response(prompt, variation)
    try:
        handler.score_experiment(variation, response_text, "prompt_injection", input_tokens, 0, output_tokens, duration)
    except Exception as e:
        print(f"Failed to score experiment for variation: {variation}, Error: {str(e)}")
        #variants_failed = variants_failed.append({"variation": variation, "error": str(e)}, ignore_index=True)
    
    # Add a 10-second delay
    time.sleep(3)

# Save the failed variants to a CSV file
#variants_failed.to_csv("Failed_Variants.csv", index=False)


handler.close()