import pandas as pd
from neo4j_qna import Neo4jHandler

# Define the file path
file_path = '/workspace/MastersQnA.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Initialize the Neo4jHandler
handler = Neo4jHandler()

# Iterate over each row in the DataFrame and store the data in Neo4j
for index, row in df.iterrows():
    question_id = f"q{index + 1}"
    question_text = row['Utterence']
    response = row['Response']
    variations = [row['Variation 1'], row['Variation 2'], row['Variation 3']]
    handler.create_question_with_variations(question_id, question_text, response, variations)

# Close the Neo4jHandler
handler.close()