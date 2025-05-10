import pandas as pd
from bert_score import score

# Define the file path
file_path = '/workspace/MastersQnA.csv'

def calculate_bertscore(references, candidates):
    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    return P, R, F1

# Basic BERTScore test
references = ["I am a llama"]
candidates = ["I am a chinchilla"]

P, R, F1 = calculate_bertscore(references, candidates)
print(f"Precision: {P.tolist()}")
print(f"Recall: {R.tolist()}")
print(f"F1: {F1.tolist()}")

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
#print(df)