import pandas as pd
import json

# Load CSV
csv_path = "MastersQnA.csv"
df = pd.read_csv(csv_path)

# Convert to JSONL format required by GraphRAG
jsonl_path = "qa_dataset.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = {"text": f"Q: {row['Utterence']}\nA: {row['Response']}"}
        f.write(json.dumps(record) + "\n")

print(f"Converted {csv_path} to {jsonl_path}")
