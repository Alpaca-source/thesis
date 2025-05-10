from neo4j_qna import Neo4jHandler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Initialize the Neo4jHandler
    handler = Neo4jHandler()
    
    # Get the latest experiment per type per variation
    df_experiments = handler.get_latest_experiment_per_type_per_variation()
    
    # Print the resulting DataFrame
    print(df_experiments)
    
    # Visualization: BERT Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='variation', y='bert_score', data=df_experiments)
    plt.xticks(rotation=90)
    plt.title('BERT Scores per Variation')
    plt.xlabel('Variation')
    plt.ylabel('BERT Score')
    plt.tight_layout()
    plt.savefig('bert_scores_per_variation.png')
    plt.show()

    # Visualization: Cosine Similarities
    plt.figure(figsize=(10, 6))
    sns.barplot(x='variation', y='cosine_similarity', data=df_experiments)
    plt.xticks(rotation=90)
    plt.title('Cosine Similarities per Variation')
    plt.xlabel('Variation')
    plt.ylabel('Cosine Similarity')
    plt.tight_layout()
    plt.savefig('cosine_similarities_per_variation.png')
    plt.show()
    
    # Close the handler
    handler.close()

if __name__ == "__main__":
    main()