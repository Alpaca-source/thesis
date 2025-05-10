from neo4j_qna import Neo4jHandler

# Initialize the Neo4jHandler
handler = Neo4jHandler()

# Clear the database
handler.clear_database()

# Close the Neo4jHandler
handler.close()