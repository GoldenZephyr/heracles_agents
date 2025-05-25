from heracles.query_interface import Neo4jWrapper

# IP / Port for database
URI = "neo4j://127.0.0.1:7687"
# Database name / password for database
AUTH = ("neo4j", "neo4j_pw")

# Assumes that the scene graph has already been loaded into the database
with Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False) as db:
    objects = db.query(
        "MATCH (n: Object) RETURN DISTINCT n.class as class, COUNT(*) as count"
    )
    print(objects)
