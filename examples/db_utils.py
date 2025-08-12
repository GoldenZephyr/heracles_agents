from heracles.query_interface import Neo4jWrapper


def query_db(dsgdb_conf, cypher_string):
    with Neo4jWrapper(
        dsgdb_conf.uri,
        (
            dsgdb_conf.username.get_secret_value(),
            dsgdb_conf.password.get_secret_value(),
        ),
        atomic_queries=True,
        print_profiles=False,
    ) as db:
        try:
            query_result = str(db.query(cypher_string))
            return True, query_result
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            return False, query_result
