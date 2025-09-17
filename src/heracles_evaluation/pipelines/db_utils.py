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
        if dsgdb_conf.n_object_verification is not None:
            v = db.query("MATCH (n: Object) RETURN COUNT(*) as count")
            count = v[0]["count"]
            assert (
                count == dsgdb_conf.n_object_verification
            ), f"Connected database has {count} objects ({dsgdb_conf.n_object_verification} expected)"
        try:
            query_result = str(db.query(cypher_string))
            return True, query_result
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            return False, query_result
