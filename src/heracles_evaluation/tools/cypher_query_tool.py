from heracles.query_interface import Neo4jWrapper

from heracles_evaluation.tool_interface import FunctionParameter, ToolDescription
from heracles_evaluation.tool_registry import ToolRegistry, register_tool
from heracles_evaluation.dsg_interfaces import HeraclesDsgInterface


def query_db(cypher_string, dsgdb_conf: HeraclesDsgInterface = None):
    if dsgdb_conf is None:
        raise ValueError(
            "query_db called with dsgdb_conf=None. Did you forget to bind the config to the tool?"
        )
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


# TODO: we need to warp the query_db in another function that takes only the cypher string, and not the dsgdb_conf
# Probably need to have the experiment runner automatically insert the experiment description into the tool call?
cypher_tool = ToolDescription(
    name="run_cypher_query",
    description="An interface for running Cypher queries on a Neo4j database containing a 3D Scene Graph.",
    parameters=[
        FunctionParameter("cypher_string", str, "Your Cypher query"),
    ],
    function=query_db,
)

register_tool(cypher_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
