from heracles_evaluation.experiment_definition import PipelineRegistry


def get_method_to_pipeline():
    pr = PipelineRegistry.pipelines
    method_to_pipeline = {
        "cypher": pr["feedforward_cypher"],
        "agentic_cypher": pr["agentic"],
        "in_context": pr["feedforward_in_context"],
        "agentic_in_context": None,  # pr["agentic_in_context"]
        "python": None,  # pr["feedforward_python"],
        "agentic_python": None,  # pr["agentic"],
    }
    return method_to_pipeline
