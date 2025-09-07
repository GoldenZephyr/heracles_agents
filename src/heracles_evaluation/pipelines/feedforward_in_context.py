import copy
import logging

import spark_dsg

from heracles_evaluation.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_evaluation.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
)
from heracles_evaluation.pipelines.comparisons import evaluate_answer
from heracles_evaluation.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PromptingFailure(Exception):
    pass


def get_region_parent_of_object(object_node, scene_graph):
    # Get the parent of the object (Place)
    parent_place_id = object_node.get_parent()
    if not parent_place_id:
        return "none"
    parent_place_node = scene_graph.get_node(parent_place_id)
    parent_region_id = parent_place_node.get_parent()
    if not parent_region_id:
        return "none"
    parent_region_node = scene_graph.get_node(parent_region_id)
    return parent_region_node.id.str(True)


def object_to_prompt(object_node, scene_graph):
    attrs = object_node.attributes
    symbol = object_node.id.str(True)
    object_labelspace = scene_graph.get_labelspace(2, 0)
    if not object_labelspace:
        raise PromptingFailure("No available object labelspace")
    semantic_type = object_labelspace.get_category(attrs.semantic_label)
    position = f"({attrs.position[0]},{attrs.position[1]})"
    parent_region = get_region_parent_of_object(object_node, scene_graph)
    object_prompt = f"\n-\t(id={symbol}, type={semantic_type}, pos={position}, parent_region={parent_region})"
    return object_prompt


def region_to_prompt(region_node, scene_graph):
    attrs = region_node.attributes
    symbol = region_node.id.str(True)
    region_labelspace = scene_graph.get_labelspace(4, 0)
    if not region_labelspace:
        raise PromptingFailure("No available region labelspace")
    semantic_type = region_labelspace.get_category(attrs.semantic_label)
    region_prompt = f"\n-\t(id={symbol}, type={semantic_type})"
    return region_prompt


def scene_graph_to_prompt(scene_graph):
    # Add the objects
    objects_prompt = ""
    for object_node in scene_graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        objects_prompt += object_to_prompt(object_node, scene_graph)
    # Add the regions
    regions_prompt = ""
    for region_node in scene_graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        regions_prompt += region_to_prompt(region_node, scene_graph)
    # Construct the prompt
    scene_graph_prompt = (
        f"<Scene Graph>"
        f"\nObjects: {objects_prompt}"
        f"\nRegions: {regions_prompt}"
        f"</Scene Graph>"
    )
    return scene_graph_prompt


def generate_prompt(
    incontext_dsg_interface,
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    dsg_desciption = scene_graph_to_prompt(incontext_dsg_interface.get_dsg())
    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, dsg_description=dsg_desciption
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        print(ex)
        raise ex

    prompt.answer_semantic_guidance = "Make you answer as concise as possible"
    prompt.answer_formatting_guidance = get_answer_formatting_guidance(
        agent_config, question
    )

    print("prompt: ")
    print(prompt)
    return prompt


def incontext_dsg(exp):
    analyzed_questions = []
    for question in exp.questions:
        cxt = AgentContext(exp.phases["main"])

        prompt = generate_prompt(exp.dsg_interface, question, exp.phases["main"])

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Final Answer: {answer}\n")

        sequence = AgentSequence(
            description="in-context pipeline", responses=cxt.get_agent_responses()
        )

        valid_format, correct = evaluate_answer(
            question.correctness_comparator, answer, question.solution
        )

        logger.info(f"\n\nCorrect? {correct}\n\n")

        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_format)
        aq = AnalyzedQuestion(
            question=question, answer=answer, sequences=[sequence], analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Map question to answer using in-context scene graph",
)

d = PipelineDescription(
    name="feedforward_in_context",
    description="in-context scene graph",
    phases=[main_phase],
    function=incontext_dsg,
)

register_pipeline(d)

if __name__ == "__main__":
    import yaml

    from heracles_evaluation.experiment_definition import ExperimentConfiguration
    from heracles_evaluation.summarize_results import display_experiment_results

    with open("experiments/dsg_incontext_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    experiment = ExperimentConfiguration(**yml)
    logger.debug(f"Loaded experiment configuration: {experiment}")

    aqs = incontext_dsg(experiment)
    with open("output/dsgdb_feedforward_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
