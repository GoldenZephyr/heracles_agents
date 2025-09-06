import logging
import os

import agentic_pipeline  # NOQA
import feedforward_cypher_pipeline  # NOQA
import feedforward_in_context  # NOQA
import test_task  # NOQA
import yaml

from heracles_evaluation.llm_interface import AnalyzedExperiment

# from heracles_evaluation.experiment_definition import (
#    AnalyzedExperiment,
# )  # , AnalyzedQuestions
from heracles_evaluation.summarize_results import (
    display_experiment_results,
    summarize_results,
)
from string import Template

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


def load_results(fn):
    with open(fn, "r") as fo:
        yml = yaml.safe_load(fo)

    ae = AnalyzedExperiment(**yml)
    return ae


latex_macro_template = """
\\newcommand{\\cypherXqaXsmall}{$cypher_qa_b45}
\\newcommand{\\agenticcypherXqaXsmall}{$agentic_cypher_qa_b45}
\\newcommand{\\incontextXqaXsmall}{$in_context_qa_b45}
\\newcommand{\\agenticincontextXqaXsmall}{$agentic_in_context_qa_b45}
\\newcommand{\\pythonXqaXsmall}{$python_qa_b45}
\\newcommand{\\agenticpythonXqaXsmall}{$agentic_python_qa_b45}
\
\\newcommand{\\cypherXqaXlarge}{$cypher_qa_westpoint}
\\newcommand{\\agenticcypherXqaXlarge}{$agentic_cypher_qa_westpoint}
\\newcommand{\\incontextXqaXlarge}{$in_context_qa_westpoint}
\\newcommand{\\agenticincontextXqaXlarge}{agentic_in_context_qa_westpoint}
\\newcommand{\\pythonXqaXlarge}{$python_qa_westpoint}
\\newcommand{\\agenticpythonXqaXlarge}{$agentic_python_qa_westpoint}
\
\\newcommand{\\cypherXpddlXsmall}{$cypher_pddl_b45}
\\newcommand{\\agenticcypherXpddlXsmall}{$agentic_cypher_pddl_b45}
\\newcommand{\\incontextXpddlXsmall}{$in_context_pddl_b45}
\\newcommand{\\agenticincontextXpddlXsmall}{$agentic_in_context_pddl_b45}
\\newcommand{\\pythonXpddlXsmall}{$python_pddl_b45}
\\newcommand{\\agenticpythonXpddlXsmall}{$agentic_python_pddl_b45}
\
\\newcommand{\\cypherXpddlXlarge}{$cypher_pddl_westpoint}
\\newcommand{\\agenticcypherXpddlXlarge}{$agentic_cypher_pddl_westpoint}
\\newcommand{\\incontextXpddlXlarge}{$in_context_pddl_westpoint}
\\newcommand{\\agenticincontextXpddlXlarge}{$agentic_in_context_pddl_westpoint}
\\newcommand{\\pythonXpddlXlarge}{$python_pddl_westpoint}
\\newcommand{\\agenticpythonXpddlXlarge}{$agentic_python_pddl_westpoint}

"""


results_dir = "tables/table1/output"
output_dir = "tables/table1/latex"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_fns = os.listdir(results_dir)

correct_ratio = {}
for fn in results_fns:
    print("=== Checking ", fn, " ===")
    element_path = os.path.join(results_dir, fn)
    results = load_results(element_path)
    assert len(results.experiment_configurations.values()) <= 1
    logger.debug(f"Loaded results: {results}")

    config_name = results.metadata["config_name"]
    if results.metadata.get("skip", False):
        print("  Had no results: ", fn)
        correct_ratio[config_name] = "-"
        continue
    analyzed_questions = list(results.experiment_configurations.values())[0]
    print("  Processing ", fn)
    display_experiment_results(analyzed_questions)
    # display_analyzed_question_table(results)
    result_dicts = [
        q.analysis.model_dump(mode="json")
        for q in analyzed_questions.analyzed_questions
    ]
    ratio_summaries, _ = summarize_results(result_dicts)
    correct_ratio[config_name] = f"{ratio_summaries['correct']:.2f}"

output_macros = Template(latex_macro_template).safe_substitute(**correct_ratio)
print(output_macros)
