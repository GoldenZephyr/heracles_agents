import logging
import os
from string import Template

import yaml

from heracles_evaluation.llm_interface import AnalyzedExperiment

# from heracles_evaluation.experiment_definition import (
#    AnalyzedExperiment,
# )  # , AnalyzedQuestions
from heracles_evaluation.summarize_results import (
    display_experiment_results_with_answer,
    # display_experiment_results,
    summarize_results,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


def load_results(fn):
    with open(fn, "r") as fo:
        yml = yaml.safe_load(fo)

    ae = AnalyzedExperiment(**yml)
    return ae


latex_macro_template_table_1 = r"""
\newcommand{\cypherXqaXsmall}{$cypher_gpt_41_qa_b45}
\newcommand{\agenticcypherXqaXsmall}{$agentic_cypher_gpt_41_qa_b45}
\newcommand{\incontextXqaXsmall}{$in_context_gpt_41_qa_b45}
\newcommand{\agenticincontextXqaXsmall}{$agentic_in_context_gpt_41_qa_b45}
\newcommand{\pythonXqaXsmall}{$python_gpt_41_qa_b45}
\newcommand{\agenticpythonXqaXsmall}{$agentic_python_gpt_41_qa_b45}

\newcommand{\cypherXqaXlarge}{$cypher_gpt_41_qa_westpoint}
\newcommand{\agenticcypherXqaXlarge}{$agentic_cypher_gpt_41_qa_westpoint}
\newcommand{\incontextXqaXlarge}{$in_context_gpt_41_qa_westpoint}
\newcommand{\agenticincontextXqaXlarge}{$agentic_in_context_gpt_41_qa_westpoint}
\newcommand{\pythonXqaXlarge}{$python_gpt_41_qa_westpoint}
\newcommand{\agenticpythonXqaXlarge}{$agentic_python_gpt_41_qa_westpoint}

\newcommand{\cypherXpddlXsmall}{$cypher_gpt_41_pddl_b45}
\newcommand{\agenticcypherXpddlXsmall}{$agentic_cypher_gpt_41_pddl_b45}
\newcommand{\incontextXpddlXsmall}{$in_context_gpt_41_pddl_b45}
\newcommand{\agenticincontextXpddlXsmall}{$agentic_in_context_gpt_41_pddl_b45}
\newcommand{\pythonXpddlXsmall}{$python_gpt_41_pddl_b45}
\newcommand{\agenticpythonXpddlXsmall}{$agentic_python_gpt_41_pddl_b45}

\newcommand{\cypherXpddlXlarge}{$cypher_gpt_41_pddl_westpoint}
\newcommand{\agenticcypherXpddlXlarge}{$agentic_cypher_gpt_41_pddl_westpoint}
\newcommand{\incontextXpddlXlarge}{$in_context_gpt_41_pddl_westpoint}
\newcommand{\agenticincontextXpddlXlarge}{$agentic_in_context_gpt_41_pddl_westpoint}
\newcommand{\pythonXpddlXlarge}{$python_gpt_41_pddl_westpoint}
\newcommand{\agenticpythonXpddlXlarge}{$agentic_python_gpt_41_pddl_westpoint}
"""

latex_macro_template_table_2 = r"""
% ==== gpt-4.1 ====
\newcommand{\agenticcypherXqaXsmallgptFourOne}{$agentic_cypher_gpt_41_qa_b45}
\newcommand{\incontextXqaXsmallgptFourOne}{$in_context_gpt_41_qa_b45}
\newcommand{\agenticincontextXqaXsmallgptFourOne}{$agentic_in_context_gpt_41_qa_b45}
\newcommand{\agenticpythonXqaXsmallgptFourOne}{$agentic_python_gpt_41_qa_b45}

\newcommand{\agenticcypherXqaXlargegptFourOne}{$agentic_cypher_gpt_41_qa_westpoint}
\newcommand{\incontextXqaXlargegptFourOne}{$in_context_gpt_41_qa_westpoint}
\newcommand{\agenticincontextXqaXlargegptFourOne}{$agentic_in_context_gpt_41_qa_westpoint}
\newcommand{\agenticpythonXqaXlargegptFourOne}{$agentic_python_gpt_41_qa_westpoint}

\newcommand{\agenticcypherXpddlXsmallgptFourOne}{$agentic_cypher_gpt_41_pddl_b45}
\newcommand{\incontextXpddlXsmallgptFourOne}{$in_context_gpt_41_pddl_b45}
\newcommand{\agenticincontextXpddlXsmallgptFourOne}{$agentic_in_context_gpt_41_pddl_b45}
\newcommand{\agenticpythonXpddlXsmallgptFourOne}{$agentic_python_gpt_41_pddl_b45}

\newcommand{\agenticcypherXpddlXlargegptFourOne}{$agentic_cypher_gpt_41_pddl_westpoint}
\newcommand{\incontextXpddlXlargegptFourOne}{$in_context_gpt_41_pddl_westpoint}
\newcommand{\agenticincontextXpddlXlargegptFourOne}{$agentic_in_context_gpt_41_pddl_westpoint}
\newcommand{\agenticpythonXpddlXlargegptFourOne}{$agentic_python_gpt_41_pddl_westpoint}

% ==== gpt-4.1-mini ====
\newcommand{\agenticcypherXqaXsmallgptFourOneMini}{$agentic_cypher_gpt_41_mini_qa_b45}
\newcommand{\incontextXqaXsmallgptFourOneMini}{$in_context_gpt_41_mini_qa_b45}
\newcommand{\agenticincontextXqaXsmallgptFourOneMini}{$agentic_in_context_gpt_41_mini_qa_b45}
\newcommand{\agenticpythonXqaXsmallgptFourOneMini}{$agentic_python_gpt_41_mini_qa_b45}

\newcommand{\agenticcypherXqaXlargegptFourOneMini}{$agentic_cypher_gpt_41_mini_qa_westpoint}
\newcommand{\incontextXqaXlargegptFourOneMini}{$in_context_gpt_41_mini_qa_westpoint}
\newcommand{\agenticincontextXqaXlargegptFourOneMini}{$agentic_in_context_gpt_41_mini_qa_westpoint}
\newcommand{\agenticpythonXqaXlargegptFourOneMini}{$agentic_python_gpt_41_mini_qa_westpoint}

\newcommand{\agenticcypherXpddlXsmallgptFourOneMini}{$agentic_cypher_gpt_41_mini_pddl_b45}
\newcommand{\incontextXpddlXsmallgptFourOneMini}{$in_context_gpt_41_mini_pddl_b45}
\newcommand{\agenticincontextXpddlXsmallgptFourOneMini}{$agentic_in_context_gpt_41_mini_pddl_b45}
\newcommand{\agenticpythonXpddlXsmallgptFourOneMini}{$agentic_python_gpt_41_mini_pddl_b45}

\newcommand{\agenticcypherXpddlXlargegptFourOneMini}{$agentic_cypher_gpt_41_mini_pddl_westpoint}
\newcommand{\incontextXpddlXlargegptFourOneMini}{$in_context_gpt_41_mini_pddl_westpoint}
\newcommand{\agenticincontextXpddlXlargegptFourOneMini}{$agentic_in_context_gpt_41_mini_pddl_westpoint}
\newcommand{\agenticpythonXpddlXlargegptFourOneMini}{$agentic_python_gpt_41_mini_pddl_westpoint}

% ==== gpt-4.1-nano ====
\newcommand{\agenticcypherXqaXsmallgptFourOneNano}{$agentic_cypher_gpt_41_nano_qa_b45}
\newcommand{\incontextXqaXsmallgptFourOneNano}{$in_context_gpt_41_nano_qa_b45}
\newcommand{\agenticincontextXqaXsmallgptFourOneNano}{$agentic_in_context_gpt_41_nano_qa_b45}
\newcommand{\agenticpythonXqaXsmallgptFourOneNano}{$agentic_python_gpt_41_nano_qa_b45}

\newcommand{\agenticcypherXqaXlargegptFourOneNano}{$agentic_cypher_gpt_41_nano_qa_westpoint}
\newcommand{\incontextXqaXlargegptFourOneNano}{$in_context_gpt_41_nano_qa_westpoint}
\newcommand{\agenticincontextXqaXlargegptFourOneNano}{$agentic_in_context_gpt_41_nano_qa_westpoint}
\newcommand{\agenticpythonXqaXlargegptFourOneNano}{$agentic_python_gpt_41_nano_qa_westpoint}

\newcommand{\agenticcypherXpddlXsmallgptFourOneNano}{$agentic_cypher_gpt_41_nano_pddl_b45}
\newcommand{\incontextXpddlXsmallgptFourOneNano}{$in_context_gpt_41_nano_pddl_b45}
\newcommand{\agenticincontextXpddlXsmallgptFourOneNano}{$agentic_in_context_gpt_41_nano_pddl_b45}
\newcommand{\agenticpythonXpddlXsmallgptFourOneNano}{$agentic_python_gpt_41_nano_pddl_b45}

\newcommand{\agenticcypherXpddlXlargegptFourOneNano}{$agentic_cypher_gpt_41_nano_pddl_westpoint}
\newcommand{\incontextXpddlXlargegptFourOneNano}{$in_context_gpt_41_nano_pddl_westpoint}
\newcommand{\agenticincontextXpddlXlargegptFourOneNano}{$agentic_in_context_gpt_41_nano_pddl_westpoint}
\newcommand{\agenticpythonXpddlXlargegptFourOneNano}{$agentic_python_gpt_41_nano_pddl_westpoint}
"""


table_to_template = {}
table_to_template["table1"] = latex_macro_template_table_1
table_to_template["table2"] = latex_macro_template_table_2


def get_per_question_info(analyzed_questions):
    result_dicts = [
        q.analysis.model_dump(mode="json")
        for q in analyzed_questions.analyzed_questions
    ]
    for rd, q in zip(result_dicts, analyzed_questions.analyzed_questions):
        rd["answer"] = q.answer
        rd["solution"] = q.question.solution
        rd["name"] = q.question.name
        rd["question"] = q.question.question
    return result_dicts


table_name = "table1"
results_dir = f"tables/{table_name}/output"
output_dir = f"tables/{table_name}/latex"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

results_fns = os.listdir(results_dir)
print(results_fns)

correct_ratio = {}
for fn in results_fns:
    print("=== Checking ", fn, " ===")
    element_path = os.path.join(results_dir, fn)
    results = load_results(element_path)
    assert len(results.experiment_configurations.values()) <= 1
    logger.debug(f"Loaded results: {results}")

    config_name = results.metadata["config_name"]
    name_for_interpolation = config_name.replace(".", "").replace("-", "_")
    if results.metadata.get("skip", False):
        print("  Had no results: ", fn)
        correct_ratio[name_for_interpolation] = "-"
        continue
    analyzed_questions = list(results.experiment_configurations.values())[0]
    print("  Processing ", fn)
    # display_experiment_results(analyzed_questions)
    # display_analyzed_question_table(results)

    per_question_summary_info = get_per_question_info(analyzed_questions)
    display_experiment_results_with_answer(per_question_summary_info, title=config_name)
    ratio_summaries, _ = summarize_results(per_question_summary_info)
    correct_ratio[name_for_interpolation] = f"{ratio_summaries['correct']:.2f}"

output_macros = Template(table_to_template[table_name]).safe_substitute(**correct_ratio)
print(output_macros)
