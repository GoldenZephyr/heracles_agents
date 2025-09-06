#!/usr/bin/env python3
import sys

import yaml
from rich.console import Console
from rich.table import Table

from heracles_evaluation.llm_interface import AnalyzedQuestions


def to_string(value):
    if type(value) is bool:
        if value:
            color = "green"
        else:
            color = "red"
        return colorize(color, value)
    else:
        return value


def colorize(color, string):
    return f"[{color}]{string}[/{color}]"


def summarize_results(questions: list[dict]):
    n_questions = len(questions)

    acc = {}
    for k, v in questions[0].items():
        if type(v) in [int, float, bool]:
            acc[k] = 0

    for q in questions:
        for k, v in q.items():
            if type(v) in [int, float, bool]:
                acc[k] += v

    # cypher_color = "green" if valid_cypher == n_questions else "red"
    # cypher_str = colorize(cypher_color, f"{valid_cypher}/{n_questions}")

    string_summaries = {k: f"{v}/{n_questions}" for k, v in acc.items()}
    string_summaries["questions"] = str(n_questions)

    ratio_summaries = {k: v / n_questions for k, v in acc.items()}
    ratio_summaries["questions"] = n_questions

    return ratio_summaries, string_summaries


def construct_per_question_info(aqs: AnalyzedQuestions):
    per_question_info = []
    for q in aqs.analyzed_questions:
        answer_dict = q.analysis.model_dump(mode="json")
        answer_dict["name"] = q.question.name
        answer_dict["question"] = q.question.question
        per_question_info.append(answer_dict)
    return per_question_info


def display_analyzed_question_table(title, aqs: AnalyzedQuestions, column_data_map={}):
    table = generate_analyzed_question_table(title, aqs, column_data_map)
    console = Console()
    console.print(table)


def generate_analyzed_question_table(title, aqs: AnalyzedQuestions, column_data_map={}):
    per_question_info = construct_per_question_info(aqs)
    return generate_table(title, per_question_info, column_data_map)


def generate_table(title, row_data, column_data_map={}):
    table = Table(title=title, show_header=True, header_style="bold cyan")

    used_fields = set()
    for c, v in column_data_map.items():
        table.add_column(c)
        used_fields.add(v)

    non_remapped_fields = []
    for k in row_data[0]:
        if k not in used_fields:
            table.add_column(k)
            non_remapped_fields.append(k)

    for q in row_data:
        data = [to_string(q[d]) for d in column_data_map.values()] + [
            to_string(q[d]) for d in non_remapped_fields
        ]
        table.add_row(*data)
    return table


def display_table(title, row_data, column_data_map={}):
    table = generate_table(title, row_data, column_data_map)
    console = Console()
    console.print(table)


def display_experiment_results(aqs):
    column_data_map = {
        "Name": "name",
        "Question": "question",
    }
    display_analyzed_question_table("Test Table", aqs, column_data_map)

    summary_column_data_map = {
        "# Questions": "questions",
    }
    result_dicts = [q.analysis.model_dump(mode="json") for q in aqs.analyzed_questions]
    summary_data = [summarize_results(result_dicts)[1]]
    display_table("Summary", summary_data, column_data_map=summary_column_data_map)


def display_experiment_results_with_answer(per_question_info, title="Title"):
    column_data_map = {
        "Name": "name",
        "Question": "question",
        "Solution": "solution",
        "Answer": "answer",
    }
    # display_analyzed_question_table("Test Table", aqs, column_data_map)

    # summary_column_data_map = {
    #    "# Questions": "questions",
    # }
    # result_dicts = [q.analysis.model_dump(mode="json") for q in aqs.analyzed_questions]

    table = generate_table(title, per_question_info, column_data_map)
    console = Console()
    console.print(table)


def main():
    if len(sys.argv) < 2:
        print("Usage: ./summarize_results.py yaml_path")
        exit(1)

    refined_out_eval = sys.argv[1]

    with open(refined_out_eval, "r") as fo:
        results = yaml.safe_load(fo)

    column_data_map = {
        "Name": "name",
        "Question": "question",
        "Valid Cypher": "valid_cypher",
        "Valid SLDP": "valid_sldp",
        "Correct": "correct",
    }

    display_table("Results", column_data_map, results["questions"])

    summary_column_data_map = {
        "Questions": "questions",
        "Valid Cypher": "valid_cypher",
        "Valid SLDP": "valid_sldp",
        "Correct": "correct",
    }
    summary_data = summarize_results(results["questions"])[1]
    display_table("Results Summary", summary_column_data_map, summary_data)


if __name__ == "__main__":
    main()
