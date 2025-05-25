from rich.table import Table
import yaml
from rich.console import Console


def colorize(color, string):
    return f"[{color}]{string}[/{color}]"


def summarize_results(questions):
    n_questions = len(questions)
    valid_cypher = 0
    valid_sldp = 0
    correct = 0
    for q in questions:
        if q["valid_cypher"]:
            valid_cypher += 1
        if q["valid_sldp"]:
            valid_sldp += 1
        if q["correct"]:
            correct += 1

    cypher_color = "green" if valid_cypher == n_questions else "red"
    cypher_str = colorize(cypher_color, f"{valid_cypher}/{n_questions}")

    sldp_color = "green" if valid_sldp == n_questions else "red"
    sldp_str = colorize(sldp_color, f"{valid_sldp}/{n_questions}")

    correct_color = "green" if correct == n_questions else "red"
    correct_str = colorize(correct_color, f"{correct}/{n_questions}")

    return {
        "questions": str(n_questions),
        "valid_cypher": cypher_str,
        "valid_sldp": sldp_str,
        "correct": correct_str,
    }


def to_string(value):
    if type(value) is bool:
        if value:
            color = "green"
        else:
            color = "red"
        return colorize(color, value)
    else:
        return value


table = Table(title="Results", show_header=True, header_style="bold cyan")

column_data_map = {
    "Name": "name",
    "Question": "question",
    "Valid Cypher": "valid_cypher",
    "Valid SLDP": "valid_sldp",
    "Correct": "correct",
}

for c in column_data_map:
    table.add_column(c)

with open("refined_out.yaml", "r") as fo:
    results = yaml.safe_load(fo)

for q in results["questions"]:
    table.add_row(*[to_string(q[d]) for d in column_data_map.values()])


console = Console()
console.print(table)


summary = Table(title="Results Summary", show_header=True, header_style="bold cyan")


column_data_map = {
    "Questions": "questions",
    "Valid Cypher": "valid_cypher",
    "Valid SLDP": "valid_sldp",
    "Correct": "correct",
}

for c in column_data_map:
    summary.add_column(c)

summary_data = summarize_results(results["questions"])
summary.add_row(*[to_string(summary_data[d]) for d in column_data_map.values()])
console.print(summary)
