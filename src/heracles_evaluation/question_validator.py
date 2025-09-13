#!/usr/bin/env python3
import yaml
import typer
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table


from llm_interface import EvalQuestion
from pypddl.pddl_goal_parser import lark_parse_pddl_goal


console = Console()
app = typer.Typer(help="Explore EvalQuestions from a YAML file.")


def load_yaml(path: Path) -> list[EvalQuestion]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return [EvalQuestion(**item) for item in data["questions"]]


def validate_solution(answer: str) -> bool:
    try:
        lark_parse_pddl_goal(answer)
        return True
    except Exception as ex:
        print(ex)
        return False


def render_table(
    questions: list[EvalQuestion], show_solutions: bool = False, show_tags=False, validate: bool = False
):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("UID", style="cyan")
    table.add_column("Name", style="yellow")
    table.add_column("Question", style="blue")
    if show_solutions:
        table.add_column("Solution", style="green")
    if show_tags:
        table.add_column("Tags", style="blue")
    if validate:
        table.add_column("Valid?", style="green")

    for q in questions:
        r = (str(q.uid), q.name, q.question)
        if show_solutions:
            r += (q.solution,)
        if show_tags:
            if q.tags and len(q.tags) > 0:
                r += (", ".join(q.tags),)
            else:
                r += []
        if validate:
            is_valid = validate_solution(q.solution)
            r += ("✅" if is_valid else "❌",)
        table.add_row(*r)
        # table.add_row(
        #    str(q.uid),
        #    q.name,
        #    q.question,
        #    q.solution if show_solutions else "",
        #    ", ".join(q.tags or []),
        # )

    console.print(table)


@app.command()
def list(
    file: Path = typer.Argument(..., help="YAML file with EvalQuestions"),
    solutions: bool = typer.Option(False, "--solutions", help="Show solutions"),
    show_tags: bool = typer.Option(False, "--show-tags", help="Show tags"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate solutions with PDDL parser"),
):
    """List all questions (optionally with solutions or filtered by tag)."""
    questions = load_yaml(file)

    if tag:
        questions = [q for q in questions if tag in (q.tags or [])]

    render_table(questions, show_solutions=solutions, show_tags=show_tags, validate=validate)


@app.command()
def repl(file: Path):
    """Interactive REPL for browsing questions."""
    questions = load_yaml(file)
    console.print("[bold green]Entering EvalQuestion REPL[/bold green]. Type 'help'.")

    while True:
        cmd = input("eval> ").strip()
        if cmd in {"quit", "exit"}:
            break

        parts = cmd.split()
        if not parts:
            continue

        if parts[0] in ["help", "-h", "--help"]:
            console.print(
                "Commands:\n"
                "  list [-s] [-t TAG]          List questions\n"
                "  search [-s] [-t TAG] WORD   Search by keyword\n"
                "  exit                        Quit REPL"
            )

        elif parts[0] == "list":
            show_solutions = "-s" in parts
            tag = None
            if "-t" in parts:
                idx = parts.index("-t")
                if idx + 1 < len(parts):
                    tag = parts[idx + 1]
            qs = questions
            if tag:
                qs = [q for q in qs if tag in (q.tags or [])]
            render_table(qs, show_solutions=show_solutions)

        elif parts[0] == "search":
            show_solutions = "-s" in parts
            tag = None
            words = []
            i = 1
            while i < len(parts):
                if parts[i] == "-s":
                    i += 1
                elif parts[i] == "-t" and i + 1 < len(parts):
                    tag = parts[i + 1]
                    i += 2
                else:
                    words.append(parts[i])
                    i += 1

            if not words:
                console.print("[red]Usage:[/red] search [-s] [-t TAG] WORD")
                continue

            keywords = [w.lower() for w in words]
            qs = questions
            if tag:
                qs = [q for q in qs if tag in (q.tags or [])]

            qs = [
                q
                for q in qs
                if any(
                    kw in q.question.lower() or kw in q.name.lower() for kw in keywords
                )
            ]

            render_table(qs, show_solutions=show_solutions)

        else:
            console.print(f"[red]Unknown command:[/red] {cmd}")


if __name__ == "__main__":
    app()
