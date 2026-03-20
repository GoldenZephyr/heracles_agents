#!/usr/bin/env python3
"""
Test several question across multiple models via agentgateway.

Runs the full heracles agent pipeline (system prompt, Cypher tools, agent loop)
for each model through agentgateway using the Chat Completions API, then prints
a comparison table.

All providers sit behind a single gateway port.  The openai_completions
client type uses /v1/chat/completions, which agentgateway supports for every
provider (OpenAI native, Anthropic translation, Gemini native, Ollama native).

Prerequisites:
  - agentgateway running: agentgateway -f config.yaml
  - Neo4j running with the scene graph loaded
  - Environment variables:
      OPENAI_API_KEY           — for agentgateway backend auth
      HERACLES_OPENAI_API_KEY  — for the heracles client (can be fake, agentgateway will handle auth)
      HERACLES_AGENTS_PATH     — repo root (for prompt YAML files)
  - pip install -e .   (editable install of this repo)
"""

import argparse
import logging
import os
import sys
import time

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console

import heracles_agents.provider_integrations.openai.openai_completions_agent_integration  # noqa: F401
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext, EvalQuestion
from heracles_agents.pipelines.agentic_pipeline import generate_prompt
from heracles_agents.pipelines.comparisons import evaluate_answer

logging.basicConfig(level="INFO", handlers=[RichHandler(level="INFO")])
logger = logging.getLogger(__name__)

console = Console()

# Model x Question-type sweep table
#
# +---------------------+---------+---------+
# |                     |   Q&A   |  PDDL   |
# |     Model           | Correct | Correct |
# +---------------------+---------+---------+
# | gpt-4.1             |    ?    |    ?    |
# | claude-sonnet-4     |    ?    |    ?    |
# | qwen2.5:14b         |    ?    |    ?    |
# +---------------------+---------+---------+

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_PATH = os.environ.get("HERACLES_AGENTS_PATH", _SCRIPT_DIR)

if not os.path.isdir(os.path.join(AGENTS_PATH, "examples", "chatdsg")):
    logger.warning(
        "HERACLES_AGENTS_PATH=%s does not contain examples/chatdsg; "
        "falling back to script directory %s",
        AGENTS_PATH,
        _SCRIPT_DIR,
    )
    AGENTS_PATH = _SCRIPT_DIR

AGW_GATEWAY = "http://localhost:3000/v1"

MODELS = [
    {"name": "gpt-4.1", "model": "gpt-4.1", "provider": "openai"},
    {"name": "claude-sonnet-4", "model": "claude-sonnet-4-20250514", "provider": "anthropic"},
    {"name": "claude-opus-4", "model": "claude-opus-4-20250514", "provider": "anthropic"},
    {"name": "gemini-2.5-flash-lite", "model": "gemini-2.5-flash-lite", "provider": "gemini"},
    {"name": "gemini-2.5-pro", "model": "gemini-2.5-pro", "provider": "gemini"},
    # {"name": "llama3.1", "model": "llama3.1", "provider": "ollama"},
    # {"name": "qwen2.5:14b", "model": "qwen2.5:14b", "provider": "ollama"},
    {"name": "qwen3:32b", "model": "qwen3:32b", "provider": "ollama"},
   # make sure to run "aws configure"
    {"name": "bedrock-claude-haiku", "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "provider": "bedrock"},
]

DEFAULT_QUESTIONS = {
    "qa": ["0A", "12A", "17B"],
    "pddl": ["0A", "22A", "31B"],
}

OUTPUT_TYPE_FOR = {
    "SLDP": "SLDP",
    "PDDL": "PDDL",
}


def load_questions(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)["questions"]


def find_question(questions, uid):
    return next((q for q in questions if str(q["uid"]) == uid), None)


def load_scene_graph(dsg_path, neo4j_uri, object_labelspace, room_labelspace):
    """Load a scene graph file into Neo4j (mirrors chatdsg.py behaviour)."""
    logger.info(f"Loading scene graph from {dsg_path} ...")
    scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_path)
    summarize_dsg(scene_graph)
    neo4j_creds = (
        os.getenv("HERACLES_NEO4J_USERNAME", "neo4j"),
        os.getenv("HERACLES_NEO4J_PASSWORD", "neo4j"),
    )
    load_dsg_to_db(
        object_labelspace,
        room_labelspace,
        neo4j_uri,
        neo4j_creds,
        scene_graph,
    )
    logger.info("Scene graph loaded!")


def make_agent_config(model_name, provider, gateway_url, neo4j_uri):
    return {
        "client": {
            "client_type": "openai_completions",
            "base_url": gateway_url,
            "timeout": 120,
            "default_headers": {"x-model-provider": provider},
        },
        "model_info": {
            "model": model_name,
            "temperature": 0.2,
        },
        "agent_info": {
            "prompt_settings": {
                "base_prompt": os.path.join(
                    AGENTS_PATH, "examples", "chatdsg", "agent_prompt.yaml"
                ),
                "output_type": "SLDP",
                "sldp_answer_type_hint": True,
            },
            "tools": [
                {
                    "name": "run_cypher_query",
                    "bound_args": {
                        "dsgdb_conf": {
                            "dsg_interface_type": "heracles",
                            "uri": neo4j_uri,
                        }
                    },
                }
            ],
            "tool_interface": "openai_completions",
            "max_iterations": 6,
        },
    }


def run_agent_for_question(model_cfg, question_dict, gateway_url, neo4j_uri):
    """Run the full heracles agent pipeline and return results."""
    config = make_agent_config(model_cfg["model"], model_cfg["provider"], gateway_url, neo4j_uri)

    comparison_type = question_dict["correctness_comparator"]["comparison_type"]
    config["agent_info"]["prompt_settings"]["output_type"] = OUTPUT_TYPE_FOR.get(
        comparison_type, "SLDP"
    )

    try:
        agent = LlmAgent(**config)
        eval_q = EvalQuestion(**question_dict)
        prompt = generate_prompt(eval_q, agent)

        cxt = AgentContext(agent)
        cxt.initialize_agent(prompt)

        start = time.time()
        success, answer = cxt.run()
        elapsed = time.time() - start

        valid_format, correct = evaluate_answer(
            eval_q.correctness_comparator, answer, eval_q.solution
        )

        return (
            str(answer) if answer else "(no answer)",
            correct,
            cxt.n_tool_calls,
            elapsed,
            None,
        )

    except Exception as e:
        logger.exception("Agent error")
        return None, False, 0, 0.0, str(e)


def build_results_table(title, rows):
    """Build a rich Table from result rows."""
    table = Table(title=title, show_lines=True)
    table.add_column("Model", justify="center", min_width=20)
    table.add_column("Answer", justify="center", min_width=40)
    table.add_column("Correct?", justify="center", min_width=9)
    table.add_column("Tools", justify="center", min_width=7)
    table.add_column("Time", justify="center", min_width=8)

    for row in rows:
        style = "green" if row[2] == "YES" else ("red" if row[2] == "no" else "dim")
        table.add_row(*row, style=style)

    return table


def get_questions(script_dir):
    qa_path = os.path.join(script_dir, "examples", "questions", "qa_questions.yaml")
    pddl_path = os.path.join(script_dir, "examples", "questions", "pddl_questions.yaml")
    return load_questions(qa_path), load_questions(pddl_path)


def get_test_cases(question_type, qa_questions, pddl_questions, uid_filter=None):
    test_cases = []
    if question_type in ("qa", "both"):
        uids = uid_filter if uid_filter else DEFAULT_QUESTIONS["qa"]
        for uid in uids:
            q = find_question(qa_questions, uid)
            if q:
                test_cases.append(("QA", q))
    if question_type in ("pddl", "both"):
        uids = uid_filter if uid_filter else DEFAULT_QUESTIONS["pddl"]
        for uid in uids:
            q = find_question(pddl_questions, uid)
            if q:
                test_cases.append(("PDDL", q))
    return test_cases


def select_models(names):
    model_index = {m["name"]: m for m in MODELS}
    selected = []
    for name in names:
        if name not in model_index:
            logger.error(f"Unknown model '{name}'. Choose from: {list(model_index.keys())}")
            sys.exit(1)
        selected.append(model_index[name])
    return selected


def main():
    all_model_names = [m["name"] for m in MODELS]

    parser = argparse.ArgumentParser(
        description="Test a hard question across multiple models via agentgateway"
    )
    parser.add_argument(
        "--gateway-url",
        default=AGW_GATEWAY,
        help="agentgateway base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=all_model_names,
        metavar="NAME",
        help=f"Models to test (choices: {', '.join(all_model_names)})",
    )
    parser.add_argument(
        "--question-type",
        choices=["qa", "pddl", "both"],
        default="qa",
        help="Which question set to test (default: qa)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        metavar="UID",
        help="Specific question UIDs to run (e.g. 0A 12A). Defaults to all for the chosen --question-type.",
    )
    parser.add_argument(
        "--scene-graph",
        default=None,
        help="DSG filepath to load into Neo4j before testing",
    )
    parser.add_argument(
        "--object-labelspace",
        default="ade20k_mit_label_space.yaml",
        help="Path to object labelspace (default: %(default)s)",
    )
    parser.add_argument(
        "--room-labelspace",
        default="b45_label_space.yaml",
        help="Path to room labelspace (default: %(default)s)",
    )
    parser.add_argument(
        "--db-ip",
        default=None,
        help="Neo4j host (default: $ADT4_HERACLES_IP or localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Neo4j port (default: $ADT4_HERACLES_PORT or 7687)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_ip = args.db_ip or os.getenv("ADT4_HERACLES_IP", "localhost")
    db_port = args.db_port or int(os.getenv("ADT4_HERACLES_PORT", "7687"))
    neo4j_uri = f"neo4j://{db_ip}:{db_port}"

    if args.scene_graph:
        load_scene_graph(
            args.scene_graph, neo4j_uri,
            args.object_labelspace, args.room_labelspace,
        )

    selected = select_models(args.models)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    qa_questions, pddl_questions = get_questions(script_dir)
    test_cases = get_test_cases(args.question_type, qa_questions, pddl_questions, args.questions)

    if not test_cases:
        logger.warning("No matching questions found.")
        return

    total_jobs = len(selected) * len(test_cases)
    logger.info(
        f"Running {total_jobs} jobs sequentially "
        f"({len(selected)} models x {len(test_cases)} questions)"
    )

    for q_type, question in test_cases:
        logger.info(
            f"[{q_type}] uid={question['uid']}  {question['name']}\n"
            f"  Q: {question['question']}\n"
            f"  Expected: {question['solution']}"
        )

    results: dict[tuple, tuple] = {}
    done_count = 0

    for q_type, question in test_cases:
        for model_cfg in selected:
            key = (q_type, question["uid"], model_cfg["name"])
            done_count += 1
            logger.info(f"[{done_count}/{total_jobs}] Running {model_cfg['name']} | {q_type}-{question['uid']} ...")
            answer, correct, n_tools, elapsed, error = run_agent_for_question(
                model_cfg, question, args.gateway_url, neo4j_uri,
            )
            results[key] = (answer, correct, n_tools, elapsed, error)
            status = f"ERROR: {error[:60]}" if error else (
                f"correct={correct}  tools={n_tools}  {elapsed:.1f}s"
            )
            logger.info(f"[{done_count}/{total_jobs}] {model_cfg['name']} | {q_type}-{question['uid']}: {status}")

    for q_type, question in test_cases:
        rows = []
        for model_cfg in selected:
            key = (q_type, question["uid"], model_cfg["name"])
            answer, correct, n_tools, elapsed, error = results[key]
            if error:
                rows.append(
                    [model_cfg["name"], f"ERROR: {error[:36]}", "--", "--", "--"]
                )
            else:
                mark = "YES" if correct else "no"
                truncated = answer if len(str(answer)) <= 40 else str(answer)[:37] + "..."
                rows.append([
                    model_cfg["name"],
                    truncated,
                    mark,
                    str(n_tools),
                    f"{elapsed:.1f}s",
                ])

        title = f"[{q_type}] uid={question['uid']}  {question['name']}"
        table = build_results_table(title, rows)
        console.print()
        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
