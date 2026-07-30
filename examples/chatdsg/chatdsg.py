#!/usr/bin/env python3
import argparse
import logging
import os
import threading

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import extract_labelspaces_from_dsg, load_dsg_to_db
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Label, RichLog, Rule, Static, TextArea

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

logger = logging.getLogger(__name__)

# Location of the backend DSG inside a saved map directory (see $ADT4_PRIOR_MAP).
PRIOR_DSG_RELPATH = os.path.join("hydra", "backend", "dsg.json")


def default_scene_graph_path():
    """Path to the prior DSG implied by $ADT4_PRIOR_MAP, or None if it is unset."""
    prior_map = os.getenv("ADT4_PRIOR_MAP")
    if not prior_map:
        return None

    return os.path.join(os.path.expanduser(prior_map), PRIOR_DSG_RELPATH)


def neo4j_credentials():
    """Neo4j credentials, preferring HERACLES_* over the launch system's ADT4_*."""
    user = os.getenv("HERACLES_NEO4J_USERNAME") or os.getenv("ADT4_NEO4J_USERNAME")
    password = os.getenv("HERACLES_NEO4J_PASSWORD") or os.getenv("ADT4_NEO4J_PASSWORD")
    return user, password


def load_prior_dsg(dsg_filepath, neo4j_uri):
    """Load a DSG from file into Neo4j using the labelspaces embedded in the graph.

    Returns True if the graph was loaded, False if it was skipped.
    """
    if not os.path.isfile(dsg_filepath):
        logger.warning(
            f"No DSG at '{dsg_filepath}'; skipping load. "
            "The database will only contain whatever the heracles publisher adds."
        )
        return False

    neo4j_creds = neo4j_credentials()
    if not all(neo4j_creds):
        logger.warning(
            "Neo4j credentials are not set (need HERACLES_NEO4J_USERNAME/PASSWORD or "
            "ADT4_NEO4J_USERNAME/PASSWORD); skipping DSG load."
        )
        return False

    logger.info(f"Loading DSG into database from filepath: {dsg_filepath}")
    scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
    summarize_dsg(scene_graph)

    # load_dsg_to_db reads the labelspaces out of the graph metadata, so a graph
    # saved without them will come back with unlabeled objects and rooms.
    object_labelspace, room_labelspace = extract_labelspaces_from_dsg(scene_graph)
    missing = [
        name
        for name, labelspace in (
            ("object", object_labelspace),
            ("room", room_labelspace),
        )
        if not labelspace
    ]
    if missing:
        logger.warning(
            f"DSG '{dsg_filepath}' has no embedded {' or '.join(missing)} labelspace; "
            "those nodes will be loaded without semantic labels."
        )

    load_dsg_to_db(neo4j_uri, neo4j_creds, scene_graph)
    logger.info("DSG loaded!")
    return True


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent: LlmAgent):
    prompt = agent.agent_info.prompt_settings.base_prompt
    return prompt


class MyTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+b", "submit", "Submit text"),
    ]

    def action_submit(self) -> None:
        self.app.action_submit()


class InputDisplayApp(App):
    def __init__(self, agent):
        self.agent = agent
        self.messages = generate_initial_prompt(agent).to_openai_json(
            "Now you will interact with the user:"
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield VerticalScroll(
            Label("Agent Chat:"),
            Rule(),
            RichLog(highlight=True, markup=True, wrap=True),
            Rule(line_style="thick"),
            Label("Enter text below:"),
            Rule(),
            MyTextArea("text", id="text_area"),
            Footer(id="footer"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""

        input_widget = self.query_one("#input_box", Input)
        # Clear the input box
        input_widget.value = ""

        self.display_text = f"You entered: {event.value}"
        self.query_one("#display_panel", Static).update(self.display_text)
        text_log = self.query_one(RichLog)
        text_log.write(event.value)

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        input_text_box = self.query_one("#text_area", MyTextArea)
        input_text = input_text_box.text
        text_log = self.query_one(RichLog)
        formatted_text = f"[bold black on white]User:[/] {input_text}"
        text_log.write(formatted_text)
        text_log.write("")
        input_text_box.text = ""

        self.messages += new_user_message(input_text)
        initial_length = len(self.messages)

        cxt = AgentContext(self.agent)
        cxt.history = self.messages

        def run_agent():
            success, answer = cxt.run()
            responses = cxt.get_agent_responses()
            for r in responses[initial_length:]:
                text_log.write(r.parsed_response)
                text_log.write("")

        thread = threading.Thread(target=run_agent)
        thread.start()

        # success, answer = cxt.run()
        # responses = cxt.get_agent_responses()
        # for r in responses[initial_length:]:
        #    text_log.write(r.parsed_response)
        #    text_log.write("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("ChatDSG agent")
    parser.add_argument(
        "--scene-graph",
        nargs="?",
        const=None,
        default=None,
        help=f"DSG filepath to load (defaults to $ADT4_PRIOR_MAP/{PRIOR_DSG_RELPATH})",
    )
    parser.add_argument(
        "--no-dsg-load",
        action="store_true",
        help="Don't load a DSG on startup (loading clears the existing database)",
    )
    parser.add_argument("--db_ip", type=str, help="Heracles database ip")
    parser.add_argument("--db_port", type=int, help="Heracles database ip")
    args = parser.parse_args()

    if args.db_ip is None:
        args.db_ip = os.getenv("ADT4_HERACLES_IP")

    if args.db_port is None:
        args.db_port = os.getenv("ADT4_HERACLES_PORT")

    dsg_filepath = args.scene_graph or default_scene_graph_path()
    if args.no_dsg_load:
        logger.info("Skipping DSG load (--no-dsg-load).")
    elif dsg_filepath is None:
        logger.warning(
            "No DSG to load: pass --scene-graph or set $ADT4_PRIOR_MAP to a saved "
            "map directory."
        )
    else:
        load_prior_dsg(dsg_filepath, f"neo4j://{args.db_ip}:{args.db_port}")

    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = InputDisplayApp(agent)
    app.run()
