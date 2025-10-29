#!/usr/bin/env python3
import argparse
import logging
import os
import threading

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Label, RichLog, Rule, Static, TextArea

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser("ChatDSG agent")
    parser.add_argument(
        "--scene-graph",
        nargs="?",
        const=None,
        default=None,
        help="DSG Filepath to load",
    )
    parser.add_argument(
        "--object-labelspace", type=str, help="Path to object labelspace"
    )
    parser.add_argument("--room-labelspace", type=str, help="Path to room labelspace")
    parser.add_argument("--db_ip", type=str, help="Heracles database ip")
    parser.add_argument("--db_port", type=int, help="Heracles database ip")
    args = parser.parse_args()

    if args.db_ip is None:
        args.db_ip = os.getenv("ADT4_HERACLES_IP")

    if args.db_port is None:
        args.db_port = os.getenv("ADT4_HERACLES_PORT")

    if args.scene_graph:
        dsg_filepath = args.scene_graph
        logger.info(f"Loading DSG into database from filepath: {dsg_filepath}")

        scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
        summarize_dsg(scene_graph)
        neo4j_uri = f"neo4j://{args.db_ip}:{args.db_port}"
        neo4j_creds = (
            os.getenv("HERACLES_NEO4J_USERNAME"),
            os.getenv("HERACLES_NEO4J_PASSWORD"),
        )

        load_dsg_to_db(
            args.object_labelspace,
            args.room_labelspace,
            neo4j_uri,
            neo4j_creds,
            scene_graph,
        )
        logger.info("DSG loaded!")

    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = InputDisplayApp(agent)
    app.run()
