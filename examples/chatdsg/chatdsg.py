#!/usr/bin/env python3
import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
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
CHATDSG_RETRY_WAIT_S = 2
CHATDSG_DEFAULT_LLM_TIMEOUT_S = 120


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent: LlmAgent):
    prompt = agent.agent_info.prompt_settings.base_prompt
    return prompt


def configure_pddl_mode(agent: LlmAgent, enable_pddl: bool):
    pddl_tool_names = [
        name for name in agent.agent_info.tools.keys() if "pddl" in name.lower()
    ]
    output_type = agent.agent_info.prompt_settings.output_type

    if enable_pddl:
        logger.info(
            "PDDL mode enabled via CLI. output_type=%s pddl_tools=%s",
            output_type,
            pddl_tool_names,
        )
        return

    for tool_name in pddl_tool_names:
        agent.agent_info.tools.pop(tool_name, None)

    if output_type in {"PDDL", "PDDL_TOOL"}:
        agent.agent_info.prompt_settings.output_type = None

    base_prompt = agent.agent_info.prompt_settings.base_prompt
    if (
        base_prompt.domain_description
        and "pddl" in base_prompt.domain_description.lower()
    ):
        base_prompt.domain_description = None
    if (
        base_prompt.answer_formatting_guidance
        and "pddl" in base_prompt.answer_formatting_guidance.lower()
    ):
        base_prompt.answer_formatting_guidance = None

    logger.info(
        "PDDL mode disabled via CLI. Removed tools=%s output_type=%s domain_description=%s answer_formatting_guidance=%s",
        pddl_tool_names,
        agent.agent_info.prompt_settings.output_type,
        bool(base_prompt.domain_description),
        bool(base_prompt.answer_formatting_guidance),
    )


@dataclass
class _Session:
    messages: list
    log_entries: list = field(default_factory=list)


class MyTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+b", "submit", "Submit text"),
    ]

    def action_submit(self) -> None:
        self.app.action_submit()


class InputDisplayApp(App):
    BINDINGS = [
        Binding("ctrl+x", "quit", "Quit"),
        Binding("ctrl+n", "new_session", "+ New Session"),
        Binding("left", "prev_session", "← Session"),
        Binding("right", "next_session", "→ Session"),
    ]

    def __init__(self, agent):
        self.agent = agent
        self._initial_messages = generate_initial_prompt(agent).to_openai_json(
            "Now you will interact with the user:"
        )
        self._sessions: list[_Session] = [
            _Session(messages=list(self._initial_messages))
        ]
        self._current_idx = 0
        super().__init__()

    @property
    def _current_session(self) -> _Session:
        return self._sessions[self._current_idx]

    @property
    def messages(self) -> list:
        return self._current_session.messages

    @messages.setter
    def messages(self, value):
        self._current_session.messages = value

    def _write_log(self, text, session: _Session | None = None):
        """Append *text* to a session's log and, if that session is active, display it."""
        target = session or self._current_session
        target.log_entries.append(text)
        if target is self._current_session:
            self.query_one(RichLog).write(text)

    def _update_session_label(self):
        total = len(self._sessions)
        current = self._current_idx + 1
        self.query_one("#session_label", Label).update(
            f"Session {current}/{total}"
        )

    def _switch_to_session(self, idx: int):
        if idx < 0 or idx >= len(self._sessions):
            return
        self._current_idx = idx
        text_log = self.query_one(RichLog)
        text_log.clear()
        for entry in self._current_session.log_entries:
            text_log.write(entry)
        self._update_session_label()
        self.query_one("#text_area", MyTextArea).text = ""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield VerticalScroll(
            Label("Session 1/1", id="session_label"),
            Label("Agent Chat:"),
            Rule(),
            RichLog(highlight=True, markup=True, wrap=True),
            Rule(line_style="thick"),
            Label("Enter text below:"),
            Rule(),
            MyTextArea("text", id="text_area"),
            Footer(id="footer"),
        )

    def action_new_session(self) -> None:
        """Create a fresh session and switch to it."""
        new = _Session(messages=list(self._initial_messages))
        self._sessions.append(new)
        self._current_idx = len(self._sessions) - 1
        text_log = self.query_one(RichLog)
        text_log.clear()
        self._write_log("[bold green]— New session started —[/]")
        self._write_log("")
        self._update_session_label()
        self.query_one("#text_area", MyTextArea).text = ""

    def action_prev_session(self) -> None:
        if self._current_idx > 0:
            self._switch_to_session(self._current_idx - 1)

    def action_next_session(self) -> None:
        if self._current_idx < len(self._sessions) - 1:
            self._switch_to_session(self._current_idx + 1)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""

        input_widget = self.query_one("#input_box", Input)
        input_widget.value = ""

        self.display_text = f"You entered: {event.value}"
        self.query_one("#display_panel", Static).update(self.display_text)
        text_log = self.query_one(RichLog)
        text_log.write(event.value)

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        input_text_box = self.query_one("#text_area", MyTextArea)
        input_text = input_text_box.text.strip()
        if input_text.lower() in ("exit", "/exit", "quit", "/quit"):
            self.exit()
            return

        session = self._current_session

        self._write_log(f"[bold black on white]User:[/] {input_text}")
        self._write_log("")
        input_text_box.text = ""

        session.messages += new_user_message(input_text)
        initial_length = len(session.messages)

        cxt = AgentContext(self.agent, retry_wait_s=CHATDSG_RETRY_WAIT_S)
        cxt.history = session.messages

        def run_agent():
            history_cursor = initial_length
            max_iter = self.agent.agent_info.max_iterations

            def on_tool_start(n, name, args):
                preview = (args[:80] + "...") if len(args) > 80 else args
                self._write_log(
                    f"[dim]  Tool #{n} [bold]{name}[/bold]({preview}) running...[/]",
                    session,
                )

            def on_tool_end(n, name, elapsed_ms, result):
                preview = str(result)[:120]
                self._write_log(
                    f"[dim]  Tool #{n} {name} done in {elapsed_ms:.0f} ms → {preview}[/]",
                    session,
                )

            for i in range(max_iter):
                self._write_log(
                    f"[dim]Step {i + 1}: calling LLM...[/]",
                    session,
                )
                t0 = time.perf_counter()
                try:
                    response = cxt.call_llm(cxt.history)
                except openai.PermissionDeniedError as exc:
                    self._write_log(
                        f"[bold red]Request blocked by guardrails (HTTP {exc.status_code}):[/] {exc.message}",
                        session,
                    )
                    logger.warning("Guardrail block: %s", exc)
                    return
                except Exception as exc:
                    self._write_log(
                        f"[bold red]LLM call failed:[/] {exc}", session
                    )
                    logger.exception("LLM call failed")
                    return

                llm_ms = (time.perf_counter() - t0) * 1000
                self._write_log(
                    f"[dim]  LLM responded in {llm_ms:.0f} ms[/]", session
                )

                try:
                    update = cxt.handle_response(
                        response,
                        on_tool_start=on_tool_start,
                        on_tool_end=on_tool_end,
                    )
                except Exception as exc:
                    self._write_log(
                        f"[bold red]Tool call failed:[/] {exc}", session
                    )
                    logger.exception("Tool call failed")
                    return

                cxt.update_history(response)
                cxt.update_history(update)

                done = cxt.check_if_done(cxt.history, response, update)

                responses = cxt.get_agent_responses()
                for r in responses[history_cursor:]:
                    if r.parsed_response:
                        self._write_log(r.parsed_response, session)
                        self._write_log("", session)
                history_cursor = len(cxt.history)

                if done:
                    self._write_log("[dim]Agent finished.[/]", session)
                    break
            else:
                self._write_log(
                    f"[dim yellow]Agent hit max iterations ({max_iter}).[/]",
                    session,
                )

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()


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
        "--object-labelspace",
        type=str,
        help="Path to object labelspace",
        default="ade20k_mit_label_space.yaml",
    )
    parser.add_argument(
        "--room-labelspace",
        type=str,
        help="Path to room labelspace",
        default="b45_label_space.yaml",
    )
    parser.add_argument("--db_ip", type=str, help="Heracles database ip")
    parser.add_argument("--db_port", type=int, help="Heracles database ip")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging to chatdsg_debug.log",
    )
    parser.add_argument(
        "--no-pddl",
        action="store_true",
        help="Disable PDDL mode for this run.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help=(
            "LLM HTTP timeout in seconds. "
            f"Defaults to {CHATDSG_DEFAULT_LLM_TIMEOUT_S}s for interactive chat."
        ),
    )
    args = parser.parse_args()

    # Ensure prompt YAML env-var paths resolve even if launcher didn't export it.
    os.environ.setdefault(
        "HERACLES_AGENTS_PATH",
        str(Path(__file__).resolve().parents[2]),
    )

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            filename="chatdsg_debug.log",
            filemode="w",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            filename="chatdsg_debug.log",
            filemode="w",
        )

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

    config_path = Path(__file__).resolve().with_name("agent_config.yaml")
    with open(config_path, "r") as fo:
        yml = yaml.safe_load(fo)
    prompt_settings = (
        yml.get("agent_info", {}).get("prompt_settings", {})
        if isinstance(yml, dict)
        else {}
    )
    base_prompt = prompt_settings.get("base_prompt")
    if isinstance(base_prompt, str):
        expanded_base_prompt = os.path.expandvars(base_prompt)
        if not os.path.isabs(expanded_base_prompt):
            prompt_settings["base_prompt"] = str(
                (config_path.parent / expanded_base_prompt).resolve()
            )
    if "client" in yml and isinstance(yml["client"], dict):
        configured_timeout = yml["client"].get("timeout")
        if args.llm_timeout is not None:
            yml["client"]["timeout"] = args.llm_timeout
            logger.info("Overriding LLM timeout via CLI to %ss", args.llm_timeout)
        elif configured_timeout is None or configured_timeout < CHATDSG_DEFAULT_LLM_TIMEOUT_S:
            # Interactive UI requests often exceed 20s due tool calls and backend queueing.
            yml["client"]["timeout"] = CHATDSG_DEFAULT_LLM_TIMEOUT_S
            logger.info(
                "Bumping LLM timeout from %s to %ss for interactive chat",
                configured_timeout,
                CHATDSG_DEFAULT_LLM_TIMEOUT_S,
            )
    agent = LlmAgent(**yml)
    if args.no_pddl:
        configure_pddl_mode(agent, enable_pddl=False)
    app = InputDisplayApp(agent)
    app.run()
