#!/usr/bin/env python3
"""ChatDSG with plan repair tool support."""

# Register the repair tool before the agent loads
# Import the app class from the original chatdsg
import importlib.util
import os

import yaml

import heracles_agents.tools.pddl_repair_tool  # noqa: F401
from heracles_agents.llm_agent import LlmAgent

original = os.path.join(os.path.dirname(__file__), "..", "chatdsg", "chatdsg.py")
spec = importlib.util.spec_from_file_location("chatdsg_original", original)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = mod.InputDisplayApp(agent)
    app.run()
