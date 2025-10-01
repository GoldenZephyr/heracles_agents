"""
Unit tests for experiment parsing functionality.

Tests the parsing of YAML experiment files into ExperimentDescription objects,
including validation, error handling, and environment variable expansion.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from heracles_agents.experiment_definition import (
    ExperimentDescription,
    PipelineDescription,
    PipelinePhase,
    PipelineRegistry,
    register_pipeline,
)
from heracles_agents.tools.answer_tool import answer_tool_desc
from heracles_agents.tools.calculator_tool import calculator_tool
from heracles_agents.tools.canary_favog_tool import favog_tool


# Helper functions for loading real experiment files
def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    # Go up 4 levels: tests -> heracles_agents -> src -> project_root
    return current_file.parent.parent.parent.parent


def load_experiment_file(file_path: Path) -> dict:
    """Load an experiment YAML file and return its contents."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_test_experiment_file() -> Path:
    """Get a specific experiment file for testing."""
    # Using canary_experiment.yaml as it has a simpler structure
    return get_project_root() / "examples" / "experiments" / "anthropic_test.yaml"


class TestExperimentParsing:
    """Test parsing of experiment YAML files into ExperimentDescription objects."""

    def setup_method(self):
        """Set up test fixtures and clear pipeline registry only."""
        PipelineRegistry.pipelines.clear()

        # Load real experiment files
        self.experiment_file = get_test_experiment_file()
        self.experiment_data = load_experiment_file(self.experiment_file)

        # Ensure required tools are registered (in case other tests cleared them)
        from heracles_agents.tool_registry import ToolRegistry, register_tool

        if "ask_favog" not in ToolRegistry.tools:
            register_tool(favog_tool)

        if "calculator" not in ToolRegistry.tools:
            register_tool(calculator_tool)

        if "answer" not in ToolRegistry.tools:
            register_tool(answer_tool_desc)

        # Create and register test pipelines
        self.mock_function = Mock()

        # Register canary pipeline
        test_pipeline = PipelineDescription(
            name="canary",
            description="Test canary pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=self.mock_function,
        )
        register_pipeline(test_pipeline)

        # Register agentic pipeline
        test_pipeline = PipelineDescription(
            name="agentic",
            description="Test agentic pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=self.mock_function,
        )
        register_pipeline(test_pipeline)

    def teardown_method(self):
        """Clear the pipeline registry after each test."""
        PipelineRegistry.pipelines.clear()

    def test_simple_parsing(self):
        """Test parsing a simple experiment YAML file."""
        # Load and parse anthropic test file which has the correct structure
        anthropic_file = (
            get_project_root() / "examples" / "experiments" / "anthropic_test.yaml"
        )
        experiment_data = load_experiment_file(anthropic_file)

        with (
            patch.dict(
                os.environ,
                {
                    "HERACLES_EVALUATION_PATH": str(get_project_root()),
                    "HERACLES_ANTHROPIC_API_KEY": "test-key",
                },
            ),
            patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ),
        ):
            ExperimentDescription(**experiment_data)

        """Test parsing an experiment from an actual YAML file."""
        # Load and parse canary experiment file
        canary_file = (
            get_project_root() / "examples" / "experiments" / "anthropic_test.yaml"
        )
        experiment_data = load_experiment_file(canary_file)

        with (
            patch.dict(
                os.environ,
                {
                    "HERACLES_EVALUATION_PATH": str(get_project_root()),
                    "HERACLES_OPENAI_API_KEY": "test-key",
                },
            ),
            patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ),
        ):
            ExperimentDescription(**experiment_data)

    def test_parse_experiment_with_env_vars(self):
        """Test parsing experiment YAML with environment variable expansion."""
        # Load and parse anthropic test file which has env vars
        anthropic_file = (
            get_project_root() / "examples" / "experiments" / "anthropic_test.yaml"
        )
        experiment_data = load_experiment_file(anthropic_file)

        # Set environment variables and test parsing
        with (
            patch.dict(
                os.environ,
                {
                    "HERACLES_EVALUATION_PATH": str(get_project_root()),
                    "HERACLES_ANTHROPIC_API_KEY": "test-key",
                },
            ),
            patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[{"name": "test_question"}],
            ),
        ):
            ExperimentDescription(**experiment_data)

    def test_parse_experiment_with_real_anthropic_structure(self):
        """Test parsing experiment that matches the anthropic_test.yaml structure."""
        # Load anthropic test experiment file
        anthropic_file = (
            get_project_root() / "examples" / "experiments" / "anthropic_test.yaml"
        )
        anthropic_data = load_experiment_file(anthropic_file)

        with (
            patch.dict(
                os.environ,
                {
                    "HERACLES_EVALUATION_PATH": str(get_project_root()),
                    "HERACLES_ANTHROPIC_API_KEY": "test-key",
                },
            ),
            patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ),
        ):
            ExperimentDescription(**anthropic_data)
