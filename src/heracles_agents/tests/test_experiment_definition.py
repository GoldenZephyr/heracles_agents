"""
Unit tests for experiment_definition module.

Tests all classes and functions in heracles_agents.experiment_definition:
- PipelinePhase
- PipelineDescription
- PipelineRegistry
- register_pipeline function
- ExperimentConfiguration
- ExperimentDescription
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from heracles_agents.experiment_definition import (
    ExperimentConfiguration,
    ExperimentDescription,
    PipelineDescription,
    PipelinePhase,
    PipelineRegistry,
    register_pipeline,
)


# Helper functions for loading real experiment files
def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    # Go up 4 levels: tests -> pypddl -> src -> project_root
    return current_file.parent.parent.parent.parent


def load_experiment_file(file_path: Path) -> dict:
    """Load an experiment YAML file and return its contents."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_test_experiment_file() -> Path:
    """Get a specific experiment file for testing."""
    # Using canary_experiment.yaml as it has a simpler structure
    return get_project_root() / "examples" / "experiments" / "canary_experiment.yaml"


class TestPipelineDescription:
    """Test the PipelineDescription class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_function = Mock(return_value="mock_result")
        self.phases = [
            PipelinePhase(name="phase1", description="First phase"),
            PipelinePhase(name="phase2", description="Second phase"),
        ]

    def test_validate_agent_phases_missing_phase(self):
        """Test validate_agent_phases with missing phase in configuration."""
        pipeline = PipelineDescription(
            name="test_pipeline",
            description="A test pipeline",
            phases=self.phases,
            function=self.mock_function,
        )

        # Mock experiment configuration missing phase2
        mock_config = Mock()
        mock_config.phases = {"phase1": Mock(), "other_phase": Mock()}

        # Should raise ValueError for missing phase
        with pytest.raises(ValueError) as exc_info:
            pipeline.validate_agent_phases(mock_config)

        assert "Pipeline test_pipeline requires agent phase phase2" in str(
            exc_info.value
        )
        assert "phase2 is not specified in the experiment configuration" in str(
            exc_info.value
        )


class TestRegisterPipeline:
    """Test the register_pipeline function."""

    def setup_method(self):
        """Set up test fixtures and clear registry."""
        PipelineRegistry.pipelines.clear()

        # Load real experiment files
        self.experiment_file = get_test_experiment_file()
        self.experiment_data = load_experiment_file(self.experiment_file)

        self.master_experiment_file = (
            get_project_root() / "examples" / "experiments" / "master_experiment.yaml"
        )
        self.master_experiment_data = load_experiment_file(self.master_experiment_file)

    def teardown_method(self):
        """Clear the registry after each test."""
        PipelineRegistry.pipelines.clear()

    def test_register_real_pipeline(self):
        """Test registering a real pipeline from canary experiment."""
        mock_function = Mock()
        pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )

        register_pipeline(pipeline)

        assert "canary" in PipelineRegistry.pipelines
        assert PipelineRegistry.pipelines["canary"] == pipeline

    def test_register_duplicate_real_pipeline(self, capsys):
        """Test registering a duplicate real pipeline."""
        mock_function = Mock()

        # Create two pipelines with same name from canary experiment
        pipeline1 = PipelineDescription(
            name="canary",
            description="First canary pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        pipeline2 = PipelineDescription(
            name="canary",
            description="Second canary pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )

        # Register first pipeline
        register_pipeline(pipeline1)
        assert PipelineRegistry.pipelines["canary"] == pipeline1

        # Register second pipeline with same name
        register_pipeline(pipeline2)

        # Should print warning but not replace
        captured = capsys.readouterr()
        assert "canary already has a registered function!" in captured.out
        assert PipelineRegistry.pipelines["canary"] == pipeline1

    def test_register_multiple_real_pipelines(self):
        """Test registering multiple real pipelines from master experiment."""
        mock_function = Mock()

        # Create and register canary pipeline
        canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        register_pipeline(canary_pipeline)

        # Create and register feedforward pipeline
        feedforward_pipeline = PipelineDescription(
            name="feedforward_cypher_qa",
            description="Feedforward Cypher QA pipeline",
            phases=[
                PipelinePhase(
                    name="generate-cypher", description="Generate Cypher query phase"
                ),
                PipelinePhase(name="refine", description="Refine results phase"),
            ],
            function=mock_function,
        )
        register_pipeline(feedforward_pipeline)

        assert "canary" in PipelineRegistry.pipelines
        assert "feedforward_cypher_qa" in PipelineRegistry.pipelines
        assert PipelineRegistry.pipelines["canary"] == canary_pipeline
        assert (
            PipelineRegistry.pipelines["feedforward_cypher_qa"] == feedforward_pipeline
        )


class TestExperimentConfiguration:
    """Test the ExperimentConfiguration class."""

    def setup_method(self):
        """Set up test fixtures and clear registry."""
        PipelineRegistry.pipelines.clear()

        # Load real experiment files
        self.experiment_file = get_test_experiment_file()
        self.experiment_data = load_experiment_file(self.experiment_file)

        self.master_experiment_file = (
            get_project_root() / "examples" / "experiments" / "master_experiment.yaml"
        )
        self.master_experiment_data = load_experiment_file(self.master_experiment_file)

        # Create and register canary pipeline
        self.mock_function = Mock()
        self.canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=self.mock_function,
        )
        register_pipeline(self.canary_pipeline)

        # Create mock agent
        self.mock_agent = Mock()

    def teardown_method(self):
        """Clear the registry after each test."""
        PipelineRegistry.pipelines.clear()

    def test_lookup_pipeline_by_name_from_real_config(self):
        """Test pipeline lookup by name using real experiment configuration."""
        config_data = {
            "pipeline": "canary",
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            config = ExperimentConfiguration(**config_data)

        assert config.pipeline == self.canary_pipeline

    def test_lookup_pipeline_by_object_from_real_config(self):
        """Test pipeline lookup when already a PipelineDescription object using real config."""
        config_data = {
            "pipeline": self.canary_pipeline,
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            config = ExperimentConfiguration(**config_data)

        assert config.pipeline == self.canary_pipeline

    def test_lookup_pipeline_not_found_with_real_config(self):
        """Test pipeline lookup with unknown pipeline name using real config structure."""
        config_data = {
            "pipeline": "unknown_pipeline",
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with pytest.raises(ValueError) as exc_info:
            with patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ):
                ExperimentConfiguration(**config_data)

        assert "Requested pipeline name unknown_pipeline not found" in str(
            exc_info.value
        )
        assert "canary" in str(exc_info.value)

    def test_load_questions_from_real_file(self):
        """Test loading questions from real experiment file."""
        # Get the questions file path from canary experiment
        questions_path = self.experiment_data["questions"]

        # Patch environment variable used in the path
        with patch.dict(
            os.environ, {"HERACLES_EVALUATION_PATH": str(get_project_root())}
        ):
            result = ExperimentConfiguration.load_questions(questions_path)
            assert len(result) > 0
            assert hasattr(result[0], "name")
            assert hasattr(result[0], "question")
            assert hasattr(result[0], "solution")

    def test_verify_pipeline_phases_success_with_real_config(self):
        """Test successful pipeline phase verification using real config."""
        config_data = {
            "pipeline": "canary",
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            config = ExperimentConfiguration(**config_data)
            assert config.pipeline == self.canary_pipeline

    def test_verify_pipeline_phases_failure_with_real_config(self):
        """Test pipeline phase verification failure using real config."""
        config_data = {
            "pipeline": "canary",
            "phases": {"wrong_phase": self.mock_agent},  # Missing main phase
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with pytest.raises(ValueError) as exc_info:
            with patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ):
                ExperimentConfiguration(**config_data)

        assert "Pipeline canary requires agent phase main" in str(exc_info.value)

    def test_serialize_pipeline_with_real_config(self):
        """Test pipeline serialization using real config."""
        config_data = {
            "pipeline": "canary",
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            config = ExperimentConfiguration(**config_data)

        serialized = config.serialize_pipeline(config.pipeline)
        assert serialized == "canary"

    def test_full_configuration_creation_with_real_config(self):
        """Test creating a complete ExperimentConfiguration using real config."""
        # Get the questions file path from canary experiment
        questions_path = self.experiment_data["questions"]

        # Create config data from real experiment
        config_data = {
            "pipeline": "canary",
            "phases": {"main": self.mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": questions_path,
        }

        # Patch environment variable used in the questions path
        with patch.dict(
            os.environ, {"HERACLES_EVALUATION_PATH": str(get_project_root())}
        ):
            config = ExperimentConfiguration(**config_data)

        assert config.pipeline == self.canary_pipeline
        assert config.phases == {"main": self.mock_agent}
        assert len(config.questions) > 0


class TestExperimentDescription:
    """Test the ExperimentDescription class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry and load real experiment files
        PipelineRegistry.pipelines.clear()

        self.experiment_file = get_test_experiment_file()
        self.experiment_data = load_experiment_file(self.experiment_file)

        self.master_experiment_file = (
            get_project_root() / "examples" / "experiments" / "master_experiment.yaml"
        )
        self.master_experiment_data = load_experiment_file(self.master_experiment_file)

        # Create and register canary pipeline
        mock_function = Mock()
        self.canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        register_pipeline(self.canary_pipeline)

        # Create real ExperimentConfiguration objects
        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            # Create config for canary experiment
            self.canary_config = ExperimentConfiguration(
                pipeline="canary",
                phases={"main": Mock()},
                dsg_interface=self.experiment_data["dsg_interface"],
                questions=[],
            )

            # Create config for canary-anthropic experiment
            self.canary_anthropic_config = ExperimentConfiguration(
                pipeline="canary",
                phases={"main": Mock()},
                dsg_interface=self.master_experiment_data["configurations"][
                    "canary-anthropic"
                ]["dsg_interface"],
                questions=[],
            )

    def teardown_method(self):
        """Clear the registry after each test."""
        PipelineRegistry.pipelines.clear()

    def test_experiment_description_creation_from_real_config(self):
        """Test ExperimentDescription creation using real experiment configuration."""
        metadata = self.master_experiment_data.get("metadata", {})
        configurations = {
            "canary": self.canary_config,
            "canary-anthropic": self.canary_anthropic_config,
        }

        experiment = ExperimentDescription(
            metadata=metadata, configurations=configurations
        )

        assert experiment.metadata == metadata
        assert experiment.configurations == configurations

    def test_experiment_description_serialization_with_real_config(self):
        """Test ExperimentDescription serialization using real config."""
        metadata = self.master_experiment_data.get("metadata", {})
        configurations = {
            "canary": self.canary_config,
            "canary-anthropic": self.canary_anthropic_config,
        }

        experiment = ExperimentDescription(
            metadata=metadata, configurations=configurations
        )

        # Test that it can be serialized (basic check)
        experiment_dict = experiment.model_dump()
        assert "metadata" in experiment_dict
        assert "configurations" in experiment_dict
        assert experiment_dict["metadata"] == metadata

    def test_experiment_description_from_real_dict(self):
        """Test ExperimentDescription creation from real experiment dict."""
        # Create experiment data based on real experiment structure
        experiment_data = {
            "metadata": self.master_experiment_data.get("metadata", {}),
            "configurations": {
                "canary": self.canary_config,
                "canary-anthropic": self.canary_anthropic_config,
            },
        }

        experiment = ExperimentDescription(**experiment_data)

        assert experiment.metadata == self.master_experiment_data.get("metadata", {})
        assert "canary" in experiment.configurations
        assert "canary-anthropic" in experiment.configurations
        assert experiment.configurations["canary"] == self.canary_config
        assert (
            experiment.configurations["canary-anthropic"]
            == self.canary_anthropic_config
        )


class TestIntegration:
    """Integration tests combining multiple components."""

    def setup_method(self):
        """Set up test fixtures and clear registry."""
        PipelineRegistry.pipelines.clear()

        # Load real experiment files
        self.experiment_file = get_test_experiment_file()
        self.experiment_data = load_experiment_file(self.experiment_file)

        self.master_experiment_file = (
            get_project_root() / "examples" / "experiments" / "master_experiment.yaml"
        )
        self.master_experiment_data = load_experiment_file(self.master_experiment_file)

    def teardown_method(self):
        """Clear the registry after each test."""
        PipelineRegistry.pipelines.clear()

    def test_full_workflow_with_real_config(self):
        """Test a complete workflow from pipeline registration to experiment creation using real config."""
        # 1. Create and register canary pipeline
        mock_function = Mock()
        canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        register_pipeline(canary_pipeline)

        # 2. Create experiment configuration
        mock_agent = Mock()

        # Create config data from real experiment
        config_data = {
            "pipeline": "canary",
            "phases": {"main": mock_agent},
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": self.experiment_data["questions"],
        }

        # Patch environment variable used in the questions path
        with patch.dict(
            os.environ, {"HERACLES_EVALUATION_PATH": str(get_project_root())}
        ):
            config = ExperimentConfiguration(**config_data)

        # 3. Create experiment description
        experiment = ExperimentDescription(
            metadata=self.master_experiment_data.get("metadata", {}),
            configurations={"canary": config},
        )

        # Verify everything is connected properly
        assert experiment.configurations["canary"].pipeline.name == "canary"
        assert len(experiment.configurations["canary"].phases) == 1
        assert "main" in experiment.configurations["canary"].phases
        assert experiment.configurations["canary"].phases["main"] == mock_agent

    def test_pipeline_validation_in_full_workflow_with_real_config(self):
        """Test that pipeline validation works in the full workflow using real config."""
        # Create and register canary pipeline
        mock_function = Mock()
        canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        register_pipeline(canary_pipeline)

        # Try to create config without required phase
        config_data = {
            "pipeline": "canary",
            "phases": {"wrong_phase": Mock()},  # Missing main phase
            "dsg_interface": self.experiment_data["dsg_interface"],
            "questions": [],
        }

        with pytest.raises(ValueError) as exc_info:
            with patch(
                "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
                return_value=[],
            ):
                ExperimentConfiguration(**config_data)

        assert "Pipeline canary requires agent phase main" in str(exc_info.value)

    def test_full_workflow_with_multiple_real_configs(self):
        """Test workflow with multiple real configurations from master experiment."""
        # 1. Create and register canary pipeline
        mock_function = Mock()
        canary_pipeline = PipelineDescription(
            name="canary",
            description="Canary experiment pipeline",
            phases=[PipelinePhase(name="main", description="Main phase")],
            function=mock_function,
        )
        register_pipeline(canary_pipeline)

        # 2. Create experiment configurations
        mock_agent = Mock()

        # Create configs for both canary and canary-anthropic
        with patch(
            "heracles_agents.experiment_definition.ExperimentConfiguration.load_questions",
            return_value=[],
        ):
            canary_config = ExperimentConfiguration(
                pipeline="canary",
                phases={"main": mock_agent},
                dsg_interface=self.experiment_data["dsg_interface"],
                questions=[],
            )

            canary_anthropic_config = ExperimentConfiguration(
                pipeline="canary",
                phases={"main": mock_agent},
                dsg_interface=self.master_experiment_data["configurations"][
                    "canary-anthropic"
                ]["dsg_interface"],
                questions=[],
            )

        # 3. Create experiment description with multiple configurations
        experiment = ExperimentDescription(
            metadata=self.master_experiment_data.get("metadata", {}),
            configurations={
                "canary": canary_config,
                "canary-anthropic": canary_anthropic_config,
            },
        )

        # Verify everything is connected properly
        assert len(experiment.configurations) == 2
        assert experiment.configurations["canary"].pipeline.name == "canary"
        assert experiment.configurations["canary-anthropic"].pipeline.name == "canary"
        assert len(experiment.configurations["canary"].phases) == 1
        assert len(experiment.configurations["canary-anthropic"].phases) == 1
        assert "main" in experiment.configurations["canary"].phases
        assert "main" in experiment.configurations["canary-anthropic"].phases
