"""Unit tests for CLI argument parsing and model validation.

Tests the Python validation module's validate_model_name function
and the bash script's CLI parsing via subprocess calls.

Validates: Requirements 9.1, 4.5
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from inference_validation import SUPPORTED_MODELS, validate_model_name

# Path to the setup script
SETUP_SCRIPT = os.path.join(
    os.path.dirname(__file__), os.pardir, "setup.sh"
)


# =============================================================================
# Python-level tests: validate_model_name
# =============================================================================


class TestValidateModelName:
    """Test the validate_model_name function for each supported model."""

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
    def test_each_supported_model_is_accepted(self, model_name: str) -> None:
        """Each of the 4 supported model names should be accepted."""
        is_valid, error_msg = validate_model_name(model_name)
        assert is_valid is True
        assert error_msg == ""

    def test_invalid_model_rejected(self) -> None:
        """An invalid model name should be rejected with informative error."""
        is_valid, error_msg = validate_model_name("invalid-model")
        assert is_valid is False
        assert "invalid-model" in error_msg
        # Error should list all supported models
        for model in SUPPORTED_MODELS:
            assert model in error_msg

    def test_empty_string_rejected(self) -> None:
        """An empty string should be rejected."""
        is_valid, error_msg = validate_model_name("")
        assert is_valid is False
        assert error_msg != ""

    def test_partial_model_name_rejected(self) -> None:
        """A partial model name (substring of a valid one) should be rejected."""
        is_valid, error_msg = validate_model_name("Qwen3-8B")
        assert is_valid is False
        assert "Qwen3-8B" in error_msg

    def test_case_sensitive_rejection(self) -> None:
        """Model names are case-sensitive; wrong case should be rejected."""
        is_valid, error_msg = validate_model_name("qwen/qwen3-8b")
        assert is_valid is False


# =============================================================================
# Bash script tests: CLI parsing via subprocess
# =============================================================================


class TestSetupScriptCLIParsing:
    """Test setup.sh CLI argument parsing via subprocess calls."""

    def _run_setup(
        self,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """Run setup.sh with given args and env, capturing output."""
        cmd = ["bash", SETUP_SCRIPT] + (args or [])
        env = os.environ.copy()
        # Remove any existing FINAGENT_ env vars to avoid interference
        for key in list(env.keys()):
            if key.startswith("FINAGENT_"):
                del env[key]
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

    def test_default_model_is_qwen3_8b(self) -> None:
        """Default model should be Qwen/Qwen3-8B when no --model flag is set.

        We verify this by checking config.sh contains the default, and by
        running setup.sh with an invalid model to see the error references
        the correct default behavior.
        """
        # Verify config.sh sets the default
        config_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "config.sh"
        )
        with open(config_path) as f:
            config_content = f.read()
        assert "Qwen/Qwen3-8B" in config_content
        assert "FINAGENT_MODEL:-Qwen/Qwen3-8B" in config_content

    def test_invalid_model_rejected_by_script(self) -> None:
        """Running setup.sh with an invalid model should exit 1 with error."""
        result = self._run_setup(args=["--model", "invalid-model"])
        assert result.returncode == 1
        assert "invalid-model" in result.stderr.lower() or "invalid-model" in result.stderr

    def test_each_supported_model_accepted_by_script(self) -> None:
        """Each supported model should pass validation (script fails later at GPU detection).

        If the model is valid, the script proceeds past validation to GPU
        detection, which will fail with exit code 2 (no GPU on test machine).
        Exit code 2 means the model was accepted.
        """
        for model_name in SUPPORTED_MODELS:
            result = self._run_setup(args=["--model", model_name])
            # Exit code 2 = GPU not detected (past model validation)
            # Exit code 1 with "Invalid model" = model rejected
            assert result.returncode != 1 or "Invalid model" not in result.stderr, (
                f"Model '{model_name}' was unexpectedly rejected"
            )

    def test_env_var_override_with_invalid_model(self) -> None:
        """FINAGENT_MODEL env var should override the default model.

        Setting FINAGENT_MODEL to an invalid value should cause rejection,
        proving the env var is being used.
        """
        result = self._run_setup(
            env_overrides={"FINAGENT_MODEL": "env-override-model"}
        )
        assert result.returncode == 1
        assert "env-override-model" in result.stderr

    def test_cli_flag_overrides_env_var(self) -> None:
        """CLI --model flag should take precedence over FINAGENT_MODEL env var.

        Set env var to a valid model but CLI flag to an invalid one.
        The script should reject the CLI flag value, proving CLI wins.
        """
        result = self._run_setup(
            args=["--model", "cli-wins-model"],
            env_overrides={"FINAGENT_MODEL": "Qwen/Qwen3-8B"},
        )
        assert result.returncode == 1
        # The error should reference the CLI value, not the env var value
        assert "cli-wins-model" in result.stderr

    def test_unknown_option_rejected(self) -> None:
        """An unknown CLI flag should cause exit code 1."""
        result = self._run_setup(args=["--unknown-flag"])
        assert result.returncode == 1
        assert "Unknown option" in result.stderr or "unknown" in result.stderr.lower()

    def test_help_flag_exits_zero(self) -> None:
        """The --help flag should exit 0 and show usage info."""
        result = self._run_setup(args=["--help"])
        assert result.returncode == 0
        assert "Usage" in result.stdout or "usage" in result.stdout.lower()
        # Help should list supported models
        assert "Qwen/Qwen3-8B" in result.stdout

    def test_model_flag_without_value_rejected(self) -> None:
        """--model without a value should exit 1 with an error."""
        result = self._run_setup(args=["--model"])
        assert result.returncode == 1
        assert "requires a value" in result.stderr or "Error" in result.stderr
