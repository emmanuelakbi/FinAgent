"""Unit tests for exit code behavior of setup.sh and health_check.sh.

Validates that scripts return correct exit codes for various failure modes.
These tests run the actual bash scripts via subprocess and verify exit codes.

Requirements: 7.7
"""

import os
import subprocess

import pytest

# Path to the inference directory containing the scripts
INFERENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETUP_SCRIPT = os.path.join(INFERENCE_DIR, "setup.sh")
HEALTH_CHECK_SCRIPT = os.path.join(INFERENCE_DIR, "health_check.sh")
CONFIG_SCRIPT = os.path.join(INFERENCE_DIR, "config.sh")


class TestExitCodeConstants:
    """Test that exit code constants in config.sh match the documented values."""

    def test_exit_success_is_zero(self):
        """EXIT_SUCCESS should be 0."""
        value = self._get_config_value("EXIT_SUCCESS")
        assert value == "0", f"EXIT_SUCCESS should be 0, got {value}"

    def test_exit_general_failure_is_one(self):
        """EXIT_GENERAL_FAILURE should be 1."""
        value = self._get_config_value("EXIT_GENERAL_FAILURE")
        assert value == "1", f"EXIT_GENERAL_FAILURE should be 1, got {value}"

    def test_exit_gpu_not_detected_is_two(self):
        """EXIT_GPU_NOT_DETECTED should be 2."""
        value = self._get_config_value("EXIT_GPU_NOT_DETECTED")
        assert value == "2", f"EXIT_GPU_NOT_DETECTED should be 2, got {value}"

    def test_exit_oom_is_three(self):
        """EXIT_OOM should be 3."""
        value = self._get_config_value("EXIT_OOM")
        assert value == "3", f"EXIT_OOM should be 3, got {value}"

    def test_exit_health_check_failure_is_four(self):
        """EXIT_HEALTH_CHECK_FAILURE should be 4."""
        value = self._get_config_value("EXIT_HEALTH_CHECK_FAILURE")
        assert value == "4", f"EXIT_HEALTH_CHECK_FAILURE should be 4, got {value}"

    def test_exit_timeout_is_five(self):
        """EXIT_TIMEOUT should be 5."""
        value = self._get_config_value("EXIT_TIMEOUT")
        assert value == "5", f"EXIT_TIMEOUT should be 5, got {value}"

    def _get_config_value(self, variable_name: str) -> str:
        """Extract a variable value from config.sh by sourcing it."""
        result = subprocess.run(
            ["bash", "-c", f'source "{CONFIG_SCRIPT}" && echo "${{{variable_name}}}"'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Failed to source config.sh: {result.stderr}"
        )
        return result.stdout.strip()


class TestSetupExitCodes:
    """Test exit codes from setup.sh under various conditions."""

    def test_setup_help_exits_zero(self):
        """setup.sh --help should exit with code 0."""
        result = subprocess.run(
            ["bash", SETUP_SCRIPT, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Expected exit code 0 for --help, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )

    def test_setup_invalid_model_exits_one(self):
        """setup.sh --model invalid-model should exit with code 1."""
        result = subprocess.run(
            ["bash", SETUP_SCRIPT, "--model", "invalid-model"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1, (
            f"Expected exit code 1 for invalid model, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )
        # Error message should mention the invalid model name
        assert "invalid-model" in result.stderr, (
            "Error message should contain the invalid model name"
        )

    def test_setup_invalid_model_lists_supported(self):
        """setup.sh with invalid model should list supported models in error."""
        result = subprocess.run(
            ["bash", SETUP_SCRIPT, "--model", "not-a-real-model"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        # Should list supported models in stderr
        assert "Qwen/Qwen3-8B" in result.stderr, (
            "Error should list supported models"
        )
        assert "Qwen/Qwen3-14B" in result.stderr, (
            "Error should list supported models"
        )

    def test_setup_gpu_not_detected_exits_two(self):
        """setup.sh should exit with code 2 when GPU is not detected.

        On a machine without rocm-smi, the script should exit with code 2.
        """
        # Check if rocm-smi is available on this machine
        rocm_check = subprocess.run(
            ["which", "rocm-smi"],
            capture_output=True,
            text=True,
        )
        if rocm_check.returncode == 0:
            pytest.skip("rocm-smi is available on this machine; cannot test GPU-not-detected path")

        result = subprocess.run(
            ["bash", SETUP_SCRIPT, "--model", "Qwen/Qwen3-8B"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2, (
            f"Expected exit code 2 (GPU not detected), got {result.returncode}. "
            f"stderr: {result.stderr}"
        )


class TestHealthCheckExitCodes:
    """Test exit codes from health_check.sh under various conditions."""

    def test_health_check_connection_refused_exits_one(self):
        """health_check.sh should exit 1 when connection is refused.

        Using a port that is almost certainly not running a vLLM server.
        """
        result = subprocess.run(
            ["bash", HEALTH_CHECK_SCRIPT, "--port", "19999", "--host", "127.0.0.1"],
            capture_output=True,
            text=True,
            timeout=35,
        )
        assert result.returncode == 1, (
            f"Expected exit code 1 (connection refused), got {result.returncode}. "
            f"stderr: {result.stderr}"
        )
        assert "Connection refused" in result.stderr or "refused" in result.stderr.lower(), (
            f"Error message should mention connection refused. Got: {result.stderr}"
        )

    def test_health_check_timeout_exits_nonzero(self):
        """health_check.sh should exit with non-zero code on timeout.

        Using a non-routable IP address (10.255.255.1) with a short timeout
        to trigger a timeout condition. The exit code should be 1 or 2
        depending on whether curl reports it as a timeout (28) or other error.
        """
        result = subprocess.run(
            ["bash", HEALTH_CHECK_SCRIPT, "--timeout", "2", "--host", "10.255.255.1", "--port", "8000"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Should be non-zero (either 1 for network unreachable or 2 for timeout)
        assert result.returncode in (1, 2), (
            f"Expected exit code 1 or 2 on timeout/unreachable, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )


class TestExitCodeDocumentation:
    """Test that exit code documentation in scripts matches config.sh constants."""

    def test_setup_script_documents_exit_codes(self):
        """setup.sh header should document all exit codes."""
        with open(SETUP_SCRIPT, "r") as f:
            content = f.read()

        # Verify documented exit codes in the script header
        assert "0 - Success" in content or "0 -" in content, (
            "setup.sh should document exit code 0"
        )
        assert "2 - GPU not detected" in content or "2 -" in content, (
            "setup.sh should document exit code 2"
        )
        assert "3 - Model load failure" in content or "3 -" in content, (
            "setup.sh should document exit code 3"
        )
        assert "4 - Health check failure" in content or "4 -" in content, (
            "setup.sh should document exit code 4"
        )
        assert "5 - Timeout" in content or "5 -" in content, (
            "setup.sh should document exit code 5"
        )

    def test_health_check_script_documents_exit_codes(self):
        """health_check.sh header should document all exit codes."""
        with open(HEALTH_CHECK_SCRIPT, "r") as f:
            content = f.read()

        # Verify documented exit codes in the script header
        assert "0 - Success" in content or "0 -" in content, (
            "health_check.sh should document exit code 0"
        )
        assert "1 - Connection error" in content or "1 -" in content, (
            "health_check.sh should document exit code 1"
        )
        assert "2 - Timeout" in content or "2 -" in content, (
            "health_check.sh should document exit code 2"
        )
        assert "3 - Non-200" in content or "3 -" in content, (
            "health_check.sh should document exit code 3"
        )
        assert "4 - Empty response" in content or "4 -" in content, (
            "health_check.sh should document exit code 4"
        )
