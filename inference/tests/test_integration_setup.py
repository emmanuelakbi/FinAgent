"""Integration tests for full inference setup on MI300X hardware.

These tests are designed to run on actual MI300X hardware with ROCm installed.
They validate the end-to-end setup flow including GPU detection, PyTorch access,
and health check verification.

Run with:
    cd inference/tests && python -m pytest test_integration_setup.py -v -m integration

Validates: Requirements 7.1, 7.2, 2.2, 2.4
"""

import os
import subprocess
import sys

import pytest

# Skip all tests in this module if ROCm is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.path.exists("/opt/rocm"),
        reason="Requires MI300X hardware with ROCm",
    ),
]

# Path to the inference scripts directory (one level up from tests/)
INFERENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETUP_SCRIPT = os.path.join(INFERENCE_DIR, "setup.sh")
HEALTH_CHECK_SCRIPT = os.path.join(INFERENCE_DIR, "health_check.sh")

# 30 minutes in seconds
SETUP_TIMEOUT_SECONDS = 30 * 60


class TestFullSetupOnMI300X:
    """Integration tests verifying the complete setup flow on MI300X hardware."""

    def test_full_setup_completes_within_30_minutes(self):
        """Run setup.sh on a fresh instance and verify it completes within 30 minutes.

        Validates: Requirements 7.1, 7.2
        - THE Setup_Script SHALL automate the full installation sequence
        - THE Setup_Script SHALL complete the full setup in under 30 minutes
        """
        result = subprocess.run(
            ["bash", SETUP_SCRIPT],
            cwd=INFERENCE_DIR,
            capture_output=True,
            text=True,
            timeout=SETUP_TIMEOUT_SECONDS,
        )

        assert result.returncode == 0, (
            f"setup.sh failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    def test_rocm_smi_detects_mi300x(self):
        """Verify rocm-smi detects MI300X GPU after setup.

        Validates: Requirement 2.2
        - THE MI300X_Instance SHALL report the MI300X GPU as detected
          when rocm-smi is executed
        """
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"rocm-smi --showid failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

        # Verify output contains GPU device information
        output = result.stdout.lower()
        assert "gpu" in output or "device" in output or "card" in output, (
            f"rocm-smi --showid did not report GPU device info.\n"
            f"Output:\n{result.stdout}"
        )

    def test_pytorch_gpu_available(self):
        """Verify torch.cuda.is_available() returns True after setup.

        Validates: Requirement 2.4
        - THE MI300X_Instance SHALL confirm GPU availability via
          torch.cuda.is_available() returning True
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; assert torch.cuda.is_available(), "
                "'torch.cuda.is_available() returned False'",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"PyTorch GPU check failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_health_check_passes_after_setup(self):
        """Verify health_check.sh passes after setup completes.

        Validates: Requirements 7.1, 7.2
        - WHEN the Setup_Script completes successfully, THE Setup_Script SHALL
          run a Health_Check that returns a valid chat completion response
        """
        result = subprocess.run(
            ["bash", HEALTH_CHECK_SCRIPT],
            cwd=INFERENCE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"health_check.sh failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify the health check reports success with latency
        assert "health check passed" in result.stdout.lower(), (
            f"Health check output did not contain success message.\n"
            f"Output: {result.stdout}"
        )
