"""
Root conftest.py for FinAgent tests.

Provides mocks for dependencies that may not be fully installable in the
test environment (e.g., crewai on Python 3.13, pandas_ta module naming).

The crewai mock is installed at session-start on ``sys.modules['crewai']``
so that any test that imports a ``crew.*`` module picks up the SAME shared
Task / Agent / Crew / Process mocks. This avoids a subtle mock-interference
bug where each test file used to patch ``sys.modules['crewai']`` locally
with its own mock classes: the first test file to import ``crew.tasks``
won — every subsequent test file's local mocks became disconnected from
the already-cached ``crew.tasks.Task`` / ``crew.crew.Crew`` references.

The shared mock classes live in ``_crewai_mocks.py`` at the project root
rather than inside this ``conftest.py``. A bare ``from conftest import
MOCK_TASK_CLS`` in a test module under ``tests/`` can resolve to a
sibling ``conftest.py`` (e.g. ``inference/tests/conftest.py``) when
multiple test directories land on ``sys.path``. The unique module name
avoids that collision.
"""

import os
import sys
from unittest.mock import MagicMock
from types import ModuleType

# Ensure the project root (this file's directory) is on ``sys.path`` so
# ``from _crewai_mocks import ...`` below resolves even when pytest is
# invoked from a subdirectory (e.g., ``inference/tests/``). Without
# this, pytest auto-discovers this conftest but can't locate sibling
# helpers living alongside it.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from _crewai_mocks import (
    MOCK_TASK_CLS,
    MOCK_AGENT_CLS,
    MOCK_CREW_CLS,
    MOCK_PROCESS,
)


def _mock_tool_decorator(name=None):
    """A no-op decorator that mimics crewai's @tool decorator for testing."""
    def decorator(func):
        return func
    if callable(name):
        # @tool used without arguments
        return name
    return decorator


# --- Shared crewai mock -----------------------------------------------------
# Install the pre-built MagicMock classes from ``_crewai_mocks`` on
# ``sys.modules['crewai']`` so that ``from crewai import Task, Agent,
# Crew, Process`` inside ``crew.*`` modules resolves to the same mocks
# tests use for assertions.

mock_crewai = MagicMock(name="crewai")
mock_crewai.Task = MOCK_TASK_CLS
mock_crewai.Agent = MOCK_AGENT_CLS
mock_crewai.Crew = MOCK_CREW_CLS
mock_crewai.Process = MOCK_PROCESS

mock_crewai_tools = ModuleType("crewai.tools")
mock_crewai_tools.tool = _mock_tool_decorator

sys.modules["crewai"] = mock_crewai
sys.modules["crewai.tools"] = mock_crewai_tools


# Mock pandas_ta / pandas_ta_remake if neither is importable so unit tests
# (which patch `pandas_ta_remake.rsi` / `.atr` etc.) can still collect and run.
# pandas-ta-remake is the maintained fork; it publishes under the name
# `pandas_ta_remake`. Upstream `pandas_ta` is the original name. Tools import
# the remake first and fall back to the original.
try:
    import pandas_ta_remake  # noqa: F401
except (ImportError, ModuleNotFoundError):
    try:
        import pandas_ta  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        mock_pandas_ta = MagicMock()
        sys.modules.setdefault("pandas_ta_remake", mock_pandas_ta)
        sys.modules.setdefault("pandas_ta", mock_pandas_ta)
