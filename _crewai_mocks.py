"""Shared crewai MagicMock classes for the FinAgent test suite.

This module lives at the project root (not under any ``tests/`` subpackage)
so that ``from _crewai_mocks import MOCK_TASK_CLS`` resolves unambiguously
regardless of which spec's tests are being collected.

The module publishes four shared MagicMock factories. ``conftest.py`` at
the repo root imports them and installs them on ``sys.modules['crewai']``
at session start, so any test that imports a ``crew.*`` module sees the
same ``Task`` / ``Agent`` / ``Crew`` / ``Process`` mocks. Tests that want
to reset or assert on the mocks should import them from here.

Why this lives outside ``conftest.py``: multiple sibling test directories
(``tests/``, ``inference/tests/``, ``gradio-frontend/tests/``) each get
added to ``sys.path`` during collection, so a bare ``from conftest
import ...`` in a test module can bind to the wrong ``conftest.py``. A
unique module name avoids the ambiguity entirely.
"""

from unittest.mock import MagicMock


def _make_task_factory(name: str) -> MagicMock:
    """Create a MagicMock class that returns a fresh MagicMock on each call."""
    mock_cls = MagicMock(name=name)
    mock_cls.side_effect = lambda **kwargs: MagicMock(
        name=f"{name}Instance", **kwargs
    )
    return mock_cls


MOCK_TASK_CLS = _make_task_factory("Task")
MOCK_AGENT_CLS = _make_task_factory("Agent")
MOCK_CREW_CLS = _make_task_factory("Crew")

MOCK_PROCESS = MagicMock(name="Process")
MOCK_PROCESS.sequential = "sequential"
