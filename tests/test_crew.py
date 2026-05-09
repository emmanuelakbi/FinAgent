"""
Unit tests for FinAgentCrew in crew/crew.py.

Tests verify:
- _build_crew creates correct number of agents and tasks
- _parse_output delegates to TradingSignalParser correctly
- run returns CrewResult with success=False and error message on exception
- CrewAI Crew execution error handling paths

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**
"""

from unittest.mock import MagicMock, patch

import pytest

# Shared crewai mock classes live in _crewai_mocks.py at the project
# root — import them here so we inspect / reset the SAME classes that
# crew.crew imports.
from _crewai_mocks import (
    MOCK_CREW_CLS as mock_crew_cls,
    MOCK_AGENT_CLS as mock_agent_cls,
    MOCK_TASK_CLS as mock_task_cls,
    MOCK_PROCESS as mock_process,
)

from crew.crew import CrewResult, FinAgentCrew
from crew.config import OrchestratorConfig
from crew.signals import TradingSignal, Action


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    mock_crew_cls.reset_mock()
    mock_agent_cls.reset_mock()
    mock_task_cls.reset_mock()
    # Make Agent() and Task() return new MagicMock instances each time
    mock_agent_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    mock_task_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    # Make Crew() return a MagicMock with a kickoff method
    mock_crew_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    yield


@pytest.fixture
def config():
    """Create a default OrchestratorConfig."""
    return OrchestratorConfig()


@pytest.fixture
def crew_instance(config):
    """Create a FinAgentCrew with default config and empty tools."""
    return FinAgentCrew(config=config, tools={})


class TestParseOutput:
    """Tests for FinAgentCrew._parse_output."""

    def test_valid_primary_format_returns_trading_signal(self, crew_instance):
        """A valid primary format string should return a TradingSignal."""
        raw_output = "AAPL — BUY (Confidence: 75%)\nEntry: $185.42\nStop Loss: $180.55\nTarget: $192.17"
        result = crew_instance._parse_output(raw_output, "AAPL")

        assert result is not None
        assert isinstance(result, TradingSignal)
        assert result.ticker == "AAPL"
        assert result.action == Action.BUY
        assert result.confidence == 75

    def test_unparseable_string_returns_none(self, crew_instance):
        """An unparseable string with no action keywords should return None."""
        raw_output = "This is just random gibberish with no trading signals whatsoever."
        result = crew_instance._parse_output(raw_output, "AAPL")

        assert result is None


class TestRunSuccess:
    """Tests for FinAgentCrew.run success path."""

    def test_run_success_returns_crew_result_with_signal(self, config):
        """When crew.kickoff() returns a valid signal string, run returns success."""
        crew_obj = FinAgentCrew(config=config, tools={})

        valid_output = "AAPL — BUY (Confidence: 80%)\nEntry: $150.00\nStop Loss: $145.00\nTarget: $160.00"

        # Mock the Crew instance returned by _build_crew
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = valid_output

        with patch.object(crew_obj, "_build_crew", return_value=mock_crew_instance):
            result = crew_obj.run("AAPL")

        assert isinstance(result, CrewResult)
        assert result.success is True
        assert result.ticker == "AAPL"
        assert result.signal is not None
        assert result.signal.action == Action.BUY
        assert result.signal.confidence == 80
        assert result.error is None


class TestRunFailure:
    """Tests for FinAgentCrew.run failure paths."""

    def test_run_exception_returns_crew_result_with_error(self, config):
        """When crew.kickoff() raises an exception, run returns success=False with error."""
        crew_obj = FinAgentCrew(config=config, tools={})

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.side_effect = RuntimeError("LLM connection timeout")

        with patch.object(crew_obj, "_build_crew", return_value=mock_crew_instance):
            result = crew_obj.run("AAPL")

        assert isinstance(result, CrewResult)
        assert result.success is False
        assert result.ticker == "AAPL"
        assert result.signal is None
        assert "LLM connection timeout" in result.error

    def test_run_unparseable_output_returns_failure(self, config):
        """When crew.kickoff() returns gibberish, run returns success=False with parsing error."""
        crew_obj = FinAgentCrew(config=config, tools={})

        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = "completely random text with no signals"

        with patch.object(crew_obj, "_build_crew", return_value=mock_crew_instance):
            result = crew_obj.run("AAPL")

        assert isinstance(result, CrewResult)
        assert result.success is False
        assert result.ticker == "AAPL"
        assert result.signal is None
        assert result.error is not None
        assert "parse" in result.error.lower() or "pars" in result.error.lower()


class TestBuildCrew:
    """Tests for FinAgentCrew._build_crew."""

    def test_build_crew_creates_five_agents_and_five_tasks(self, config):
        """_build_crew should create a Crew with 5 agents and 5 tasks."""
        crew_obj = FinAgentCrew(config=config, tools={})
        crew_obj._build_crew("AAPL")

        # Crew() should have been called once
        mock_crew_cls.assert_called_once()
        call_kwargs = mock_crew_cls.call_args[1]

        assert len(call_kwargs["agents"]) == 5
        assert len(call_kwargs["tasks"]) == 5
        assert call_kwargs["process"] == mock_process.sequential
