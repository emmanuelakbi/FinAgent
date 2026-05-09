"""
Unit tests for task factory functions in crew/tasks.py.

Tests verify:
- Each task factory returns a Task with correct agent assignment
- create_risk_task has context containing the technical task
- create_strategy_task has context containing all 4 predecessor tasks
- Task descriptions include the ticker symbol

**Validates: Requirements 7.1, 7.2, 7.3**
"""

from unittest.mock import MagicMock

import pytest

# Shared crewai mock classes live in _crewai_mocks.py at the project
# root — import them here so we inspect / reset the SAME Task class that
# crew.tasks imports.
from _crewai_mocks import MOCK_TASK_CLS as mock_task_cls

from crew.tasks import (
    create_fundamental_task,
    create_market_scan_task,
    create_risk_task,
    create_strategy_task,
    create_technical_task,
)


@pytest.fixture(autouse=True)
def reset_mock_task():
    """Reset the mock Task class before each test so call args are clean."""
    mock_task_cls.reset_mock()
    # Make Task() return a new MagicMock each time
    mock_task_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    yield


@pytest.fixture
def fake_agent():
    """Create a fake agent instance."""
    agent = MagicMock()
    agent.role = "Test Agent"
    return agent


class TestCreateMarketScanTask:
    """Tests for create_market_scan_task factory."""

    def test_agent_is_passed_correctly(self, fake_agent):
        """Task is created with the correct agent assignment."""
        task = create_market_scan_task(fake_agent, "AAPL")
        # The Task was called with agent=fake_agent
        mock_task_cls.assert_called_once()
        call_kwargs = mock_task_cls.call_args[1]
        assert call_kwargs["agent"] is fake_agent

    def test_description_contains_ticker(self, fake_agent):
        """Task description includes the ticker symbol."""
        create_market_scan_task(fake_agent, "AAPL")
        call_kwargs = mock_task_cls.call_args[1]
        assert "AAPL" in call_kwargs["description"]

    def test_expected_output_contains_ticker(self, fake_agent):
        """Task expected_output includes the ticker symbol."""
        create_market_scan_task(fake_agent, "AAPL")
        call_kwargs = mock_task_cls.call_args[1]
        assert "AAPL" in call_kwargs["expected_output"]


class TestCreateFundamentalTask:
    """Tests for create_fundamental_task factory."""

    def test_agent_is_passed_correctly(self, fake_agent):
        """Task is created with the correct agent assignment."""
        task = create_fundamental_task(fake_agent, "MSFT")
        mock_task_cls.assert_called_once()
        call_kwargs = mock_task_cls.call_args[1]
        assert call_kwargs["agent"] is fake_agent

    def test_description_contains_ticker(self, fake_agent):
        """Task description includes the ticker symbol."""
        create_fundamental_task(fake_agent, "MSFT")
        call_kwargs = mock_task_cls.call_args[1]
        assert "MSFT" in call_kwargs["description"]

    def test_expected_output_contains_ticker(self, fake_agent):
        """Task expected_output includes the ticker symbol."""
        create_fundamental_task(fake_agent, "MSFT")
        call_kwargs = mock_task_cls.call_args[1]
        assert "MSFT" in call_kwargs["expected_output"]


class TestCreateTechnicalTask:
    """Tests for create_technical_task factory."""

    def test_agent_is_passed_correctly(self, fake_agent):
        """Task is created with the correct agent assignment."""
        task = create_technical_task(fake_agent, "GOOGL")
        mock_task_cls.assert_called_once()
        call_kwargs = mock_task_cls.call_args[1]
        assert call_kwargs["agent"] is fake_agent

    def test_description_contains_ticker(self, fake_agent):
        """Task description includes the ticker symbol."""
        create_technical_task(fake_agent, "GOOGL")
        call_kwargs = mock_task_cls.call_args[1]
        assert "GOOGL" in call_kwargs["description"]

    def test_expected_output_contains_ticker(self, fake_agent):
        """Task expected_output includes the ticker symbol."""
        create_technical_task(fake_agent, "GOOGL")
        call_kwargs = mock_task_cls.call_args[1]
        assert "GOOGL" in call_kwargs["expected_output"]


class TestCreateRiskTask:
    """Tests for create_risk_task factory."""

    def test_agent_is_passed_correctly(self, fake_agent):
        """Task is created with the correct agent assignment."""
        tech_task = MagicMock(name="technical_task")
        create_risk_task(fake_agent, "TSLA", [tech_task])
        mock_task_cls.assert_called_once()
        call_kwargs = mock_task_cls.call_args[1]
        assert call_kwargs["agent"] is fake_agent

    def test_context_contains_technical_task(self, fake_agent):
        """Risk task context contains the technical task dependency."""
        tech_task = MagicMock(name="technical_task")
        create_risk_task(fake_agent, "TSLA", [tech_task])
        call_kwargs = mock_task_cls.call_args[1]
        assert tech_task in call_kwargs["context"]

    def test_description_contains_ticker(self, fake_agent):
        """Task description includes the ticker symbol."""
        tech_task = MagicMock(name="technical_task")
        create_risk_task(fake_agent, "TSLA", [tech_task])
        call_kwargs = mock_task_cls.call_args[1]
        assert "TSLA" in call_kwargs["description"]

    def test_expected_output_contains_ticker(self, fake_agent):
        """Task expected_output includes the ticker symbol."""
        tech_task = MagicMock(name="technical_task")
        create_risk_task(fake_agent, "TSLA", [tech_task])
        call_kwargs = mock_task_cls.call_args[1]
        assert "TSLA" in call_kwargs["expected_output"]


class TestCreateStrategyTask:
    """Tests for create_strategy_task factory."""

    def test_agent_is_passed_correctly(self, fake_agent):
        """Task is created with the correct agent assignment."""
        t1 = MagicMock(name="market_task")
        t2 = MagicMock(name="fundamental_task")
        t3 = MagicMock(name="technical_task")
        t4 = MagicMock(name="risk_task")
        create_strategy_task(fake_agent, "NVDA", [t1, t2, t3, t4])
        mock_task_cls.assert_called_once()
        call_kwargs = mock_task_cls.call_args[1]
        assert call_kwargs["agent"] is fake_agent

    def test_context_contains_all_four_predecessor_tasks(self, fake_agent):
        """Strategy task context contains all 4 predecessor tasks."""
        t1 = MagicMock(name="market_task")
        t2 = MagicMock(name="fundamental_task")
        t3 = MagicMock(name="technical_task")
        t4 = MagicMock(name="risk_task")
        create_strategy_task(fake_agent, "NVDA", [t1, t2, t3, t4])
        call_kwargs = mock_task_cls.call_args[1]
        context = call_kwargs["context"]
        assert t1 in context
        assert t2 in context
        assert t3 in context
        assert t4 in context
        assert len(context) == 4

    def test_description_contains_ticker(self, fake_agent):
        """Task description includes the ticker symbol."""
        t1 = MagicMock(name="market_task")
        t2 = MagicMock(name="fundamental_task")
        t3 = MagicMock(name="technical_task")
        t4 = MagicMock(name="risk_task")
        create_strategy_task(fake_agent, "NVDA", [t1, t2, t3, t4])
        call_kwargs = mock_task_cls.call_args[1]
        assert "NVDA" in call_kwargs["description"]

    def test_description_contains_output_format_spec(self, fake_agent):
        """Strategy task description includes the output format specification."""
        t1 = MagicMock(name="market_task")
        t2 = MagicMock(name="fundamental_task")
        t3 = MagicMock(name="technical_task")
        t4 = MagicMock(name="risk_task")
        create_strategy_task(fake_agent, "NVDA", [t1, t2, t3, t4])
        call_kwargs = mock_task_cls.call_args[1]
        description = call_kwargs["description"]
        # The format spec includes these key elements
        assert "NVDA" in description
        assert "ACTION" in description or "BUY" in description
        assert "Confidence" in description
        assert "Entry" in description
        assert "Stop Loss" in description
        assert "Target" in description
