"""
Unit tests for WatchlistRunner in crew/runner.py.

Tests verify:
- Fault isolation: a failed ticker does not prevent subsequent tickers from running
- Callbacks: on_ticker_start and on_ticker_complete are emitted for each ticker
- WatchlistResult counts match actual outcomes
- _parse_watchlist correctly normalizes and filters ticker strings

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**
"""

from unittest.mock import MagicMock, patch

import pytest

# Shared crewai mock classes live in _crewai_mocks.py at the project root.
from _crewai_mocks import (
    MOCK_CREW_CLS as mock_crew_cls,
    MOCK_AGENT_CLS as mock_agent_cls,
    MOCK_TASK_CLS as mock_task_cls,
    MOCK_PROCESS as mock_process,
)

from crew.runner import WatchlistRunner, WatchlistResult
from crew.crew import CrewResult
from crew.config import OrchestratorConfig
from crew.callbacks import ActivityFeedCallback, EventType


@pytest.fixture
def config():
    """Create a default OrchestratorConfig."""
    return OrchestratorConfig()


@pytest.fixture
def tools():
    """Create an empty tools dict."""
    return {}


class TestFaultIsolation:
    """Test that a failed ticker does not prevent subsequent tickers from running.

    Validates: Requirement 10.3
    """

    def test_failed_ticker_does_not_block_subsequent(self, config, tools):
        """First ticker fails, second succeeds — both are processed."""
        runner = WatchlistRunner(config=config, tools=tools)

        fail_result = CrewResult(
            ticker="AAPL",
            signal=None,
            raw_output="",
            success=False,
            error="LLM timeout",
        )
        success_result = CrewResult(
            ticker="MSFT",
            signal=MagicMock(),
            raw_output="MSFT — BUY (Confidence: 80%)",
            success=True,
        )

        with patch.object(
            runner, "_run_single", side_effect=[fail_result, success_result]
        ):
            result = runner.run("AAPL, MSFT")

        assert result.total_tickers == 2
        assert result.failed == 1
        assert result.successful == 1
        assert len(result.signals) == 2
        assert result.signals[0].success is False
        assert result.signals[1].success is True


class TestCallbacks:
    """Test that callbacks are emitted for ticker start and complete.

    Validates: Requirement 10.4
    """

    def test_on_ticker_start_and_complete_called_for_each_ticker(self, config, tools):
        """on_ticker_start and on_ticker_complete are called for every ticker."""
        mock_handler = MagicMock()
        callback = ActivityFeedCallback(handler=mock_handler)

        runner = WatchlistRunner(config=config, tools=tools, callback=callback)

        crew_result = CrewResult(
            ticker="AAPL",
            signal=None,
            raw_output="",
            success=True,
        )

        with patch.object(runner, "_run_single", return_value=crew_result):
            runner.run("AAPL, MSFT")

        # The handler should have been called for start and complete of each ticker
        # 2 tickers × 2 events (start + complete) = 4 calls
        assert mock_handler.call_count == 4

        # Verify event types in order
        events = [call.args[0] for call in mock_handler.call_args_list]

        assert events[0].event_type == EventType.TICKER_START
        assert events[0].ticker == "AAPL"
        assert events[1].event_type == EventType.TICKER_COMPLETE
        assert events[1].ticker == "AAPL"
        assert events[2].event_type == EventType.TICKER_START
        assert events[2].ticker == "MSFT"
        assert events[3].event_type == EventType.TICKER_COMPLETE
        assert events[3].ticker == "MSFT"


class TestWatchlistResultCounts:
    """Test that WatchlistResult counts match actual outcomes.

    Validates: Requirements 10.1, 10.2, 10.3
    """

    def test_counts_match_with_mixed_results(self, config, tools):
        """Run with 3 tickers (2 success, 1 failure) — counts are correct."""
        runner = WatchlistRunner(config=config, tools=tools)

        results_sequence = [
            CrewResult(ticker="AAPL", signal=MagicMock(), raw_output="", success=True),
            CrewResult(ticker="MSFT", signal=None, raw_output="", success=False, error="timeout"),
            CrewResult(ticker="GOOGL", signal=MagicMock(), raw_output="", success=True),
        ]

        with patch.object(runner, "_run_single", side_effect=results_sequence):
            result = runner.run("AAPL, MSFT, GOOGL")

        assert result.total_tickers == 3
        assert result.successful == 2
        assert result.failed == 1
        assert len(result.signals) == 3


class TestParseWatchlist:
    """Test _parse_watchlist normalizes and filters ticker strings.

    Validates: Requirement 10.1
    """

    def test_basic_parsing_with_mixed_case(self, config, tools):
        """Comma-separated tickers are uppercased and stripped."""
        runner = WatchlistRunner(config=config, tools=tools)
        result = runner._parse_watchlist("AAPL, msft , GOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_empty_entries_removed(self, config, tools):
        """Empty entries from extra commas or whitespace-only segments are removed."""
        runner = WatchlistRunner(config=config, tools=tools)
        result = runner._parse_watchlist("  aapl  ,  ,  msft  ")
        assert result == ["AAPL", "MSFT"]
