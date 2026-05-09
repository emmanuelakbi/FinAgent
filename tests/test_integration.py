"""
Integration tests for the full agent orchestration pipeline.

Tests verify end-to-end flow from config → agents → tasks → crew → runner
with mocked LLM responses, ensuring the pipeline produces correct results.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 10.2, 10.3, 11.1, 11.3**
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

from crew.config import OrchestratorConfig
from crew.crew import CrewResult, FinAgentCrew
from crew.runner import WatchlistRunner, WatchlistResult
from crew.callbacks import ActivityFeedCallback, ActivityEvent, EventType
from crew.signals import TradingSignal, Action


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test."""
    mock_crew_cls.reset_mock()
    mock_agent_cls.reset_mock()
    mock_task_cls.reset_mock()
    mock_agent_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    mock_task_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    mock_crew_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)
    yield


# --- Helper fixtures ---


@pytest.fixture
def config():
    """Create a default OrchestratorConfig."""
    return OrchestratorConfig()


@pytest.fixture
def tools():
    """Create an empty tools dict (tools are not needed for mocked execution)."""
    return {}


def make_valid_signal_output(ticker: str, action: str = "BUY", confidence: int = 75) -> str:
    """Generate a valid trading signal output string for a given ticker."""
    return (
        f"{ticker} — {action} (Confidence: {confidence}%)\n"
        f"Entry: $185.42\n"
        f"Stop Loss: $180.55\n"
        f"Target: $192.17\n"
        f"Reasoning:\n"
        f"- Market: Positive earnings surprise detected\n"
        f"- Fundamental: Undervalued relative to peers\n"
        f"- Technical: MACD bullish crossover\n"
        f"- Risk: 1:2 risk-reward ratio"
    )


# =============================================================================
# Test 1: End-to-end flow with mocked LLM
# =============================================================================


class TestEndToEndFlow:
    """Test the full pipeline: config → agents → tasks → crew → runner.

    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 8.1
    """

    def test_single_ticker_produces_valid_trading_signal(self, config, tools):
        """Running a single ticker through the pipeline produces a valid TradingSignal."""
        runner = WatchlistRunner(config=config, tools=tools)

        valid_output = make_valid_signal_output("AAPL")
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = valid_output

        # Patch _build_crew on FinAgentCrew to return our mocked crew
        with patch.object(FinAgentCrew, "_build_crew", return_value=mock_crew_instance):
            result = runner.run("AAPL")

        assert isinstance(result, WatchlistResult)
        assert result.total_tickers == 1
        assert result.successful == 1
        assert result.failed == 0
        assert len(result.signals) == 1

        crew_result = result.signals[0]
        assert isinstance(crew_result, CrewResult)
        assert crew_result.success is True
        assert crew_result.ticker == "AAPL"
        assert crew_result.signal is not None
        assert isinstance(crew_result.signal, TradingSignal)
        assert crew_result.signal.ticker == "AAPL"
        assert crew_result.signal.action == Action.BUY
        assert crew_result.signal.confidence == 75
        assert crew_result.signal.entry_price == 185.42
        assert crew_result.signal.stop_loss == 180.55
        assert crew_result.signal.target_price == 192.17

    def test_pipeline_uses_config_correctly(self, tools):
        """The pipeline passes config through to crew construction."""
        custom_config = OrchestratorConfig()
        custom_config.crew.verbose = False

        runner = WatchlistRunner(config=custom_config, tools=tools)

        valid_output = make_valid_signal_output("TSLA", "SELL", 60)
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = valid_output

        with patch.object(FinAgentCrew, "_build_crew", return_value=mock_crew_instance):
            result = runner.run("TSLA")

        assert result.successful == 1
        crew_result = result.signals[0]
        assert crew_result.signal.action == Action.SELL
        assert crew_result.signal.confidence == 60


# =============================================================================
# Test 2: Multi-ticker watchlist
# =============================================================================


class TestMultiTickerWatchlist:
    """Test that a multi-ticker watchlist produces correct WatchlistResult structure.

    Validates: Requirements 10.2, 10.3
    """

    def test_three_tickers_all_succeed(self, config, tools):
        """Running 3 tickers produces WatchlistResult with total=3, successful=3, failed=0."""
        runner = WatchlistRunner(config=config, tools=tools)

        # Each call to kickoff returns a signal for the respective ticker
        outputs = {
            "AAPL": make_valid_signal_output("AAPL", "BUY", 80),
            "MSFT": make_valid_signal_output("MSFT", "HOLD", 55),
            "GOOGL": make_valid_signal_output("GOOGL", "SELL", 70),
        }

        def mock_run_single(ticker: str) -> CrewResult:
            """Simulate FinAgentCrew.run for each ticker."""
            from crew.signals import TradingSignalParser

            parser = TradingSignalParser()
            raw = outputs[ticker]
            signal = parser.parse(raw, ticker)
            return CrewResult(
                ticker=ticker,
                signal=signal,
                raw_output=raw,
                success=True,
            )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            result = runner.run("AAPL, MSFT, GOOGL")

        assert isinstance(result, WatchlistResult)
        assert result.total_tickers == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.signals) == 3

        # Verify each signal has the correct ticker
        tickers_in_result = [r.ticker for r in result.signals]
        assert tickers_in_result == ["AAPL", "MSFT", "GOOGL"]

        # Verify each signal has the correct action
        assert result.signals[0].signal.action == Action.BUY
        assert result.signals[1].signal.action == Action.HOLD
        assert result.signals[2].signal.action == Action.SELL

    def test_multi_ticker_with_whitespace_and_case_normalization(self, config, tools):
        """Tickers with mixed case and extra whitespace are normalized correctly."""
        runner = WatchlistRunner(config=config, tools=tools)

        def mock_run_single(ticker: str) -> CrewResult:
            return CrewResult(
                ticker=ticker,
                signal=MagicMock(ticker=ticker),
                raw_output="",
                success=True,
            )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            result = runner.run("  aapl ,  Msft  , googl  ")

        assert result.total_tickers == 3
        # Verify tickers were normalized to uppercase
        assert result.signals[0].ticker == "AAPL"
        assert result.signals[1].ticker == "MSFT"
        assert result.signals[2].ticker == "GOOGL"


# =============================================================================
# Test 3: Callback event ordering
# =============================================================================


class TestCallbackEventOrdering:
    """Test that callback events are emitted in correct order.

    Expected order for a single ticker:
    TICKER_START → TASK_START (from Crew) → TASK_COMPLETE → TICKER_COMPLETE

    Validates: Requirements 11.1, 11.3
    """

    def test_single_ticker_event_order(self, config, tools):
        """Events are emitted in order: ticker_start → task_start → task_complete → ticker_complete."""
        events_log: list[ActivityEvent] = []

        def record_event(event: ActivityEvent):
            events_log.append(event)

        callback = ActivityFeedCallback(handler=record_event)
        runner = WatchlistRunner(config=config, tools=tools, callback=callback)

        valid_output = make_valid_signal_output("AAPL")
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = valid_output

        with patch.object(FinAgentCrew, "_build_crew", return_value=mock_crew_instance):
            result = runner.run("AAPL")

        assert result.successful == 1

        # Extract event types in order
        event_types = [e.event_type for e in events_log]

        # First event must be TICKER_START
        assert event_types[0] == EventType.TICKER_START

        # Last event must be TICKER_COMPLETE
        assert event_types[-1] == EventType.TICKER_COMPLETE

        # TASK_START should come after TICKER_START
        assert EventType.TASK_START in event_types
        task_start_idx = event_types.index(EventType.TASK_START)
        assert task_start_idx > 0  # After TICKER_START

        # TASK_COMPLETE should come after TASK_START and before TICKER_COMPLETE
        assert EventType.TASK_COMPLETE in event_types
        task_complete_idx = event_types.index(EventType.TASK_COMPLETE)
        assert task_complete_idx > task_start_idx
        assert task_complete_idx < len(event_types) - 1  # Before TICKER_COMPLETE

    def test_all_events_have_correct_ticker(self, config, tools):
        """All emitted events reference the correct ticker."""
        events_log: list[ActivityEvent] = []

        def record_event(event: ActivityEvent):
            events_log.append(event)

        callback = ActivityFeedCallback(handler=record_event)
        runner = WatchlistRunner(config=config, tools=tools, callback=callback)

        valid_output = make_valid_signal_output("MSFT", "HOLD", 50)
        mock_crew_instance = MagicMock()
        mock_crew_instance.kickoff.return_value = valid_output

        with patch.object(FinAgentCrew, "_build_crew", return_value=mock_crew_instance):
            runner.run("MSFT")

        # All events should reference MSFT
        for event in events_log:
            assert event.ticker == "MSFT"

    def test_multi_ticker_events_interleaved_correctly(self, config, tools):
        """For multiple tickers, each ticker's events are grouped together."""
        events_log: list[ActivityEvent] = []

        def record_event(event: ActivityEvent):
            events_log.append(event)

        callback = ActivityFeedCallback(handler=record_event)
        runner = WatchlistRunner(config=config, tools=tools, callback=callback)

        def mock_run_single(ticker: str) -> CrewResult:
            # Simulate the callback events that FinAgentCrew.run would emit
            callback.on_task_start("Crew", ticker)
            callback.on_task_complete("Chief Strategist", ticker, f"BUY (80%)")
            return CrewResult(
                ticker=ticker,
                signal=MagicMock(),
                raw_output="",
                success=True,
            )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            result = runner.run("AAPL, MSFT")

        # Verify events are grouped by ticker
        # Expected: AAPL_START, (AAPL crew events), AAPL_COMPLETE, MSFT_START, (MSFT crew events), MSFT_COMPLETE
        aapl_events = [e for e in events_log if e.ticker == "AAPL"]
        msft_events = [e for e in events_log if e.ticker == "MSFT"]

        # Each ticker should have at least start and complete
        aapl_types = [e.event_type for e in aapl_events]
        msft_types = [e.event_type for e in msft_events]

        assert aapl_types[0] == EventType.TICKER_START
        assert aapl_types[-1] == EventType.TICKER_COMPLETE
        assert msft_types[0] == EventType.TICKER_START
        assert msft_types[-1] == EventType.TICKER_COMPLETE

        # AAPL events should all come before MSFT events (sequential execution)
        last_aapl_idx = max(i for i, e in enumerate(events_log) if e.ticker == "AAPL")
        first_msft_idx = min(i for i, e in enumerate(events_log) if e.ticker == "MSFT")
        assert last_aapl_idx < first_msft_idx


# =============================================================================
# Test 4: Graceful degradation
# =============================================================================


class TestGracefulDegradation:
    """Test graceful degradation: one ticker fails, others succeed.

    Validates: Requirements 8.1, 8.3, 10.3
    """

    def test_one_ticker_fails_others_succeed(self, config, tools):
        """When one ticker raises an exception, others still produce valid results."""
        runner = WatchlistRunner(config=config, tools=tools)

        def mock_run_single(ticker: str) -> CrewResult:
            if ticker == "AAPL":
                # Simulate a failure (e.g., LLM timeout)
                return CrewResult(
                    ticker=ticker,
                    signal=None,
                    raw_output="",
                    success=False,
                    error="LLM connection timeout after 120s",
                )
            else:
                from crew.signals import TradingSignalParser

                parser = TradingSignalParser()
                raw = make_valid_signal_output(ticker, "BUY", 85)
                signal = parser.parse(raw, ticker)
                return CrewResult(
                    ticker=ticker,
                    signal=signal,
                    raw_output=raw,
                    success=True,
                )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            result = runner.run("AAPL, MSFT")

        assert result.total_tickers == 2
        assert result.successful == 1
        assert result.failed == 1

        # The failed ticker has an error message
        failed_result = result.signals[0]
        assert failed_result.ticker == "AAPL"
        assert failed_result.success is False
        assert failed_result.error is not None
        assert "timeout" in failed_result.error.lower()
        assert failed_result.signal is None

        # The successful ticker has a valid signal
        success_result = result.signals[1]
        assert success_result.ticker == "MSFT"
        assert success_result.success is True
        assert success_result.signal is not None
        assert success_result.signal.action == Action.BUY

    def test_exception_in_crew_run_is_caught(self, config, tools):
        """An exception raised during crew execution is caught and reported."""
        runner = WatchlistRunner(config=config, tools=tools)

        # Patch _build_crew to raise an exception for the first ticker
        call_count = [0]

        def mock_build_crew(self_crew, ticker):
            call_count[0] += 1
            if ticker == "FAIL":
                raise RuntimeError("Unexpected agent crash")
            mock_crew_instance = MagicMock()
            mock_crew_instance.kickoff.return_value = make_valid_signal_output(
                ticker, "HOLD", 50
            )
            return mock_crew_instance

        with patch.object(FinAgentCrew, "_build_crew", mock_build_crew):
            result = runner.run("FAIL, GOOD")

        assert result.total_tickers == 2
        assert result.failed == 1
        assert result.successful == 1

        # The failed ticker has the error message
        failed = result.signals[0]
        assert failed.success is False
        assert "Unexpected agent crash" in failed.error

        # The successful ticker completed normally
        success = result.signals[1]
        assert success.success is True
        assert success.ticker == "GOOD"

    def test_all_tickers_fail_gracefully(self, config, tools):
        """When all tickers fail, WatchlistResult reflects total failure without crashing."""
        runner = WatchlistRunner(config=config, tools=tools)

        def mock_run_single(ticker: str) -> CrewResult:
            return CrewResult(
                ticker=ticker,
                signal=None,
                raw_output="",
                success=False,
                error=f"Failed for {ticker}",
            )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            result = runner.run("AAPL, MSFT, GOOGL")

        assert result.total_tickers == 3
        assert result.successful == 0
        assert result.failed == 3
        for r in result.signals:
            assert r.success is False
            assert r.error is not None

    def test_failed_ticker_with_callbacks_still_emits_events(self, config, tools):
        """Even when a ticker fails, ticker_start and ticker_complete callbacks are emitted."""
        events_log: list[ActivityEvent] = []

        def record_event(event: ActivityEvent):
            events_log.append(event)

        callback = ActivityFeedCallback(handler=record_event)
        runner = WatchlistRunner(config=config, tools=tools, callback=callback)

        def mock_run_single(ticker: str) -> CrewResult:
            return CrewResult(
                ticker=ticker,
                signal=None,
                raw_output="",
                success=False,
                error="Simulated failure",
            )

        with patch.object(runner, "_run_single", side_effect=mock_run_single):
            runner.run("AAPL")

        event_types = [e.event_type for e in events_log]
        # Even on failure, TICKER_START and TICKER_COMPLETE should be emitted
        assert EventType.TICKER_START in event_types
        assert EventType.TICKER_COMPLETE in event_types
