"""Pytest configuration for the gradio-frontend test suite.

This file does two things:

1. **sys.path setup** — Adds the ``gradio-frontend`` directory to
   ``sys.path`` so that tests can import top-level modules such as
   ``validation``, ``rendering``, and ``app`` directly.

2. **Shared test fixtures** — Provides reusable building blocks for the
   integration tests added in task 11.1:

   * :class:`FakeAction` — enum-like stand-in for ``crew.signals.Action``
     with a ``.value`` attribute. ``rendering.py`` reads the action via
     ``getattr(action, "value", action)`` so a minimal class is enough
     to exercise the BUY/SELL/HOLD code paths without pulling in the
     real ``crew`` package.
   * :class:`FakeSignal` — dataclass mirroring the attributes of
     :class:`crew.signals.TradingSignal` read by ``render_signal_card``:
     ``ticker``, ``action``, ``confidence``, ``entry_price``,
     ``stop_loss``, ``target_price``, ``reasoning``.
   * :class:`FakeCrewResult` — dataclass shaped like the result object
     returned from ``WatchlistRunner._run_single``: ``success``,
     ``signal``, ``error``.
   * :class:`FakeActivityEvent` — dataclass mirroring
     :class:`crew.callbacks.ActivityEvent` attributes read by
     ``app.run_analysis`` when translating pending callback events into
     activity feed HTML: ``timestamp``, ``agent_name``, ``message``,
     ``event_type``.
   * :func:`make_fake_crew_package` — builds ``crew`` and
     ``crew.callbacks`` module stand-ins suitable for injection into
     ``sys.modules``. The fake :class:`WatchlistRunner`'s
     ``_run_single(ticker)`` looks the ticker up in a caller-supplied
     result dict, defaulting to a ``not configured`` failure. This gives
     integration tests full control over the success / partial-failure
     mix emitted by the pipeline without depending on the real
     ``crewai`` runtime.

   Plus the pytest fixtures ``buy_signal``, ``sell_signal``,
   ``hold_signal``, ``trading_signals``, and ``sample_activity_events``
   that produce canned instances of the dataclasses above.

All fixture names are public (no leading underscore) so test files can
request them by parameter name. The existing rendering / handler tests
continue to use their local underscore-prefixed helpers, so nothing here
shadows or breaks them.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
#
# The tests/ directory lives inside gradio-frontend/. Walk one level up so
# ``import validation`` resolves to gradio-frontend/validation.py.

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# FakeAction — enum-like stand-in for crew.signals.Action
# ---------------------------------------------------------------------------


class FakeAction:
    """Stand-in for an enum member with a ``.value`` string attribute.

    ``rendering.render_signal_card`` and
    ``app._render_signals_dashboard`` both read the action label via
    ``getattr(action, "value", action)``, which matches both a real
    :class:`enum.Enum` member (where ``.value`` is the string payload)
    and a plain string. Using a lightweight class here keeps the
    fixtures hermetic — no need to import ``crew.signals.Action``.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"FakeAction({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeAction):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


# ---------------------------------------------------------------------------
# FakeSignal — minimal TradingSignal stand-in
# ---------------------------------------------------------------------------


@dataclass
class FakeSignal:
    """Minimal stand-in for :class:`crew.signals.TradingSignal`.

    Mirrors the attribute surface read by ``render_signal_card`` and
    ``_render_signals_dashboard``. Optional price fields default to
    ``None`` so HOLD-style signals without explicit entry / stop / target
    prices render cleanly via the renderer's ``N/A`` fallback.
    """

    ticker: str
    action: FakeAction
    confidence: int
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    reasoning: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FakeCrewResult — stand-in for WatchlistRunner._run_single's return value
# ---------------------------------------------------------------------------


@dataclass
class FakeCrewResult:
    """Stand-in for the result object returned by ``_run_single``.

    ``app.run_analysis`` only reads three attributes off the result:
    ``success`` (bool), ``signal`` (a TradingSignal-shaped object or
    ``None``), and ``error`` (an optional message). Matching that shape
    exactly lets the fake runner return a value the real handler
    consumes without complaint.
    """

    success: bool
    signal: Optional[FakeSignal] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# FakeActivityEvent — stand-in for crew.callbacks.ActivityEvent
# ---------------------------------------------------------------------------


@dataclass
class FakeActivityEvent:
    """Minimal stand-in for :class:`crew.callbacks.ActivityEvent`.

    ``app.run_analysis`` reads ``timestamp``, ``agent_name``,
    ``message``, and ``event_type`` off each pending event when
    translating them into activity feed HTML entries. The ``event_type``
    is only compared for equality against ``EventType.TASK_START`` to
    decide whether to render a spinner, so a plain string (or the string
    payload of the real :class:`crew.callbacks.EventType` enum) is
    sufficient here.
    """

    timestamp: datetime
    agent_name: str
    message: str
    event_type: str


# ---------------------------------------------------------------------------
# Canned signal fixtures
# ---------------------------------------------------------------------------
#
# Three concrete TradingSignal examples spanning the full BUY / SELL /
# HOLD color-coding contract in Requirements 5.1, 5.2. Tests can request
# them by parameter name and rely on the values being stable across runs
# (no randomness, no datetime.now calls).


@pytest.fixture
def buy_signal() -> FakeSignal:
    """A canonical BUY-action signal for AAPL.

    Used by integration tests to exercise the BUY branch of the signal
    card renderer and the buy-count increment in
    ``_render_signals_dashboard``.

    Validates: Requirements 5.1, 5.2
    """
    return FakeSignal(
        ticker="AAPL",
        action=FakeAction("BUY"),
        confidence=85,
        entry_price=100.50,
        stop_loss=95.00,
        target_price=115.25,
        reasoning={"scanner": "bullish pattern"},
    )


@pytest.fixture
def sell_signal() -> FakeSignal:
    """A canonical SELL-action signal for TSLA.

    Mirrors :func:`buy_signal`'s field coverage but with a SELL action
    and a plausible "stop above, target below" price layout for a short
    position.

    Validates: Requirements 5.1, 5.2
    """
    return FakeSignal(
        ticker="TSLA",
        action=FakeAction("SELL"),
        confidence=72,
        entry_price=250.00,
        stop_loss=260.00,
        target_price=230.00,
        reasoning={"scanner": "bearish divergence"},
    )


@pytest.fixture
def hold_signal() -> FakeSignal:
    """A canonical HOLD-action signal for MSFT with no prices.

    Exercises the "optional prices" branch of ``render_signal_card`` —
    all three price fields default to ``None`` so the renderer falls
    back to ``N/A`` via ``_format_price``. Reasoning is also left empty
    to cover the no-reasoning render path.

    Validates: Requirements 5.1, 5.2
    """
    return FakeSignal(
        ticker="MSFT",
        action=FakeAction("HOLD"),
        confidence=60,
        entry_price=None,
        stop_loss=None,
        target_price=None,
        reasoning={},
    )


@pytest.fixture
def trading_signals(
    buy_signal: FakeSignal,
    sell_signal: FakeSignal,
    hold_signal: FakeSignal,
) -> List[FakeSignal]:
    """The three canned signals packaged as an ordered list.

    The fixed ``[BUY, SELL, HOLD]`` ordering gives tests a stable
    expectation for summary counts (one of each) and the order in which
    cards appear in the dashboard HTML.
    """
    return [buy_signal, sell_signal, hold_signal]


# ---------------------------------------------------------------------------
# Canned activity event fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_activity_events() -> List[FakeActivityEvent]:
    """A canned event sequence for a typical single-ticker pipeline run.

    The sequence walks through the event types the real
    :class:`crew.callbacks.ActivityFeedCallback` emits during a
    successful per-ticker analysis:

      1. ``ticker_start``  — system announces the ticker being analyzed.
      2. ``task_start``    — an agent begins work (spinner in the feed).
      3. ``agent_output``  — the agent emits an intermediate update.
      4. ``task_complete`` — the agent finishes and posts its summary.
      5. ``ticker_complete`` — system closes out the ticker.

    Timestamps are spaced one second apart starting from a fixed base
    datetime so tests can assert on chronological ordering without
    flakiness.
    """
    # A UTC-aware base timestamp so the values match what the real
    # ``ActivityFeedCallback`` emits (it uses ``datetime.now(timezone.utc)``).
    base = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    return [
        FakeActivityEvent(
            timestamp=base,
            agent_name="system",
            message="Starting analysis for AAPL",
            event_type="ticker_start",
        ),
        FakeActivityEvent(
            timestamp=base + timedelta(seconds=1),
            agent_name="market_scanner",
            message="market_scanner started task for AAPL",
            event_type="task_start",
        ),
        FakeActivityEvent(
            timestamp=base + timedelta(seconds=2),
            agent_name="market_scanner",
            message="Detected bullish breakout pattern",
            event_type="agent_output",
        ),
        FakeActivityEvent(
            timestamp=base + timedelta(seconds=3),
            agent_name="market_scanner",
            message="Scan complete: 1 high-confidence pattern",
            event_type="task_complete",
        ),
        FakeActivityEvent(
            timestamp=base + timedelta(seconds=4),
            agent_name="system",
            message="Analysis complete for AAPL: BUY (Confidence: 85%)",
            event_type="ticker_complete",
        ),
    ]


# ---------------------------------------------------------------------------
# Fake crew package builder
# ---------------------------------------------------------------------------
#
# ``app.run_analysis`` imports from ``crew`` lazily, inside the try block
# wrapping the pipeline loop::
#
#     from crew import LLMConfig, OrchestratorConfig, WatchlistRunner
#     from crew.callbacks import ActivityEvent, ActivityFeedCallback, EventType
#
# That deferred import means integration tests can inject a stand-in
# ``crew`` / ``crew.callbacks`` pair via ``sys.modules`` *before* the
# handler reaches the try block, and drive the pipeline entirely through
# the fake :class:`WatchlistRunner`. :func:`make_fake_crew_package`
# builds exactly that pair, threading a per-ticker result map through
# the runner so each ticker can succeed or fail independently.


def make_fake_crew_package(
    signals: Dict[str, FakeCrewResult],
    events_for_ticker: Optional[Callable[[str], List[FakeActivityEvent]]] = None,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Build fake ``crew`` and ``crew.callbacks`` modules for injection.

    The returned pair is ready to be installed into ``sys.modules`` via
    ``patch.dict(sys.modules, {"crew": crew, "crew.callbacks": cb})``
    so ``app.run_analysis``'s deferred imports succeed.

    The fake :class:`WatchlistRunner`:

      * accepts any ``(config, tools, callback)`` kwargs in its
        constructor and stores the callback for later use,
      * exposes ``_run_single(ticker)`` which looks ``ticker`` up in
        ``signals`` and returns the corresponding :class:`FakeCrewResult`,
        defaulting to ``FakeCrewResult(success=False, signal=None,
        error="not configured")`` when the ticker isn't in the map,
      * optionally replays a canned sequence of :class:`FakeActivityEvent`
        instances through the callback handler for each ticker when
        ``events_for_ticker`` is provided. This mirrors how the real
        runner emits callback events during ``_run_single``, letting
        tests verify the activity feed translation without spinning up
        the real orchestrator.

    Args:
        signals: Mapping from ticker symbol (uppercase) to the
            :class:`FakeCrewResult` the fake runner should return for
            that ticker. Missing tickers receive a ``not configured``
            failure result.
        events_for_ticker: Optional callable that returns the list of
            activity events to replay through the callback handler for
            a given ticker. When ``None`` (the default), no events are
            emitted and the activity feed will only contain the
            System-scoped entries produced by ``run_analysis`` itself.

    Returns:
        ``(crew_module, callbacks_module)`` — two
        :class:`types.ModuleType` instances ready for ``sys.modules``
        injection.
    """
    crew_mod = types.ModuleType("crew")
    callbacks_mod = types.ModuleType("crew.callbacks")

    # --- Config stubs (accept anything, do nothing) ---
    # ``app.run_analysis`` instantiates ``OrchestratorConfig`` and
    # ``LLMConfig`` but never reads their attributes, so ``**kwargs``
    # acceptors are sufficient.
    class _LLMConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _OrchestratorConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    # --- Callback re-exports ---
    # ``run_analysis`` type-annotates its ``event_handler`` with
    # :class:`ActivityEvent` but only reads the same four attributes our
    # :class:`FakeActivityEvent` exposes, so we re-export the dataclass
    # under the canonical name.
    _ActivityEvent = FakeActivityEvent

    class _ActivityFeedCallback:
        """Minimal callback that stores its handler for later replay."""

        def __init__(self, handler: Callable[[_ActivityEvent], None]) -> None:
            self._handler = handler

        # Mirrors ``crew.callbacks.ActivityFeedCallback._emit`` so code
        # that replays events through the callback has a stable API.
        def emit(self, event: _ActivityEvent) -> None:
            self._handler(event)

    class _EventType:
        """String-keyed event type constants matching the real enum."""

        TICKER_START = "ticker_start"
        TICKER_COMPLETE = "ticker_complete"
        TASK_START = "task_start"
        TASK_COMPLETE = "task_complete"
        TASK_FAILED = "task_failed"
        AGENT_OUTPUT = "agent_output"
        CREW_ERROR = "crew_error"

    # --- WatchlistRunner stand-in ---
    # Captures the callback handler on construction so ``_run_single``
    # can replay pre-canned events through it before returning the
    # ticker's result, faithfully reproducing the real runner's
    # "event stream + final result" pattern.
    class _WatchlistRunner:
        def __init__(
            self,
            config: Any = None,
            tools: Any = None,
            callback: Optional[_ActivityFeedCallback] = None,
            **kwargs: Any,
        ) -> None:
            self.config = config
            self.tools = tools
            self.callback = callback

        def _run_single(self, ticker: str) -> FakeCrewResult:
            # Replay any canned events for this ticker through the
            # callback's handler before returning the result, matching
            # the real runner's emit-then-return flow.
            if events_for_ticker is not None and self.callback is not None:
                for event in events_for_ticker(ticker):
                    self.callback.emit(event)

            return signals.get(
                ticker,
                FakeCrewResult(
                    success=False,
                    signal=None,
                    error="not configured",
                ),
            )

    # --- Module surface ---
    crew_mod.LLMConfig = _LLMConfig
    crew_mod.OrchestratorConfig = _OrchestratorConfig
    crew_mod.WatchlistRunner = _WatchlistRunner

    callbacks_mod.ActivityEvent = _ActivityEvent
    callbacks_mod.ActivityFeedCallback = _ActivityFeedCallback
    callbacks_mod.EventType = _EventType

    return crew_mod, callbacks_mod
