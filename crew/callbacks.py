"""ActivityFeedCallback implementation for real-time UI updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from crew.signals import TradingSignal


class EventType(str, Enum):
    """Types of activity feed events."""

    TICKER_START = "ticker_start"
    TICKER_COMPLETE = "ticker_complete"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    AGENT_OUTPUT = "agent_output"
    CREW_ERROR = "crew_error"


@dataclass
class ActivityEvent:
    """Structured payload for activity feed callbacks."""

    event_type: EventType
    agent_name: str
    ticker: str
    message: str
    timestamp: datetime


class ActivityFeedCallback:
    """Manages activity feed event dispatch to the Gradio UI."""

    def __init__(self, handler: Callable[[ActivityEvent], None]) -> None:
        """Initialize with a handler function that receives ActivityEvent payloads.

        Args:
            handler: Function that receives ActivityEvent payloads.
                     Typically connected to a Gradio state update.
        """
        self._handler = handler

    def on_ticker_start(self, ticker: str) -> None:
        """Emit event when a ticker analysis begins."""
        event = ActivityEvent(
            event_type=EventType.TICKER_START,
            agent_name="system",
            ticker=ticker,
            message=f"Starting analysis for {ticker}",
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def on_ticker_complete(
        self, ticker: str, signal: Optional[TradingSignal] = None
    ) -> None:
        """Emit event when a ticker analysis completes."""
        if signal is not None:
            message = f"Analysis complete for {ticker}: {signal.action.value} (Confidence: {signal.confidence}%)"
        else:
            message = f"Analysis complete for {ticker}"
        event = ActivityEvent(
            event_type=EventType.TICKER_COMPLETE,
            agent_name="system",
            ticker=ticker,
            message=message,
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def on_task_start(self, agent_name: str, ticker: str) -> None:
        """Emit event when an agent task begins execution."""
        event = ActivityEvent(
            event_type=EventType.TASK_START,
            agent_name=agent_name,
            ticker=ticker,
            message=f"{agent_name} started task for {ticker}",
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def on_task_complete(
        self, agent_name: str, ticker: str, output_summary: str
    ) -> None:
        """Emit event when an agent task completes successfully."""
        event = ActivityEvent(
            event_type=EventType.TASK_COMPLETE,
            agent_name=agent_name,
            ticker=ticker,
            message=output_summary,
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def on_task_failed(self, agent_name: str, ticker: str, error: str) -> None:
        """Emit event when an agent task fails."""
        event = ActivityEvent(
            event_type=EventType.TASK_FAILED,
            agent_name=agent_name,
            ticker=ticker,
            message=f"{agent_name} failed for {ticker}: {error}",
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def on_agent_output(self, agent_name: str, ticker: str, output: str) -> None:
        """Emit event for intermediate agent output."""
        event = ActivityEvent(
            event_type=EventType.AGENT_OUTPUT,
            agent_name=agent_name,
            ticker=ticker,
            message=output,
            timestamp=datetime.now(timezone.utc),
        )
        self._emit(event)

    def _emit(self, event: ActivityEvent) -> None:
        """Dispatch event to the registered handler."""
        self._handler(event)
