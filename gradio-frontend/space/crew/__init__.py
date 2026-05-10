"""CrewAI multi-agent orchestration layer for FinAgent.

This package coordinates five specialized AI agents to analyze financial
tickers and produce structured trading signals (BUY/SELL/HOLD).
"""

from crew.callbacks import ActivityEvent, ActivityFeedCallback, EventType
from crew.config import CrewConfig, LLMConfig, OrchestratorConfig, TradePreferences
from crew.crew import CrewResult, FinAgentCrew
from crew.runner import WatchlistResult, WatchlistRunner
from crew.signals import Action, TradingSignal

__all__ = [
    "Action",
    "ActivityEvent",
    "ActivityFeedCallback",
    "CrewConfig",
    "CrewResult",
    "EventType",
    "FinAgentCrew",
    "LLMConfig",
    "OrchestratorConfig",
    "TradePreferences",
    "TradingSignal",
    "WatchlistResult",
    "WatchlistRunner",
]
