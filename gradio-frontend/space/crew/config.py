"""LLM configuration, timeouts, and constants for the orchestration layer."""

from dataclasses import dataclass, field
from typing import Literal


RiskTolerance = Literal["Conservative", "Moderate", "Aggressive"]
TradingStyle = Literal["Day Trading", "Swing Trading", "Position Trading"]


@dataclass(frozen=True)
class TradePreferences:
    """User-selected risk / style / portfolio preferences.

    These come from the Gradio UI dropdowns and are threaded end-to-end
    into the Chief Strategist's task description so the final signal
    reflects the user's stated profile. They also tune the re-ground
    clamps in :class:`FinAgentCrew` — Conservative + Day Trading pushes
    stop / target toward tight, short-horizon bands; Aggressive +
    Position Trading opens them up.

    Numbers here are percentages of entry (0.03 == 3 %).
    """

    risk_tolerance: RiskTolerance = "Moderate"
    trading_style: TradingStyle = "Swing Trading"
    portfolio_value: float = 10000.0

    @property
    def stop_pct(self) -> float:
        return {
            "Conservative": 0.015,
            "Moderate": 0.03,
            "Aggressive": 0.05,
        }[self.risk_tolerance]

    @property
    def target_pct(self) -> float:
        return {
            "Day Trading": 0.02,
            "Swing Trading": 0.05,
            "Position Trading": 0.10,
        }[self.trading_style]

    @property
    def stop_clamp(self) -> float:
        """Upper bound used by _reground_if_drifted. Slightly looser than
        ``stop_pct`` so the LLM has headroom, but tight enough to catch
        the wide ATR-style stops Qwen sometimes emits."""
        return self.stop_pct * 2.5

    @property
    def target_clamp(self) -> float:
        return self.target_pct * 3.0


@dataclass
class LLMConfig:
    """Configuration for the vLLM endpoint connection."""

    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-14B"
    temperature: float = 0.7
    max_tokens: int = 1024
    request_timeout: int = 120  # seconds


@dataclass
class CrewConfig:
    """Configuration for crew execution parameters."""

    max_iterations: int = 5
    task_timeout: int = 120  # seconds per task
    verbose: bool = True


@dataclass
class OrchestratorConfig:
    """Top-level configuration combining all settings."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    crew: CrewConfig = field(default_factory=CrewConfig)
