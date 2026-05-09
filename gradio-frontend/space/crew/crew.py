"""FinAgentCrew class — main orchestrator for the multi-agent analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from crewai import Crew, Process

from crew.config import OrchestratorConfig
from crew.agents import (
    create_llm,
    create_market_scanner,
    create_fundamental_analyst,
    create_technical_analyst,
    create_risk_manager,
    create_chief_strategist,
)
from crew.tasks import (
    create_market_scan_task,
    create_fundamental_task,
    create_technical_task,
    create_risk_task,
    create_strategy_task,
)
from crew.signals import TradingSignal, TradingSignalParser
from crew.callbacks import ActivityFeedCallback


@dataclass
class CrewResult:
    """Result of a single ticker crew execution."""

    ticker: str
    signal: Optional[TradingSignal]
    raw_output: str
    success: bool
    error: Optional[str] = None


class FinAgentCrew:
    """Orchestrates the multi-agent analysis pipeline for a single ticker."""

    def __init__(
        self,
        config: OrchestratorConfig,
        tools: dict[str, list],
        callback: Optional[ActivityFeedCallback] = None,
    ):
        """Initialize the crew orchestrator.

        Args:
            config: Full orchestrator configuration
            tools: Dict mapping agent names to their tool lists:
                {
                    "market_scanner": [search_news, get_price_change, get_volume],
                    "fundamental_analyst": [get_financials, get_earnings, get_peers],
                    "technical_analyst": [get_price_history, calculate_indicators],
                    "risk_manager": [calculate_position_size, set_stop_loss],
                }
            callback: Optional activity feed callback for real-time UI updates
        """
        self._config = config
        self._tools = tools
        self._callback = callback
        self._parser = TradingSignalParser()

    def run(self, ticker: str) -> CrewResult:
        """Execute the full analysis pipeline for a single ticker.

        Builds the crew, kicks off execution, parses the output into a
        TradingSignal, and returns a CrewResult. Emits callback events
        at task start, completion, and failure points.

        Args:
            ticker: Stock ticker symbol to analyze (e.g., "AAPL")

        Returns:
            CrewResult with parsed TradingSignal on success,
            or error details on failure.
        """
        try:
            if self._callback:
                self._callback.on_task_start("Crew", ticker)

            crew = self._build_crew(ticker)
            result = crew.kickoff()
            raw_output = str(result)

            signal = self._parse_output(raw_output, ticker)

            if signal is not None:
                if self._callback:
                    self._callback.on_task_complete(
                        "Chief Strategist",
                        ticker,
                        f"{signal.action.value} ({signal.confidence}%)",
                    )
                return CrewResult(
                    ticker=ticker,
                    signal=signal,
                    raw_output=raw_output,
                    success=True,
                )
            else:
                if self._callback:
                    self._callback.on_task_failed(
                        "Chief Strategist",
                        ticker,
                        "Failed to parse trading signal",
                    )
                return CrewResult(
                    ticker=ticker,
                    signal=None,
                    raw_output=raw_output,
                    success=False,
                    error="Failed to parse trading signal from crew output",
                )
        except Exception as e:
            error_msg = str(e)
            if self._callback:
                self._callback.on_task_failed("Crew", ticker, error_msg)
            return CrewResult(
                ticker=ticker,
                signal=None,
                raw_output="",
                success=False,
                error=error_msg,
            )

    def _build_crew(self, ticker: str) -> Crew:
        """Assemble the Crew with agents, tasks, and process configuration.

        Creates all five agents with their respective tools, builds tasks
        with proper dependency chains, and returns a configured Crew instance.

        Args:
            ticker: Stock ticker symbol for task descriptions

        Returns:
            Configured Crew instance ready for kickoff
        """
        llm = create_llm(self._config.llm)

        # Create agents with their assigned tools
        market_scanner = create_market_scanner(
            llm, self._tools.get("market_scanner", [])
        )
        fundamental_analyst = create_fundamental_analyst(
            llm, self._tools.get("fundamental_analyst", [])
        )
        technical_analyst = create_technical_analyst(
            llm, self._tools.get("technical_analyst", [])
        )
        risk_manager = create_risk_manager(
            llm, self._tools.get("risk_manager", [])
        )
        chief_strategist = create_chief_strategist(llm)

        # Create tasks with dependency chain
        market_task = create_market_scan_task(market_scanner, ticker)
        fundamental_task = create_fundamental_task(fundamental_analyst, ticker)
        technical_task = create_technical_task(technical_analyst, ticker)
        risk_task = create_risk_task(risk_manager, ticker, [technical_task])
        strategy_task = create_strategy_task(
            chief_strategist,
            ticker,
            [market_task, fundamental_task, technical_task, risk_task],
        )

        return Crew(
            agents=[
                market_scanner,
                fundamental_analyst,
                technical_analyst,
                risk_manager,
                chief_strategist,
            ],
            tasks=[
                market_task,
                fundamental_task,
                technical_task,
                risk_task,
                strategy_task,
            ],
            process=Process.sequential,
            verbose=self._config.crew.verbose,
        )

    def _parse_output(
        self, raw_output: str, ticker: str
    ) -> Optional[TradingSignal]:
        """Parse crew output into a TradingSignal.

        Delegates to TradingSignalParser which attempts primary structured
        format first, then falls back to heuristic extraction.

        Args:
            raw_output: Raw text output from the crew execution
            ticker: Expected ticker symbol for validation

        Returns:
            TradingSignal if parsing succeeds, None if output is unparseable
        """
        return self._parser.parse(raw_output, ticker)
