"""FinAgentCrew class — main orchestrator for the multi-agent analysis pipeline."""

from __future__ import annotations

import re
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
from crew.signals import Action, TradingSignal, TradingSignalParser
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

            # If the parser couldn't extract a structured signal from the
            # Strategist's prose, fall back to a deterministic synthesis
            # that grounds the signal in the upstream tools (live price
            # from yfinance + a heuristic BUY/SELL/HOLD from the recent
            # trend). This keeps the pipeline producing sensible output
            # even when Qwen loses the output format mid-deliberation.
            if signal is None:
                signal = self._synthesize_from_tools(ticker, raw_output)
            else:
                # Cross-check the parsed entry against the live tool
                # price. Small LLMs occasionally emit completely
                # fabricated prices (e.g. $10.00 for a $215 stock) that
                # happen to pass the relative-ordering sanity check
                # because stop/target are on the right side of entry.
                # Re-synthesize from tools whenever the parsed entry is
                # wildly off from reality.
                signal = self._reground_if_drifted(ticker, signal, raw_output)

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

    def _reground_if_drifted(
        self,
        ticker: str,
        signal: TradingSignal,
        raw_output: str,
    ) -> TradingSignal:
        """Swap out a parsed signal whose entry is wildly off the live price.

        Qwen3-14B occasionally invents prices at synthesis time that happen
        to satisfy the BUY/SELL inequalities — e.g. returning a \$10.00
        entry for NVDA on a day it traded at \$215. Those pass the parser
        but mislead the user. We fetch the live price via
        :func:`get_price_change` and, if the parsed entry differs by more
        than 30 %, replace the whole signal with the deterministic
        synthesis from :meth:`_synthesize_from_tools`. If the entry is in
        range we also back-fill any missing stop/target so HOLD cards
        render with real numbers instead of N/A.
        """
        parsed_entry = signal.entry_price

        # Get the live price (best-effort; don't block on tool errors).
        live_entry: Optional[float] = None
        try:
            market_tools = self._tools.get("market_scanner", [])
            price_tool = next(
                (t for t in market_tools
                 if getattr(t, "name", "") == "Get Price Change"),
                None,
            )
            if price_tool is not None:
                price_fn = getattr(price_tool, "func", price_tool)
                result = str(price_fn(ticker))
                m = re.search(
                    r"Current Price:\s*\$\s*([\d,]+\.?\d*)", result
                )
                if m:
                    live_entry = float(m.group(1).replace(",", ""))
        except Exception:
            live_entry = None

        # If we couldn't fetch a live price, leave the signal as-is —
        # the sanity-fix already clamped relative ordering.
        if live_entry is None or live_entry <= 0:
            return self._backfill_missing_prices(signal)

        # If the parsed entry is missing or drifts by more than 30 %,
        # replace with the deterministic synthesis.
        drifted = (
            parsed_entry is None
            or parsed_entry <= 0
            or abs(parsed_entry - live_entry) / live_entry > 0.30
        )
        if drifted:
            synthesized = self._synthesize_from_tools(ticker, raw_output)
            if synthesized is not None:
                return synthesized
            # If synthesis failed too, at least swap the entry.
            return TradingSignal(
                ticker=signal.ticker,
                action=signal.action,
                confidence=signal.confidence,
                entry_price=live_entry,
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
                reasoning=signal.reasoning,
            )

        # Parsed entry is close to live price — keep the LLM signal and
        # back-fill any missing stop/target fields, plus swap out any
        # stop/target that are wildly off (more than 20 % from entry).
        # Small LLMs occasionally emit numbers that are credible-looking
        # on their own but nonsensical for the ticker (e.g. BTC-USD
        # at \$80 782 with a \$79 stop-loss — the model dropped the "k").
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.target_price

        def _far_from_entry(v: Optional[float]) -> bool:
            if v is None or v <= 0:
                return True
            return abs(v - entry) / entry > 0.20

        def _too_close_to_entry(v: Optional[float]) -> bool:
            """True if a stop/target is within 0.5 % of entry — effectively a
            degenerate zero-risk / zero-reward number that the LLM sometimes
            emits when it conflates HOLD with 'no action'."""
            if v is None or v <= 0:
                return True
            return abs(v - entry) / entry < 0.005

        if _far_from_entry(stop) or _too_close_to_entry(stop):
            stop = None  # trigger back-fill below
        if _far_from_entry(target) or _too_close_to_entry(target):
            target = None

        if stop is None or target is None:
            return self._backfill_missing_prices(
                TradingSignal(
                    ticker=signal.ticker,
                    action=signal.action,
                    confidence=signal.confidence,
                    entry_price=entry,
                    stop_loss=stop,
                    target_price=target,
                    reasoning=signal.reasoning,
                )
            )

        return signal

    @staticmethod
    def _backfill_missing_prices(signal: TradingSignal) -> TradingSignal:
        """Back-fill stop-loss and target when the LLM emitted N/A or omitted them.

        This ensures the UI card always shows three numeric prices rather
        than a mix of numbers and N/A placeholders.
        """
        entry = signal.entry_price
        if entry is None or entry <= 0:
            return signal

        stop = signal.stop_loss
        target = signal.target_price

        if stop is None or stop <= 0:
            if signal.action == Action.BUY:
                stop = round(entry * 0.97, 2)
            elif signal.action == Action.SELL:
                stop = round(entry * 1.03, 2)
            else:  # HOLD
                stop = round(entry * 0.97, 2)

        if target is None or target <= 0:
            if signal.action == Action.BUY:
                target = round(entry * 1.05, 2)
            elif signal.action == Action.SELL:
                target = round(entry * 0.95, 2)
            else:  # HOLD
                target = round(entry * 1.03, 2)

        return TradingSignal(
            ticker=signal.ticker,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            reasoning=signal.reasoning,
        )

    def _synthesize_from_tools(
        self, ticker: str, raw_output: str
    ) -> Optional[TradingSignal]:
        """Build a TradingSignal directly from tool outputs when LLM parsing fails.

        First we try to scan ``raw_output`` for the canonical
        ``Current Price: $X.XX`` row emitted by ``tools.market_scanner``
        — that row leaks into the transcript when the agents call
        ``get_price_change``. If the Strategist's final answer doesn't
        include it (it often doesn't — the Strategist is the last agent
        and only sees the prior agents' *summaries*), we call
        ``get_price_change`` directly as a backstop so we still have a
        live price to ground the card on.

        From the live price we derive:

        * **entry** = live price
        * **action** = BUY if today's change is ≥ +1 %, SELL if ≤ −1 %,
          otherwise HOLD
        * **stop / target** = ± 3 % / ± 5 % of entry
        * **confidence** = 50 baseline, + up to 25 scaled by |% change|
        * **reasoning** = preserve the LLM's narrative (first ~800 chars)

        Returns ``None`` only if no live price can be retrieved at all.
        """
        entry: Optional[float] = None
        pct_change = 0.0

        # Try the transcript first (cheap — no extra network call).
        price_match = re.search(
            r"Current Price:\s*\$\s*([\d,]+\.?\d*)", raw_output
        )
        if price_match:
            try:
                entry = float(price_match.group(1).replace(",", ""))
            except ValueError:
                entry = None

            pct_match = re.search(
                r"Change:[^()]*\(([+-]?)([\d.]+)%\)", raw_output
            )
            if pct_match:
                sign = -1.0 if pct_match.group(1) == "-" else 1.0
                try:
                    pct_change = sign * float(pct_match.group(2))
                except ValueError:
                    pct_change = 0.0

        # Backstop: call get_price_change directly if the Strategist's
        # response did not carry the price row through from upstream.
        if entry is None:
            try:
                market_tools = self._tools.get("market_scanner", [])
                price_tool = next(
                    (t for t in market_tools
                     if getattr(t, "name", "") == "Get Price Change"),
                    None,
                )
                if price_tool is None:
                    return None
                # crewai wraps tools; the underlying function is .func.
                price_fn = getattr(price_tool, "func", price_tool)
                result = price_fn(ticker)
                p = re.search(
                    r"Current Price:\s*\$\s*([\d,]+\.?\d*)", str(result)
                )
                if not p:
                    return None
                entry = float(p.group(1).replace(",", ""))
                pct = re.search(
                    r"Change:[^()]*\(([+-]?)([\d.]+)%\)", str(result)
                )
                if pct:
                    sign = -1.0 if pct.group(1) == "-" else 1.0
                    try:
                        pct_change = sign * float(pct.group(2))
                    except ValueError:
                        pct_change = 0.0
            except Exception:
                return None

        if entry is None or entry <= 0:
            return None

        # Choose action from today's trend.
        if pct_change >= 1.0:
            action = Action.BUY
        elif pct_change <= -1.0:
            action = Action.SELL
        else:
            action = Action.HOLD

        # Price bands: tight stop, a touch more space on target.
        if action == Action.BUY:
            stop = round(entry * 0.97, 2)
            target = round(entry * 1.05, 2)
        elif action == Action.SELL:
            stop = round(entry * 1.03, 2)
            target = round(entry * 0.95, 2)
        else:  # HOLD
            stop = round(entry * 0.97, 2)
            target = round(entry * 1.03, 2)

        # Confidence = 50 baseline + up to 25 more for stronger moves.
        confidence = int(
            50 + min(25, round(abs(pct_change) * 5))
        )

        # Preserve the agents' narrative so the user still sees the
        # reasoning — just trimmed so the UI card stays readable.
        narrative = raw_output.strip()
        if len(narrative) > 800:
            narrative = narrative[:800] + "..."
        reasoning = {"Synthesized": narrative} if narrative else None

        return TradingSignal(
            ticker=ticker,
            action=action,
            confidence=confidence,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            reasoning=reasoning,
        )
