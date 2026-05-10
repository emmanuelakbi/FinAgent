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
        r"""Anchor every signal's entry price to the live quote.

        Small LLMs frequently emit training-era prices (Qwen3-14B once
        handed back \$203 for NVDA when it was trading at \$215) that
        pass the parser because the BUY/SELL inequalities still hold
        relative to the invented entry. The card then shows a stale
        price — useless for a trading signal.

        The fix is absolute: we fetch the current quote via
        :func:`get_price_change`, force ``entry_price`` to that value,
        and **rescale** stop / target by the same ratio so the LLM's
        risk / reward geometry is preserved. Example: LLM emits
        entry=\$200, stop=\$194, target=\$210 for an NVDA card while
        the live price is \$215 — we scale 1.075× so the card renders
        entry=\$215, stop=\$208.52, target=\$225.75. The narrative
        reasoning is kept as-is.

        When the parsed entry is missing, zero, or more than 50 % off
        the live price we fall back to the deterministic tool-grounded
        synthesis in :meth:`_synthesize_from_tools`, which picks a
        BUY / SELL / HOLD from the day's % change and derives bands
        from scratch.
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

        # Parsed entry is missing or grossly off (>50 %): the model
        # probably fabricated the entire signal, so re-synthesise from
        # tools. Below 50 % drift we keep the LLM's narrative but
        # rescale the numbers.
        badly_drifted = (
            parsed_entry is None
            or parsed_entry <= 0
            or abs(parsed_entry - live_entry) / live_entry > 0.50
        )
        if badly_drifted:
            synthesized = self._synthesize_from_tools(ticker, raw_output)
            if synthesized is not None:
                return synthesized
            # If synthesis failed too, at least swap the entry.
            return self._backfill_missing_prices(
                TradingSignal(
                    ticker=signal.ticker,
                    action=signal.action,
                    confidence=signal.confidence,
                    entry_price=live_entry,
                    stop_loss=None,
                    target_price=None,
                    reasoning=signal.reasoning,
                )
            )

        # Anchor to live price and rescale stop / target to preserve
        # the LLM's risk-reward geometry. If either side is missing or
        # degenerate, let the back-fill derive a fresh band.
        scale = live_entry / parsed_entry
        stop = signal.stop_loss
        target = signal.target_price

        def _bad(v: Optional[float]) -> bool:
            """Treat very-close-to-entry (< 0.5 %) stops / targets as
            degenerate zero-risk numbers the model sometimes emits on
            HOLD calls."""
            if v is None or v <= 0:
                return True
            return abs(v - parsed_entry) / parsed_entry < 0.005

        new_stop = None if _bad(stop) else round(stop * scale, 2)
        new_target = None if _bad(target) else round(target * scale, 2)

        # Extra guard: after rescaling, stop / target should still be
        # within a reasonable band of the new entry (tight stop < 10 %,
        # target < 15 %). If the LLM emitted something implausibly wide
        # we let the back-fill replace it.
        def _unreasonable(v: Optional[float]) -> bool:
            if v is None:
                return True
            return abs(v - live_entry) / live_entry > 0.15

        if _unreasonable(new_stop):
            new_stop = None
        if _unreasonable(new_target):
            new_target = None

        rescaled = TradingSignal(
            ticker=signal.ticker,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=round(live_entry, 2),
            stop_loss=new_stop,
            target_price=new_target,
            reasoning=signal.reasoning,
        )
        return self._backfill_missing_prices(rescaled)

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
