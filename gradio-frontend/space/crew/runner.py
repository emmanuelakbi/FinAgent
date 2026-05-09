"""WatchlistRunner — multi-ticker sequential execution with fault isolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from crew.config import OrchestratorConfig
from crew.crew import CrewResult, FinAgentCrew
from crew.callbacks import ActivityFeedCallback


@dataclass
class WatchlistResult:
    """Aggregated result of running the analysis pipeline across multiple tickers."""

    signals: list[CrewResult] = field(default_factory=list)
    total_tickers: int = 0
    successful: int = 0
    failed: int = 0


class WatchlistRunner:
    """Runs the FinAgentCrew pipeline for each ticker in a watchlist sequentially."""

    def __init__(
        self,
        config: OrchestratorConfig,
        tools: dict[str, list],
        callback: Optional[ActivityFeedCallback] = None,
    ):
        self._config = config
        self._tools = tools
        self._callback = callback

    def run(self, watchlist: str) -> WatchlistResult:
        """Parse the watchlist and run the analysis pipeline for each ticker.

        Args:
            watchlist: Comma-separated string of ticker symbols.

        Returns:
            WatchlistResult with aggregated signals and success/failure counts.
        """
        tickers = self._parse_watchlist(watchlist)
        results: list[CrewResult] = []
        successful = 0
        failed = 0

        for ticker in tickers:
            if self._callback:
                self._callback.on_ticker_start(ticker)

            result = self._run_single(ticker)
            results.append(result)

            if result.success:
                successful += 1
            else:
                failed += 1

            if self._callback:
                self._callback.on_ticker_complete(ticker, result.signal)

        return WatchlistResult(
            signals=results,
            total_tickers=len(tickers),
            successful=successful,
            failed=failed,
        )

    def _parse_watchlist(self, watchlist: str) -> list[str]:
        """Split watchlist string on commas, strip whitespace, uppercase, remove empties.

        Args:
            watchlist: Raw comma-separated ticker string.

        Returns:
            List of cleaned, uppercased ticker symbols.
        """
        parts = watchlist.split(",")
        tickers = []
        for part in parts:
            stripped = part.strip().upper()
            if stripped:
                tickers.append(stripped)
        return tickers

    def _run_single(self, ticker: str) -> CrewResult:
        """Run the full analysis pipeline for a single ticker with error isolation.

        Args:
            ticker: Uppercased ticker symbol.

        Returns:
            CrewResult on success or a failure CrewResult if an exception occurs.
        """
        try:
            crew = FinAgentCrew(
                config=self._config,
                tools=self._tools,
                callback=self._callback,
            )
            return crew.run(ticker)
        except Exception as e:
            return CrewResult(
                ticker=ticker,
                signal=None,
                raw_output="",
                success=False,
                error=str(e),
            )
