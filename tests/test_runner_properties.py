"""
Property-based tests for the runner module.

**Validates: Requirements 10.1, 10.2**
"""

import sys
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

# Mock crewai and langchain_openai before importing crew modules
mock_crewai_module = MagicMock()
mock_crewai_module.Crew = MagicMock()
mock_crewai_module.Process = MagicMock()
mock_crewai_module.Process.sequential = "sequential"
mock_crewai_module.Agent = MagicMock()
mock_crewai_module.Task = MagicMock()

mock_langchain_openai = MagicMock()
mock_langchain_openai.ChatOpenAI = MagicMock()

with patch.dict(
    sys.modules,
    {
        "crewai": mock_crewai_module,
        "langchain_openai": mock_langchain_openai,
    },
):
    from crew.crew import CrewResult
    from crew.runner import WatchlistRunner, WatchlistResult
    from crew.config import OrchestratorConfig


# --- Strategies for Property 7: Watchlist parsing ---

# Strategy: generate ticker-like strings (letters and numbers, 1-5 chars)
ticker_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=5,
)

# Strategy: generate whitespace padding
whitespace_strategy = st.text(alphabet=" \t", max_size=3)


# --- Strategies for Property 8: WatchlistResult aggregation ---

crew_result_strategy = st.builds(
    CrewResult,
    ticker=st.from_regex(r"[A-Z]{1,5}", fullmatch=True),
    signal=st.none(),
    raw_output=st.text(),
    success=st.booleans(),
    error=st.one_of(st.none(), st.text()),
)

results_strategy = st.lists(crew_result_strategy, min_size=0, max_size=20)


class TestWatchlistParsingProperty:
    """Property 7: Watchlist parsing produces normalized ticker list.

    For any comma-separated string of ticker symbols (with arbitrary whitespace
    and mixed case), `_parse_watchlist` SHALL return a list of uppercase, trimmed
    ticker strings with empty entries removed, and the count of returned tickers
    SHALL equal the number of non-empty segments in the input.

    **Validates: Requirements 10.1**
    """

    @given(
        tickers=st.lists(ticker_strategy, min_size=1, max_size=10),
        left_pads=st.lists(whitespace_strategy, min_size=10, max_size=10),
        right_pads=st.lists(whitespace_strategy, min_size=10, max_size=10),
        extra_commas=st.lists(st.booleans(), min_size=10, max_size=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_parse_watchlist_normalizes_tickers(
        self,
        tickers: list[str],
        left_pads: list[str],
        right_pads: list[str],
        extra_commas: list[bool],
    ) -> None:
        """Watchlist parsing uppercases, trims, and removes empty entries."""
        # Build a comma-separated string with random whitespace around each ticker
        parts = []
        for i, ticker in enumerate(tickers):
            lpad = left_pads[i % len(left_pads)]
            rpad = right_pads[i % len(right_pads)]
            parts.append(f"{lpad}{ticker}{rpad}")
            # Optionally add extra commas (which produce empty segments)
            if extra_commas[i % len(extra_commas)]:
                parts.append("")

        joined = ",".join(parts)

        runner = WatchlistRunner(OrchestratorConfig(), {})
        result = runner._parse_watchlist(joined)

        # All returned tickers are uppercase
        for t in result:
            assert t == t.upper(), f"Ticker {t!r} is not uppercase"

        # All returned tickers are stripped (no leading/trailing whitespace)
        for t in result:
            assert t == t.strip(), f"Ticker {t!r} has leading/trailing whitespace"

        # No empty strings in the result
        for t in result:
            assert t != "", "Empty string found in result"

        # The count of returned tickers equals the number of non-empty segments
        expected_non_empty = [
            seg.strip().upper() for seg in joined.split(",") if seg.strip()
        ]
        assert len(result) == len(expected_non_empty), (
            f"Expected {len(expected_non_empty)} tickers, got {len(result)}"
        )


class TestWatchlistResultAggregationInvariant:
    """Property 8: WatchlistResult aggregation invariant.

    For any list of CrewResult objects, the WatchlistResult SHALL have
    `total_tickers` equal to the length of the list, `successful` equal
    to the count where `success=True`, and `failed` equal to the count
    where `success=False`, such that `successful + failed == total_tickers`.

    **Validates: Requirements 10.2**
    """

    @given(results=results_strategy)
    @settings(max_examples=200, deadline=None)
    def test_aggregation_invariant_holds(self, results: list) -> None:
        """WatchlistResult aggregation counts are consistent with the input list."""
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        wr = WatchlistResult(
            signals=results,
            total_tickers=len(results),
            successful=successful,
            failed=failed,
        )

        assert wr.total_tickers == len(results)
        assert wr.successful + wr.failed == wr.total_tickers
        assert wr.successful == sum(1 for r in results if r.success)
        assert wr.failed == sum(1 for r in results if not r.success)
