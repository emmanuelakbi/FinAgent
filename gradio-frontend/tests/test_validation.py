"""Unit tests for the ``validation`` module.

These example-based tests complement the property-based tests in
``test_validation_properties.py`` by locking down specific behaviours
called out in the spec (empty input, whitespace trimming, invalid
characters, the ten-ticker cap, and portfolio-value signedness).

Each test is tagged with the acceptance criteria it exercises via a
``Validates: Requirements X.Y`` note in the docstring.
"""

from validation import MAX_TICKERS, validate_portfolio_value, validate_tickers


# ---------------------------------------------------------------------------
# validate_tickers — empty / whitespace-only input
# ---------------------------------------------------------------------------


def test_validate_tickers_rejects_empty_string():
    """Empty input is rejected with an informative error.

    Validates: Requirements 1.2
    """
    result = validate_tickers("")

    assert result.valid is False
    assert result.tickers == []
    assert result.error_message is not None
    assert result.error_message.strip() != ""


def test_validate_tickers_rejects_whitespace_only_input():
    """Whitespace-only input is rejected the same way as empty input.

    Validates: Requirements 1.2
    """
    result = validate_tickers("   \t  ")

    assert result.valid is False
    assert result.tickers == []
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# validate_tickers — happy-path normalization
# ---------------------------------------------------------------------------


def test_validate_tickers_accepts_single_valid_ticker():
    """A single well-formed ticker round-trips unchanged.

    Validates: Requirements 1.3
    """
    result = validate_tickers("AAPL")

    assert result.valid is True
    assert result.tickers == ["AAPL"]
    assert result.error_message is None


def test_validate_tickers_trims_whitespace_and_uppercases():
    """Surrounding whitespace is stripped and tickers are uppercased.

    Validates: Requirements 1.3
    """
    result = validate_tickers(" aapl , nvda , tsla ")

    assert result.valid is True
    assert result.tickers == ["AAPL", "NVDA", "TSLA"]
    assert result.error_message is None


def test_validate_tickers_uppercases_mixed_case_input():
    """Mixed-case input is normalized to uppercase without reordering.

    Validates: Requirements 1.3
    """
    result = validate_tickers("aapl,NVDA")

    assert result.valid is True
    assert result.tickers == ["AAPL", "NVDA"]


def test_validate_tickers_accepts_periods_and_hyphens():
    """Periods and hyphens are part of the allowed ticker alphabet.

    Tickers such as ``BRK.B`` and ``BF-B`` (Berkshire Hathaway class B,
    Brown-Forman class B) must be accepted as written.

    Validates: Requirements 1.3, 1.4
    """
    result = validate_tickers("BRK.B,BF-B")

    assert result.valid is True
    assert result.tickers == ["BRK.B", "BF-B"]
    assert result.error_message is None


def test_validate_tickers_drops_empty_segments_from_extra_commas():
    """Empty segments produced by adjacent commas are discarded.

    Validates: Requirements 1.3
    """
    result = validate_tickers("AAPL,,NVDA")

    assert result.valid is True
    assert result.tickers == ["AAPL", "NVDA"]
    assert result.error_message is None


# ---------------------------------------------------------------------------
# validate_tickers — invalid-character branch
# ---------------------------------------------------------------------------


def test_validate_tickers_rejects_dollar_sign():
    """A ``$`` anywhere in a ticker triggers the invalid-character branch.

    Validates: Requirements 1.4
    """
    result = validate_tickers("AAPL$")

    assert result.valid is False
    assert result.tickers == []
    assert result.error_message is not None
    assert "$" in result.error_message


def test_validate_tickers_rejects_at_sign():
    """An ``@`` anywhere in a ticker triggers the invalid-character branch.

    Validates: Requirements 1.4
    """
    result = validate_tickers("GOOG@L")

    assert result.valid is False
    assert result.tickers == []
    assert result.error_message is not None
    assert "@" in result.error_message


# ---------------------------------------------------------------------------
# validate_tickers — ticker-count cap
# ---------------------------------------------------------------------------


def test_validate_tickers_accepts_exactly_max_tickers():
    """Exactly ``MAX_TICKERS`` (10) tickers is the boundary of acceptance.

    Validates: Requirements 1.5
    """
    tickers = [f"T{i}" for i in range(MAX_TICKERS)]
    raw_input = ",".join(tickers)

    result = validate_tickers(raw_input)

    assert result.valid is True
    assert result.tickers == tickers
    assert len(result.tickers) == MAX_TICKERS
    assert result.error_message is None


def test_validate_tickers_rejects_one_over_max_tickers():
    """One more than ``MAX_TICKERS`` tickers trips the cap.

    Validates: Requirements 1.5
    """
    tickers = [f"T{i}" for i in range(MAX_TICKERS + 1)]
    raw_input = ",".join(tickers)

    result = validate_tickers(raw_input)

    assert result.valid is False
    assert result.tickers == []
    assert result.error_message is not None
    # The error must surface the numeric limit so the user knows why
    # their input was rejected.
    assert str(MAX_TICKERS) in result.error_message


# ---------------------------------------------------------------------------
# validate_portfolio_value
# ---------------------------------------------------------------------------


def test_validate_portfolio_value_rejects_negative():
    """Negative portfolio values return an error string.

    Validates: Requirements 2.2
    """
    error = validate_portfolio_value(-100)

    assert error is not None
    assert isinstance(error, str)
    assert error.strip() != ""


def test_validate_portfolio_value_accepts_zero():
    """Zero is treated as valid (non-negative).

    Validates: Requirements 2.2
    """
    assert validate_portfolio_value(0) is None


def test_validate_portfolio_value_accepts_positive():
    """A typical positive value validates cleanly.

    Validates: Requirements 2.2
    """
    assert validate_portfolio_value(10000) is None
