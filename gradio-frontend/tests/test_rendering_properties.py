"""Property-based tests for the ``rendering`` module.

Each property mirrors a numbered correctness property from the
gradio-frontend design document and is linked to the originating
acceptance criterion via a ``Validates`` tag in the docstring.

Module-level helper strategies are shared across the property tests in
this file (``test_activity_entry_rendering`` and — once added in tasks
4.8, 4.9, 4.10 — the signal-card, summary, and error-card properties).
"""

import string
from datetime import datetime

from hypothesis import given, strategies as st

from rendering import render_activity_entry, render_signal_card


# ---------------------------------------------------------------------------
# Shared helper strategies
# ---------------------------------------------------------------------------
#
# The activity-feed renderer (and the later signal/summary/error renderers
# added in 4.8-4.10) interpolates user-supplied strings directly into an
# HTML template using ``str.format``. To keep the substring assertions
# reliable we deliberately restrict the text alphabets so generated values
# cannot:
#
#   * coincide with the timestamp format ``HH:MM:SS`` produced by
#     ``datetime.strftime("%H:%M:%S")`` — we exclude digits and the
#     colon character so an agent name / message can never look like a
#     timestamp,
#   * collide with the spinner glyph ``⟳`` (U+27F3) — we restrict to
#     printable ASCII so the character cannot appear by chance,
#   * break HTML substring checks — we exclude the angle brackets
#     ``<`` and ``>`` (and the ampersand) so generated text cannot
#     accidentally introduce or close a tag around an assertion target.

# Characters that are safe to embed in HTML-producing renderers without
# disturbing substring assertions. Letters and ASCII punctuation minus
# the HTML-special/reserved characters listed above, plus spaces.
_SAFE_TEXT_CHARS = (
    string.ascii_letters
    + " "
    + "".join(
        ch
        for ch in string.punctuation
        # HTML-special characters that could confuse substring checks.
        if ch not in {"<", ">", "&"}
        # Colon would allow a generated string to mimic the "HH:MM:SS"
        # timestamp format, breaking the timestamp-presence assertion.
        and ch != ":"
    )
)


def _safe_non_empty_text() -> st.SearchStrategy[str]:
    """Generate a non-empty string from the rendering-safe alphabet.

    The result:
      * is guaranteed non-empty after stripping surrounding whitespace,
      * contains only characters from ``_SAFE_TEXT_CHARS`` (no digits,
        no ``:``, no HTML-special characters, no ``⟳``),
      * has a bounded length to keep counterexamples compact.
    """
    return st.text(alphabet=_SAFE_TEXT_CHARS, min_size=1, max_size=30).filter(
        lambda s: s.strip() != ""
    )


# Datetime strategy restricted to years well inside the supported range of
# ``datetime.strftime`` on every platform (Windows historically rejects
# pre-1970 dates). Timezone-naive datetimes are fine — ``render_activity_entry``
# only ever calls ``strftime("%H:%M:%S")``, which does not require tzinfo.
_DATETIME_STRATEGY = st.datetimes(
    min_value=datetime(1970, 1, 1),
    max_value=datetime(2100, 12, 31),
)


# The spinner glyph asserted on/off by ``test_activity_entry_rendering``.
# Kept as a module-level constant so future rendering properties can reuse it.
_SPINNER_CHAR = "\u27f3"  # ⟳


# ---------------------------------------------------------------------------
# Property 5: Activity entry rendering
# ---------------------------------------------------------------------------


@given(
    timestamp=_DATETIME_STRATEGY,
    agent_name=_safe_non_empty_text(),
    message=_safe_non_empty_text(),
    is_spinner=st.booleans(),
)
def test_activity_entry_rendering(timestamp, agent_name, message, is_spinner):
    """Property 5: Activity entry rendering includes timestamp, agent name,
    and spinner control.

    For any datetime ``timestamp``, non-empty ``agent_name``, non-empty
    ``message``, and boolean ``is_spinner`` flag, the HTML produced by
    ``render_activity_entry`` must contain:

      * the timestamp formatted as ``HH:MM:SS``,
      * the agent name verbatim,
      * the message verbatim, and
      * the spinner glyph ``⟳`` if and only if ``is_spinner=True``.

    Validates: Requirements 3.1, 3.2, 3.3, 3.4
    """
    html = render_activity_entry(
        timestamp=timestamp,
        agent_name=agent_name,
        message=message,
        is_spinner=is_spinner,
    )

    # Output is a single HTML string.
    assert isinstance(html, str)

    # --- Timestamp (Requirement 3.1, 3.3) -----------------------------------
    # The renderer formats the timestamp as HH:MM:SS using ``strftime``.
    # The safe-text alphabet excludes digits and ':' so neither the agent
    # name nor the message can spuriously satisfy this assertion.
    expected_time = timestamp.strftime("%H:%M:%S")
    assert expected_time in html, (
        f"Expected timestamp {expected_time!r} to appear in rendered HTML, "
        f"got: {html!r}"
    )

    # --- Agent name (Requirement 3.2) ---------------------------------------
    assert agent_name in html, (
        f"Expected agent name {agent_name!r} to appear in rendered HTML, "
        f"got: {html!r}"
    )

    # --- Message (Requirement 3.3) ------------------------------------------
    assert message in html, (
        f"Expected message {message!r} to appear in rendered HTML, "
        f"got: {html!r}"
    )

    # --- Spinner control (Requirement 3.2, 3.4) -----------------------------
    # The spinner glyph ``⟳`` MUST appear iff ``is_spinner`` is True.
    # The safe-text alphabet is ASCII-only, so neither the agent name nor
    # the message can contain this character by chance.
    if is_spinner:
        assert _SPINNER_CHAR in html, (
            f"Expected spinner glyph {_SPINNER_CHAR!r} in rendered HTML "
            f"when is_spinner=True, got: {html!r}"
        )
    else:
        assert _SPINNER_CHAR not in html, (
            f"Expected no spinner glyph in rendered HTML when "
            f"is_spinner=False, got: {html!r}"
        )


# ---------------------------------------------------------------------------
# Fake TradingSignal for property 7
# ---------------------------------------------------------------------------
#
# ``render_signal_card`` reads the following attributes off its ``signal``
# argument:
#   * ``ticker`` — string interpolated into the card header,
#   * ``action`` — enum-like value whose ``.value`` is uppercased and used
#     to derive the CSS class (``signal-{action_lower}``),
#   * ``confidence`` — integer rendered as ``{confidence}%``,
#   * ``entry_price`` / ``stop_loss`` / ``target_price`` — optional floats
#     formatted via ``_format_price`` as ``$X.XX``,
#   * ``reasoning`` — a mapping rendered as a ``<ul>`` (kept empty here).
#
# Declaring minimal stand-ins keeps the property test hermetic: we do not
# need to import the real ``TradingSignal`` dataclass from ``crew``, which
# would pull in heavy runtime dependencies.


from dataclasses import dataclass, field
from typing import Any, Dict


class _FakeAction:
    """Stand-in for an enum member with a ``.value`` string attribute.

    ``render_signal_card`` pulls the action label via ``getattr(action,
    "value", action)`` and then uppercases it, matching the behaviour of
    a real ``enum.Enum`` member.
    """

    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class _FakeSignal:
    """Minimal stand-in for the ``TradingSignal`` dataclass.

    Mirrors the attribute names read by ``render_signal_card``.
    """

    ticker: str
    action: _FakeAction
    confidence: int
    entry_price: float
    stop_loss: float
    target_price: float
    reasoning: Dict[str, Any] = field(default_factory=dict)


# Tickers are restricted to uppercase letters and digits so they cannot
# introduce HTML-special characters or a literal ``$``/``%`` that would
# make price or confidence substring checks ambiguous. We additionally
# reject any ticker containing one of the action keywords as a substring
# so the ``action_str in html`` check cannot be satisfied by the ticker
# alone — the renderer must actually emit the action text.
_TICKER_ALPHABET = string.ascii_uppercase + string.digits
_ACTION_WORDS = ("BUY", "SELL", "HOLD")


def _ticker_strategy() -> st.SearchStrategy[str]:
    return st.text(
        alphabet=_TICKER_ALPHABET, min_size=1, max_size=6
    ).filter(lambda t: not any(word in t for word in _ACTION_WORDS))


# ---------------------------------------------------------------------------
# Property 7: Signal card renders all fields with correct action color class
# ---------------------------------------------------------------------------


@given(
    ticker=_ticker_strategy(),
    action_str=st.sampled_from(list(_ACTION_WORDS)),
    confidence=st.integers(min_value=0, max_value=100),
    entry_price=st.floats(
        min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
    ),
    stop_loss=st.floats(
        min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
    ),
    target_price=st.floats(
        min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False
    ),
)
def test_signal_card_renders_all_fields_with_correct_color(
    ticker,
    action_str,
    confidence,
    entry_price,
    stop_loss,
    target_price,
):
    """Property 7: Signal card renders all fields with the correct action
    color class.

    For any valid ``TradingSignal`` with:
      * action in ``{BUY, SELL, HOLD}``,
      * confidence in ``[0, 100]``,
      * non-negative, finite prices,

    the HTML produced by ``render_signal_card`` must contain:
      * the ticker symbol verbatim,
      * the uppercase action label (``BUY``/``SELL``/``HOLD``),
      * the confidence formatted as ``{confidence}%``,
      * each price formatted as ``$X.XX`` (matching ``_format_price``),
      * the action-specific CSS class ``signal-{action_lower}``.

    Validates: Requirements 5.1, 5.2, 5.4
    """
    signal = _FakeSignal(
        ticker=ticker,
        action=_FakeAction(action_str),
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        reasoning={},
    )

    html = render_signal_card(signal)

    # Output is a single HTML string.
    assert isinstance(html, str)

    # --- Ticker (Requirement 5.1) -------------------------------------------
    assert ticker in html, (
        f"Expected ticker {ticker!r} to appear in rendered HTML, got: {html!r}"
    )

    # --- Action text (Requirement 5.1, 5.2) ---------------------------------
    # The renderer uppercases the action value; ``action_str`` is already
    # uppercase so it must appear verbatim in the card.
    assert action_str in html, (
        f"Expected action text {action_str!r} to appear in rendered HTML, "
        f"got: {html!r}"
    )

    # --- Confidence (Requirement 5.1) ---------------------------------------
    expected_confidence = f"{confidence}%"
    assert expected_confidence in html, (
        f"Expected confidence {expected_confidence!r} to appear in rendered "
        f"HTML, got: {html!r}"
    )

    # --- Prices (Requirement 5.1) -------------------------------------------
    # Mirror ``_format_price``'s ``f"${value:.2f}"`` formatting.
    for label, value in (
        ("entry", entry_price),
        ("stop_loss", stop_loss),
        ("target", target_price),
    ):
        expected_price = f"${value:.2f}"
        assert expected_price in html, (
            f"Expected {label} price {expected_price!r} to appear in rendered "
            f"HTML, got: {html!r}"
        )

    # --- Action-specific CSS class (Requirement 5.2, 5.4) -------------------
    expected_class = f"signal-{action_str.lower()}"
    assert expected_class in html, (
        f"Expected CSS class {expected_class!r} to appear in rendered HTML "
        f"for action {action_str!r}, got: {html!r}"
    )


# ---------------------------------------------------------------------------
# Property 8: Summary counts are consistent with the signal list
# ---------------------------------------------------------------------------

from rendering import render_summary


@given(
    buy_count=st.integers(min_value=0, max_value=100),
    sell_count=st.integers(min_value=0, max_value=100),
    hold_count=st.integers(min_value=0, max_value=100),
)
def test_summary_counts_consistency(buy_count, sell_count, hold_count):
    """Property 8: Summary counts are consistent with signal list.

    For any non-negative ``buy_count``, ``sell_count``, and ``hold_count``
    with ``total = buy_count + sell_count + hold_count``, the HTML produced
    by ``render_summary`` must contain each of the four counts in its
    designated summary cell (immediately followed by the closing
    ``</div>`` tag emitted by the renderer).

    We assert against the ``>{count}</div>`` pattern rather than the bare
    digit string so a count of ``1`` cannot be spuriously satisfied by a
    substring of ``10`` appearing elsewhere in the summary.

    Validates: Requirements 5.3
    """
    total = buy_count + sell_count + hold_count

    # Sanity check on the input invariant.
    assert buy_count + sell_count + hold_count == total

    html = render_summary(total, buy_count, sell_count, hold_count)

    # Output is a single HTML string.
    assert isinstance(html, str)

    # --- Each count appears in its summary cell (Requirement 5.3) ----------
    # ``render_summary`` renders each number immediately before a closing
    # ``</div>`` (no intervening whitespace), so ``>{value}</div>`` uniquely
    # identifies a count cell and avoids digit-substring ambiguity
    # (e.g. ``1`` vs ``10``).
    for label, value in (
        ("total", total),
        ("buy", buy_count),
        ("sell", sell_count),
        ("hold", hold_count),
    ):
        expected = f">{value}</div>"
        assert expected in html, (
            f"Expected {label} count cell {expected!r} to appear in rendered "
            f"HTML, got: {html!r}"
        )

    # Bare string presence checks (as specified by the property): each
    # count must appear somewhere in the output. These are weaker than the
    # cell-position checks above but are kept for completeness.
    assert str(total) in html
    assert str(buy_count) in html
    assert str(sell_count) in html
    assert str(hold_count) in html

# ---------------------------------------------------------------------------
# Property 9: Error card contains ticker and error message
# ---------------------------------------------------------------------------

from rendering import render_error_card


@given(
    ticker=_ticker_strategy(),
    error_message=_safe_non_empty_text(),
)
def test_error_card_contains_ticker_and_message(ticker, error_message):
    """Property 9: Error card contains ticker and error message.

    For any non-empty ``ticker`` drawn from the uppercase-alphanumeric
    alphabet (same as Property 7) and any non-empty ``error_message``
    drawn from the rendering-safe text alphabet, the HTML produced by
    ``render_error_card`` must contain:

      * the ticker symbol verbatim,
      * the error message verbatim,
      * the CSS class ``signal-error`` that flags the card as an error
        variant (per the design document).

    Validates: Requirements 5.5
    """
    html = render_error_card(ticker, error_message)

    # Output is a single HTML string.
    assert isinstance(html, str)

    # --- Ticker (Requirement 5.5) -------------------------------------------
    assert ticker in html, (
        f"Expected ticker {ticker!r} to appear in rendered error card, "
        f"got: {html!r}"
    )

    # --- Error message (Requirement 5.5) ------------------------------------
    assert error_message in html, (
        f"Expected error message {error_message!r} to appear in rendered "
        f"error card, got: {html!r}"
    )

    # --- Error CSS class (Requirement 5.5) ----------------------------------
    # The design specifies the error card uses the ``signal-error`` class
    # to distinguish it from BUY/SELL/HOLD signal cards.
    assert "signal-error" in html, (
        f"Expected CSS class 'signal-error' to appear in rendered error "
        f"card, got: {html!r}"
    )
