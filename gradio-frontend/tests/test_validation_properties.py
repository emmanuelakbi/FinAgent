"""Property-based tests for the ``validation`` module.

Each property mirrors a numbered correctness property from the
gradio-frontend design document and is linked to the originating
acceptance criterion via a ``Validates`` tag in the docstring.
"""

import string

from hypothesis import given, strategies as st

from validation import validate_tickers


# Characters that are accepted by ``TICKER_PATTERN`` — letters, digits,
# hyphen, and period. Whitespace is NOT part of the alphabet because
# the pattern rejects whitespace once the segment has been trimmed.
_VALID_TICKER_CHARS = string.ascii_letters + string.digits + "-."

# Whitespace alphabet for padding around tickers and for building
# empty (whitespace-only) segments. Kept to spaces and tabs so the
# generated strings stay readable when a counterexample is reported.
_WHITESPACE_CHARS = " \t"


@st.composite
def _valid_ticker_input(draw):
    """Build an input string that ``validate_tickers`` should accept.

    The strategy constructs a comma-separated input where:
      * Between 1 and ``MAX_TICKERS`` non-empty ticker segments are present
        (ensuring the count-limit rule does not fire).
      * Each ticker core is drawn from the allowed alphabet and wrapped
        in arbitrary leading/trailing whitespace.
      * Additional whitespace-only (or fully empty) segments may appear
        before and after every ticker. These should be silently dropped
        by ``validate_tickers``, which is exactly what the property
        under test asserts.

    Returns:
        A tuple ``(cores, raw_input)`` where ``cores`` is the ordered
        list of ticker cores as drawn (pre-uppercase) and ``raw_input``
        is the comma-joined string that simulates user input.
    """

    ticker_core_strategy = st.text(
        alphabet=_VALID_TICKER_CHARS, min_size=1, max_size=10
    )
    whitespace_strategy = st.text(alphabet=_WHITESPACE_CHARS, max_size=5)

    n_valid = draw(st.integers(min_value=1, max_value=10))

    cores: list[str] = []
    segments: list[str] = []

    for _ in range(n_valid):
        # Optionally prefix with whitespace-only segments that should be
        # filtered out by the validator.
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            segments.append(draw(whitespace_strategy))

        core = draw(ticker_core_strategy)
        leading = draw(whitespace_strategy)
        trailing = draw(whitespace_strategy)
        cores.append(core)
        segments.append(leading + core + trailing)

    # Optionally append trailing whitespace-only segments.
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        segments.append(draw(whitespace_strategy))

    return cores, ",".join(segments)


@given(_valid_ticker_input())
def test_ticker_normalization_preserves_content(data):
    """Property 1: Ticker normalization preserves content.

    For any comma-separated string of valid ticker symbols with
    arbitrary whitespace padding (and optional empty/whitespace-only
    segments), ``validate_tickers`` returns a list where each element
    is the trimmed, uppercase version of the corresponding input
    segment and the count equals the number of non-empty segments.

    Validates: Requirements 1.3
    """
    cores, raw_input = data

    result = validate_tickers(raw_input)

    # The generator only produces inputs that satisfy every other rule,
    # so validation must succeed here.
    assert result.valid, (
        f"Expected valid result but got error: {result.error_message!r} "
        f"for input {raw_input!r}"
    )
    assert result.error_message is None

    expected = [core.upper() for core in cores]

    # Each returned ticker is the trimmed, uppercase version of the
    # matching non-empty input segment, in the original order.
    assert result.tickers == expected

    # Count matches the number of non-empty segments.
    assert len(result.tickers) == len(cores)


# Whitespace alphabet used for Property 2. Covers every character that
# ``str.strip()`` treats as whitespace so the property exercises the full
# surface area of "empty or whitespace-only" inputs: ASCII space/tab,
# newlines, carriage returns, form feeds, and vertical tabs.
_EMPTY_OR_WHITESPACE_CHARS = " \t\n\r\f\v"


@given(st.text(alphabet=_EMPTY_OR_WHITESPACE_CHARS, min_size=0, max_size=50))
def test_empty_or_whitespace_input_rejected(raw_input):
    """Property 2: Empty and whitespace-only input is rejected.

    For any string composed entirely of whitespace characters (including
    the empty string), ``validate_tickers`` returns ``valid=False`` with
    a non-empty error message and an empty tickers list.

    Validates: Requirements 1.2
    """
    result = validate_tickers(raw_input)

    # The validator must reject the input outright.
    assert result.valid is False, (
        f"Expected validation to fail for whitespace-only input "
        f"{raw_input!r}, but it succeeded."
    )

    # The tickers list must be empty — no content can be extracted
    # from an all-whitespace input.
    assert result.tickers == [], (
        f"Expected empty tickers list for whitespace-only input "
        f"{raw_input!r}, got {result.tickers!r}."
    )

    # The user must receive a non-empty, informative error message.
    assert result.error_message is not None
    assert isinstance(result.error_message, str)
    assert result.error_message.strip() != "", (
        f"Expected a non-empty error message, got {result.error_message!r}."
    )


# Alphabet of characters that are guaranteed to be rejected by
# ``TICKER_PATTERN``. We deliberately exclude:
#   * Alphanumerics, hyphen, period — these are the allowed set.
#   * Comma — it is the segment delimiter and would alter the ticker list
#     shape rather than trigger the invalid-character branch.
#   * Whitespace — leading/trailing whitespace is stripped by the
#     validator and whitespace-only segments are dropped entirely, so
#     it would not reliably surface as an "invalid character" in the
#     error message.
# What remains are printable ASCII symbols whose ``.upper()`` form is
# the character itself, so the validator's uppercasing step does not
# alter them before they are embedded in the error message.
_INVALID_ONLY_CHARS = "!@#$%^&*()+={}[]|:;<>?/~_"


@st.composite
def _ticker_input_with_invalid_chars(draw):
    """Build an input guaranteed to trigger the invalid-character branch.

    The strategy constructs a single ticker segment that:
      * Contains at least one character drawn from ``_INVALID_ONLY_CHARS``.
      * May also contain valid ticker characters around the invalid chunk.
      * May be padded with leading/trailing whitespace (which the
        validator strips before the pattern check).

    Returns:
        A tuple ``(invalid_chunk, raw_input)`` where ``invalid_chunk``
        is the non-empty string of invalid characters inserted into
        the input and ``raw_input`` is the string handed to
        ``validate_tickers``.
    """
    invalid_chunk = draw(
        st.text(alphabet=_INVALID_ONLY_CHARS, min_size=1, max_size=5)
    )
    valid_prefix = draw(st.text(alphabet=_VALID_TICKER_CHARS, max_size=5))
    valid_suffix = draw(st.text(alphabet=_VALID_TICKER_CHARS, max_size=5))
    leading_ws = draw(st.text(alphabet=_WHITESPACE_CHARS, max_size=3))
    trailing_ws = draw(st.text(alphabet=_WHITESPACE_CHARS, max_size=3))

    raw_input = (
        leading_ws + valid_prefix + invalid_chunk + valid_suffix + trailing_ws
    )
    return invalid_chunk, raw_input


@given(_ticker_input_with_invalid_chars())
def test_invalid_characters_detected(data):
    """Property 3: Invalid characters are detected and reported.

    For any ticker input containing at least one character outside the
    set ``[A-Za-z0-9\\-\\.]``, ``validate_tickers`` returns
    ``valid=False`` with an empty ticker list and an error message
    that contains at least one of the invalid characters present in
    the input.

    Validates: Requirements 1.4
    """
    invalid_chunk, raw_input = data

    result = validate_tickers(raw_input)

    # Input must be rejected because at least one character is outside
    # the allowed alphabet.
    assert result.valid is False, (
        f"Expected validation to fail for input {raw_input!r} which "
        f"contains invalid characters {invalid_chunk!r}, but it succeeded."
    )

    # No tickers are returned when validation fails.
    assert result.tickers == [], (
        f"Expected empty tickers list on failure, got {result.tickers!r}."
    )

    # A human-readable error message must be supplied.
    assert result.error_message is not None
    assert isinstance(result.error_message, str)
    assert result.error_message.strip() != ""

    # The error message must surface at least one of the invalid
    # characters that caused the rejection, so the user can identify
    # what to fix.
    assert any(ch in result.error_message for ch in invalid_chunk), (
        f"Expected error message {result.error_message!r} to contain "
        f"at least one of the invalid characters from {invalid_chunk!r}."
    )


from validation import MAX_TICKERS


@st.composite
def _too_many_valid_tickers_input(draw):
    """Build an input with more than ``MAX_TICKERS`` valid ticker segments.

    The strategy constructs a comma-separated input where:
      * Between ``MAX_TICKERS + 1`` and ``MAX_TICKERS + 20`` ticker
        segments are produced, guaranteeing the count-limit rule fires.
      * Each ticker core is drawn from the allowed alphabet so no
        individual segment triggers the invalid-character branch first.
      * Optional leading/trailing whitespace around each ticker is
        included to confirm that whitespace padding does not affect
        the count check (the validator strips whitespace before counting).

    Returns:
        A tuple ``(n_tickers, raw_input)`` where ``n_tickers`` is the
        actual number of non-empty ticker segments generated and
        ``raw_input`` is the comma-joined string fed to the validator.
    """
    ticker_core_strategy = st.text(
        alphabet=_VALID_TICKER_CHARS, min_size=1, max_size=10
    )
    whitespace_strategy = st.text(alphabet=_WHITESPACE_CHARS, max_size=3)

    n_tickers = draw(
        st.integers(min_value=MAX_TICKERS + 1, max_value=MAX_TICKERS + 20)
    )

    segments: list[str] = []
    for _ in range(n_tickers):
        core = draw(ticker_core_strategy)
        leading = draw(whitespace_strategy)
        trailing = draw(whitespace_strategy)
        segments.append(leading + core + trailing)

    return n_tickers, ",".join(segments)


@given(_too_many_valid_tickers_input())
def test_max_ticker_count_enforced(data):
    """Property 4: Maximum ticker count is enforced.

    For any list of more than ``MAX_TICKERS`` valid ticker symbols
    joined by commas, ``validate_tickers`` returns ``valid=False`` with
    an empty ticker list and an error message that mentions the
    maximum limit.

    Validates: Requirements 1.5
    """
    n_tickers, raw_input = data

    result = validate_tickers(raw_input)

    # Sanity check on the generator: we should indeed have produced
    # more than ``MAX_TICKERS`` segments.
    assert n_tickers > MAX_TICKERS

    # The validator must reject the input because it exceeds the cap.
    assert result.valid is False, (
        f"Expected validation to fail for input with {n_tickers} tickers "
        f"(> {MAX_TICKERS}), but it succeeded."
    )

    # No tickers are returned on failure.
    assert result.tickers == [], (
        f"Expected empty tickers list on failure, got {result.tickers!r}."
    )

    # A human-readable error message must be supplied.
    assert result.error_message is not None
    assert isinstance(result.error_message, str)
    assert result.error_message.strip() != ""

    # The error message must reference the maximum limit so the user
    # understands why their input was rejected. We check for the numeric
    # limit, which is the most specific signal that this branch fired
    # rather than some unrelated error path.
    assert str(MAX_TICKERS) in result.error_message, (
        f"Expected error message {result.error_message!r} to mention "
        f"the maximum limit ({MAX_TICKERS})."
    )
