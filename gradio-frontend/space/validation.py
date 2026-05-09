"""Input validation module for the FinAgent Gradio frontend.

Provides pure validation functions (no Gradio dependencies) for ticker
symbols and portfolio value inputs. These functions are used by the
event handlers in ``app.py`` and are directly testable in isolation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Valid ticker characters: letters, digits, hyphens, periods.
TICKER_PATTERN = re.compile(r"^[A-Za-z0-9\-\.]+$")

# Maximum number of tickers allowed per analysis run.
MAX_TICKERS = 10


@dataclass
class ValidationResult:
    """Result of input validation.

    Attributes:
        valid: Whether the input passed all validation checks.
        tickers: Normalized (trimmed, uppercased) ticker list. Empty if invalid.
        error_message: Human-readable error message. ``None`` when valid.
    """

    valid: bool
    tickers: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


def validate_tickers(raw_input: str) -> ValidationResult:
    """Validate and normalize a comma-separated ticker input string.

    Rules enforced:
        - Input must not be empty or whitespace-only
        - Each ticker is trimmed and converted to uppercase
        - Empty segments after splitting on commas are discarded
        - Only letters, digits, hyphens, and periods are allowed
        - Maximum of ``MAX_TICKERS`` tickers per submission

    Args:
        raw_input: Raw user input string (e.g., ``"aapl, nvda, tsla"``).

    Returns:
        A ``ValidationResult`` with normalized tickers on success, or an
        error message on failure.
    """
    # Reject empty or whitespace-only input early.
    if not raw_input or not raw_input.strip():
        return ValidationResult(
            valid=False,
            tickers=[],
            error_message="Please enter at least one ticker symbol.",
        )

    # Split, trim, uppercase, and drop empty segments (e.g., trailing commas).
    raw_tickers = [segment.strip().upper() for segment in raw_input.split(",")]
    tickers = [t for t in raw_tickers if t]

    if not tickers:
        return ValidationResult(
            valid=False,
            tickers=[],
            error_message="Please enter at least one ticker symbol.",
        )

    # Character validation — collect every ticker that contains disallowed chars.
    invalid_tickers: list[str] = []
    for ticker in tickers:
        if not TICKER_PATTERN.match(ticker):
            invalid_chars = sorted(set(re.findall(r"[^A-Za-z0-9\-\.]", ticker)))
            invalid_tickers.append(f"{ticker} (invalid: {''.join(invalid_chars)})")

    if invalid_tickers:
        return ValidationResult(
            valid=False,
            tickers=[],
            error_message=f"Invalid characters in: {', '.join(invalid_tickers)}",
        )

    # Enforce maximum ticker count.
    if len(tickers) > MAX_TICKERS:
        return ValidationResult(
            valid=False,
            tickers=[],
            error_message=(
                f"Maximum {MAX_TICKERS} tickers per analysis. "
                f"You entered {len(tickers)}."
            ),
        )

    return ValidationResult(valid=True, tickers=tickers, error_message=None)


def validate_portfolio_value(value: float) -> Optional[str]:
    """Validate a portfolio value.

    A portfolio value is considered valid when it is non-negative (``>= 0``).
    Negative values are rejected with a human-readable error message.

    Args:
        value: The portfolio value to validate.

    Returns:
        An error message string if the value is invalid, or ``None`` if valid.
    """
    if value < 0:
        return "Portfolio value must be non-negative."
    return None
