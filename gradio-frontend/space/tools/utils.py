"""
Utility functions for Agent Tools.

Provides shared helpers: ticker validation, currency formatting,
safe dictionary access, and percentage formatting.
"""


def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Validate and normalize a ticker string.

    Strips whitespace from input. If empty or whitespace-only after strip,
    returns an error tuple. Otherwise returns the uppercase normalized ticker.

    Args:
        ticker: Raw ticker string from user/agent input.

    Returns:
        (True, normalized_ticker) if valid
        (False, error_message) if invalid
    """
    stripped = ticker.strip()
    if not stripped:
        return (False, "Error: Invalid ticker provided. Ticker must be a non-empty string.")
    return (True, stripped.upper())


def format_currency(value: float, precision: int = 2) -> str:
    """Format monetary values with appropriate units (B/M/K).

    Args:
        value: The monetary value to format. If None, returns "N/A".
        precision: Number of decimal places (default 2).

    Returns:
        Formatted string like "$1.50B", "$250.00M", "$45.00K", or "$123.45".

    Examples:
        >>> format_currency(1_500_000_000)
        '$1.50B'
        >>> format_currency(250_000_000)
        '$250.00M'
        >>> format_currency(45_000)
        '$45.00K'
        >>> format_currency(123.456)
        '$123.46'
        >>> format_currency(None)
        'N/A'
    """
    if value is None:
        return "N/A"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000:
        return f"{sign}${abs_value / 1_000_000_000:.{precision}f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}${abs_value / 1_000_000:.{precision}f}M"
    elif abs_value >= 1_000:
        return f"{sign}${abs_value / 1_000:.{precision}f}K"
    else:
        return f"{sign}${abs_value:.{precision}f}"


def safe_get(info: dict, key: str, default: str = "N/A") -> str:
    """Safely extract a value from a dict, returning default if missing or None.

    Args:
        info: Dictionary to look up.
        key: Key to retrieve.
        default: Value to return if key is missing or value is None.

    Returns:
        String representation of the value, or default.
    """
    value = info.get(key)
    if value is None:
        return default
    return str(value)


def format_percent(value: float, precision: int = 2) -> str:
    """Format a decimal or percentage value as a string with % sign.

    If value is None, returns "N/A". If value appears to be a decimal
    (absolute value less than 1), it is multiplied by 100 first to
    convert to a percentage.

    Args:
        value: The value to format. Decimals (e.g., 0.082) are converted
               to percentages (8.20%). Values >= 1 are treated as already
               being percentages.
        precision: Number of decimal places (default 2).

    Returns:
        Formatted string like "8.20%" or "N/A".

    Examples:
        >>> format_percent(0.082)
        '8.20%'
        >>> format_percent(25.5)
        '25.50%'
        >>> format_percent(None)
        'N/A'
    """
    if value is None:
        return "N/A"

    # If value is a decimal (abs < 1), multiply by 100 to get percentage
    if abs(value) < 1:
        value = value * 100

    return f"{value:.{precision}f}%"
