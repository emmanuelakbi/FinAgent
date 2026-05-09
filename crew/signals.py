"""TradingSignal dataclass and parser for structured output handling."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(str, Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Structured trading signal output."""

    ticker: str
    action: Action
    confidence: int  # 0-100
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    reasoning: Optional[dict[str, str]] = None  # {agent_name: summary}

    @staticmethod
    def validate_confidence(value: int) -> int:
        """Clamp confidence to 0-100 range."""
        return max(0, min(100, value))


class TradingSignalParser:
    """Parses raw LLM output into structured TradingSignal objects."""

    # Primary format: "AAPL — BUY (Confidence: 75%)"
    PRIMARY_PATTERN = re.compile(
        r"([A-Z\-\.]+)\s*[—–-]\s*(BUY|SELL|HOLD)\s*\(Confidence:\s*(\d{1,3})%\)",
        re.IGNORECASE,
    )

    # Fallback patterns for less structured output
    ACTION_PATTERN = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)
    CONFIDENCE_PATTERN = re.compile(r"(\d{1,3})\s*%")
    PRICE_PATTERN = re.compile(r"\$\s*([\d,]+\.?\d*)")

    def parse(self, raw_output: str, ticker: str) -> Optional[TradingSignal]:
        """Parse raw output into a TradingSignal.

        Attempts primary pattern first, falls back to heuristic extraction.

        Args:
            raw_output: Raw text from Chief Strategist agent
            ticker: Expected ticker symbol

        Returns:
            TradingSignal if parsing succeeds, None if output is unparseable
        """
        signal = self._parse_primary(raw_output, ticker)
        if signal is not None:
            return signal
        return self._parse_fallback(raw_output, ticker)

    def _parse_primary(self, raw_output: str, ticker: str) -> Optional[TradingSignal]:
        """Attempt to parse using the primary structured format."""
        match = self.PRIMARY_PATTERN.search(raw_output)
        if not match:
            return None

        parsed_ticker = match.group(1).upper()
        action_str = match.group(2).upper()
        confidence_raw = int(match.group(3))

        action = Action(action_str)
        confidence = TradingSignal.validate_confidence(confidence_raw)

        prices = self._extract_prices(raw_output)
        reasoning = self._extract_reasoning(raw_output)

        return TradingSignal(
            ticker=parsed_ticker,
            action=action,
            confidence=confidence,
            entry_price=prices.get("entry"),
            stop_loss=prices.get("stop_loss"),
            target_price=prices.get("target"),
            reasoning=reasoning,
        )

    def _parse_fallback(self, raw_output: str, ticker: str) -> Optional[TradingSignal]:
        """Attempt heuristic extraction from unstructured output."""
        action_match = self.ACTION_PATTERN.search(raw_output)
        if not action_match:
            return None

        action = Action(action_match.group(1).upper())

        confidence_match = self.CONFIDENCE_PATTERN.search(raw_output)
        if confidence_match:
            confidence = TradingSignal.validate_confidence(int(confidence_match.group(1)))
        else:
            confidence = 50  # Default confidence when not specified

        prices = self._extract_prices(raw_output)

        return TradingSignal(
            ticker=ticker.upper(),
            action=action,
            confidence=confidence,
            entry_price=prices.get("entry"),
            stop_loss=prices.get("stop_loss"),
            target_price=prices.get("target"),
        )

    def _extract_prices(self, raw_output: str) -> dict[str, Optional[float]]:
        """Extract entry, stop-loss, and target prices from text.

        Finds all $XX.XX patterns and assigns:
        - First as entry price
        - Second as stop_loss
        - Third as target price
        """
        matches = self.PRICE_PATTERN.findall(raw_output)
        prices: dict[str, Optional[float]] = {
            "entry": None,
            "stop_loss": None,
            "target": None,
        }

        # Parse matched price strings, removing commas
        parsed = []
        for m in matches:
            cleaned = m.replace(",", "")
            try:
                parsed.append(float(cleaned))
            except ValueError:
                continue

        if len(parsed) >= 1:
            prices["entry"] = parsed[0]
        if len(parsed) >= 2:
            prices["stop_loss"] = parsed[1]
        if len(parsed) >= 3:
            prices["target"] = parsed[2]

        return prices

    def _extract_reasoning(self, raw_output: str) -> Optional[dict[str, str]]:
        """Extract per-agent reasoning summaries from text.

        Looks for lines like:
        - Market: ...
        - Fundamental: ...
        - Technical: ...
        - Risk: ...
        """
        reasoning_pattern = re.compile(
            r"^-\s*(Market|Fundamental|Technical|Risk)\s*:\s*(.+)$",
            re.MULTILINE | re.IGNORECASE,
        )
        matches = reasoning_pattern.findall(raw_output)

        if not matches:
            return None

        reasoning = {}
        for key, value in matches:
            reasoning[key.strip()] = value.strip()

        return reasoning if reasoning else None
