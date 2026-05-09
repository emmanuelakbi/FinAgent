"""
Agent Tools Module for FinAgent.

This module provides 10 CrewAI tool functions across four domains:
- Market Scanner: search_news, get_price_change, get_volume
- Fundamental Analyst: get_financials, get_earnings, get_peers
- Technical Analyst: get_price_history, calculate_indicators
- Risk Manager: calculate_position_size, set_stop_loss

All tools return formatted strings and handle errors gracefully.
"""

# Market Scanner Tools
from tools.market_scanner import search_news, get_price_change, get_volume

# Fundamental Analyst Tools
from tools.fundamental_analyst import get_financials, get_earnings, get_peers

# Technical Analyst Tools
from tools.technical_analyst import get_price_history, calculate_indicators

# Risk Manager Tools
from tools.risk_manager import calculate_position_size, set_stop_loss

__all__ = [
    # Market Scanner
    "search_news",
    "get_price_change",
    "get_volume",
    # Fundamental Analyst
    "get_financials",
    "get_earnings",
    "get_peers",
    # Technical Analyst
    "get_price_history",
    "calculate_indicators",
    # Risk Manager
    "calculate_position_size",
    "set_stop_loss",
]
