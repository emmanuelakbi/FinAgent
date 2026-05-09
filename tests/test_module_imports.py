"""
Verification test for task 12.1: Ensure all 10 tool functions are importable
from the tools package, have @tool decorator, and proper docstrings.

Requirements: 1.1, 1.5
"""

import inspect


def test_all_tools_importable_from_package():
    """Verify all 10 tool functions can be imported from the tools package."""
    from tools import (
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    )

    # All 10 should be non-None and callable
    all_tools = [
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    ]
    assert len(all_tools) == 10
    for tool_fn in all_tools:
        assert tool_fn is not None
        assert callable(tool_fn)


def test_all_tools_have_tool_decorator():
    """Verify each function has the @tool decorator by inspecting source code."""
    from tools import market_scanner, fundamental_analyst, technical_analyst, risk_manager

    # Map each tool function name to its source module
    tool_modules = {
        "search_news": market_scanner,
        "get_price_change": market_scanner,
        "get_volume": market_scanner,
        "get_financials": fundamental_analyst,
        "get_earnings": fundamental_analyst,
        "get_peers": fundamental_analyst,
        "get_price_history": technical_analyst,
        "calculate_indicators": technical_analyst,
        "calculate_position_size": risk_manager,
        "set_stop_loss": risk_manager,
    }

    for func_name, module in tool_modules.items():
        func = getattr(module, func_name)
        source = inspect.getsource(func)
        # The @tool decorator should appear in the source before the def
        assert "@tool(" in source, (
            f"{func_name} is missing @tool decorator in its source"
        )


def test_all_tools_have_docstrings():
    """Verify each tool function has a non-empty docstring."""
    from tools import (
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    )

    all_tools = {
        "search_news": search_news,
        "get_price_change": get_price_change,
        "get_volume": get_volume,
        "get_financials": get_financials,
        "get_earnings": get_earnings,
        "get_peers": get_peers,
        "get_price_history": get_price_history,
        "calculate_indicators": calculate_indicators,
        "calculate_position_size": calculate_position_size,
        "set_stop_loss": set_stop_loss,
    }

    for name, tool_fn in all_tools.items():
        doc = tool_fn.__doc__
        assert doc is not None, f"{name} has no docstring"
        assert len(doc.strip()) > 10, f"{name} has too short a docstring: '{doc}'"


def test_all_exports_in_dunder_all():
    """Verify __all__ in tools package lists all 10 functions."""
    import tools

    expected = {
        "search_news",
        "get_price_change",
        "get_volume",
        "get_financials",
        "get_earnings",
        "get_peers",
        "get_price_history",
        "calculate_indicators",
        "calculate_position_size",
        "set_stop_loss",
    }

    assert hasattr(tools, "__all__"), "tools package missing __all__"
    actual = set(tools.__all__)
    assert actual == expected, f"__all__ mismatch. Missing: {expected - actual}, Extra: {actual - expected}"


def test_tools_are_callable_functions():
    """Verify each tool is a callable function with proper signature."""
    from tools import (
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    )

    all_tools = [
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    ]

    for tool_fn in all_tools:
        assert callable(tool_fn), f"{tool_fn} is not callable"
        assert inspect.isfunction(tool_fn), f"{tool_fn} is not a function"


def test_tools_have_return_type_str():
    """Verify each tool function has str return type annotation."""
    from tools import (
        search_news,
        get_price_change,
        get_volume,
        get_financials,
        get_earnings,
        get_peers,
        get_price_history,
        calculate_indicators,
        calculate_position_size,
        set_stop_loss,
    )

    all_tools = {
        "search_news": search_news,
        "get_price_change": get_price_change,
        "get_volume": get_volume,
        "get_financials": get_financials,
        "get_earnings": get_earnings,
        "get_peers": get_peers,
        "get_price_history": get_price_history,
        "calculate_indicators": calculate_indicators,
        "calculate_position_size": calculate_position_size,
        "set_stop_loss": set_stop_loss,
    }

    for name, tool_fn in all_tools.items():
        sig = inspect.signature(tool_fn)
        assert sig.return_annotation == str, (
            f"{name} return annotation is {sig.return_annotation}, expected str"
        )
