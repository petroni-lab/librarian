"""Minimal vendored tool exports needed by the literature subset."""

from __future__ import annotations

from typing import Any

__all__ = [
    "async_make_asta_mcp_tools",
    "make_asta_mcp_tools",
    "make_asta_toolsource",
    "make_native_search_tools",
    "report_editor",
    "table_editor",
]


async def async_make_asta_mcp_tools(*args: Any, **kwargs: Any):
    from .asta_tools import async_make_asta_mcp_tools as _impl

    return await _impl(*args, **kwargs)


def make_asta_mcp_tools(*args: Any, **kwargs: Any):
    from .asta_tools import make_asta_mcp_tools as _impl

    return _impl(*args, **kwargs)


def make_asta_toolsource(*args: Any, **kwargs: Any):
    from .asta_tools import make_asta_toolsource as _impl

    return _impl(*args, **kwargs)


def make_native_search_tools(*args: Any, **kwargs: Any):
    from .native_provider_tools import make_native_search_tools as _impl

    return _impl(*args, **kwargs)


def report_editor(*args: Any, **kwargs: Any):
    from .report import report_editor as _impl

    return _impl(*args, **kwargs)


def table_editor(*args: Any, **kwargs: Any):
    from .table import table_editor as _impl

    return _impl(*args, **kwargs)
