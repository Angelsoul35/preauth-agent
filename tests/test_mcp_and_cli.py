from __future__ import annotations

import pytest

from preauth import cli


def test_cli_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "APPROVE" in out
    assert "final status" in out


def test_mcp_server_registers_all_preauth_tools():
    pytest.importorskip("mcp")  # optional extra
    import asyncio

    from preauth.mcp_server import build_server

    tools = asyncio.run(build_server().list_tools())
    names = {tool.name for tool in tools}
    # Every pre-auth tool the agent uses must be exposed over MCP — no more, no less.
    assert names == {
        "eligibility",
        "coverage",
        "documents_required",
        "documents_missing",
        "clinical",
    }
