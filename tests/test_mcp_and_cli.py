from __future__ import annotations

import pytest

from preauth import cli


def test_cli_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "APPROVE" in out
    assert "final status" in out


def test_mcp_server_builds():
    mcp = pytest.importorskip("mcp")  # optional extra
    assert mcp is not None
    from preauth.mcp_server import build_server

    server = build_server()
    assert server is not None
