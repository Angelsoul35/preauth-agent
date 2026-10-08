"""Expose the pre-auth tools over the Model Context Protocol (MCP).

Running ``python -m preauth.mcp_server`` starts an MCP server over stdio that
offers the same tools the agent uses. Any MCP client (an IDE, another agent, a
desktop assistant) can then call ``check_eligibility``, ``check_coverage``,
``check_documents``, and ``check_clinical_criteria`` directly. This keeps one
source of truth for the tools and makes them reusable outside this service.

Requires the optional extra: ``pip install 'preauth-agent[mcp]'``.
"""

from __future__ import annotations

from .tools import (
    check_clinical_criteria,
    check_coverage,
    check_documents,
    check_eligibility,
    required_documents,
)


def build_server():
    """Build and return the MCP server with the pre-auth tools registered."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("preauth-tools")

    @server.tool(description="Check whether a member is active and return their plan.")
    def eligibility(member_id: str) -> dict:
        return check_eligibility(member_id)

    @server.tool(description="Check whether a plan covers a procedure.")
    def coverage(procedure_code: str, plan: str) -> dict:
        return check_coverage(procedure_code, plan)

    @server.tool(description="List the documents a procedure requires.")
    def documents_required(procedure_code: str) -> dict:
        return required_documents(procedure_code)

    @server.tool(description="Return which required documents are missing.")
    def documents_missing(procedure_code: str, submitted: list[str]) -> dict:
        return check_documents(procedure_code, submitted)

    @server.tool(description="Check the clinical criteria for a procedure.")
    def clinical(procedure_code: str, diagnosis_code: str, clinical_data: dict) -> dict:
        return check_clinical_criteria(procedure_code, diagnosis_code, clinical_data)

    return server


def main() -> None:  # pragma: no cover - runs an stdio server
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
