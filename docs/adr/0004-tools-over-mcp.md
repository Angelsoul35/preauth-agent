# ADR-0004: Expose the tools over MCP

* Status: accepted
* Date: 2026-02-19

## Context

The checks (eligibility, coverage, documents, clinical criteria) are useful beyond this one
service. Re-implementing them for each client would cause drift and bugs.

## Decision

The checks are plain functions in a registry. The agent calls them directly, and an MCP
server exposes the same functions. One definition serves the service and any external MCP
client (an IDE, another agent, a desktop assistant).

## Consequences

* **Good:** one source of truth for the tools; they are reusable and testable on their own;
  other systems can call them over a standard protocol.
* **Cost:** MCP is an extra optional dependency. It is behind an extra (`[mcp]`) so the core
  service does not require it.
