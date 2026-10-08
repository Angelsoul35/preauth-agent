"""Tool: required and missing documents for a procedure."""

from __future__ import annotations

from preauth.data import REQUIRED_DOCS


def required_documents(procedure_code: str) -> dict:
    """List the documents a procedure requires."""
    return {"required": REQUIRED_DOCS.get(procedure_code, [])}


def check_documents(procedure_code: str, submitted: list[str]) -> dict:
    """Return which required documents are missing from the submitted set."""
    required = REQUIRED_DOCS.get(procedure_code, [])
    missing = [doc for doc in required if doc not in submitted]
    reason = "all required documents present" if not missing else f"missing: {', '.join(missing)}"
    return {"complete": not missing, "missing": missing, "required": required, "reason": reason}
