"""Tool: check whether a plan covers a procedure."""

from __future__ import annotations

from preauth.data import COVERAGE, PROCEDURES


def check_coverage(procedure_code: str, plan: str) -> dict:
    """Return whether the given plan covers the procedure."""
    covered = procedure_code in COVERAGE.get(plan, set())
    name = PROCEDURES.get(procedure_code, procedure_code)
    verb = "is" if covered else "is not"
    return {"covered": covered, "reason": f"{name} {verb} covered by the {plan} plan"}
