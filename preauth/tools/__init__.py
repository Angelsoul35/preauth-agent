"""Tool registry shared by the agent and the MCP server."""

from __future__ import annotations

from .clinical import check_clinical_criteria
from .coverage import check_coverage
from .documents import check_documents, required_documents
from .eligibility import check_eligibility

TOOLS: dict[str, dict] = {
    "check_eligibility": {
        "func": check_eligibility,
        "description": "Check whether a member is active and return their plan.",
    },
    "check_coverage": {
        "func": check_coverage,
        "description": "Check whether a plan covers a procedure.",
    },
    "required_documents": {
        "func": required_documents,
        "description": "List the documents a procedure requires.",
    },
    "check_documents": {
        "func": check_documents,
        "description": "Return which required documents are missing.",
    },
    "check_clinical_criteria": {
        "func": check_clinical_criteria,
        "description": "Check the clinical criteria for a procedure.",
    },
}

__all__ = [
    "TOOLS",
    "check_eligibility",
    "check_coverage",
    "required_documents",
    "check_documents",
    "check_clinical_criteria",
]
