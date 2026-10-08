"""Tool: check clinical criteria for a procedure.

The criteria are deterministic rules (an accepted diagnosis set plus numeric
thresholds). A procedure is clinically justified only when the diagnosis is
accepted and every rule passes. Keeping this rule-based — not left to an LLM —
is deliberate: a clinical decision must be explainable and reproducible.
"""

from __future__ import annotations

import operator

from preauth.data import CRITERIA

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def check_clinical_criteria(procedure_code: str, diagnosis_code: str, clinical_data: dict) -> dict:
    """Return whether the clinical criteria are met, and which ones failed."""
    criteria = CRITERIA.get(procedure_code)
    if criteria is None:
        return {"met": False, "failed": ["unknown procedure"], "detail": "no criteria on file"}

    failed: list[str] = []
    if diagnosis_code not in criteria["diagnoses"]:
        failed.append(f"diagnosis {diagnosis_code} is not accepted for this procedure")

    for field, op, value, description in criteria["rules"]:
        actual = clinical_data.get(field)
        if actual is None or not _OPS[op](actual, value):
            failed.append(f"{description} (got {actual})")

    met = not failed
    detail = "all criteria met" if met else "criteria not met"
    return {"met": met, "failed": failed, "detail": detail}
