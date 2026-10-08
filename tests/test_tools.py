from __future__ import annotations

from preauth.tools import (
    TOOLS,
    check_clinical_criteria,
    check_coverage,
    check_documents,
    check_eligibility,
)


def test_eligibility():
    active = check_eligibility("M1001")
    assert active == {"eligible": True, "plan": "premium", "reason": "active member"}
    assert check_eligibility("M1003")["eligible"] is False  # inactive
    assert check_eligibility("nobody")["eligible"] is False  # not found


def test_coverage():
    assert check_coverage("BAR001", "premium")["covered"] is True
    assert check_coverage("BAR001", "basic")["covered"] is False


def test_documents():
    full = check_documents("BAR001", ["bmi_report", "nutrition_evaluation", "psych_evaluation"])
    assert full["complete"] is True and full["missing"] == []
    partial = check_documents("BAR001", ["bmi_report"])
    assert partial["complete"] is False
    assert "psych_evaluation" in partial["missing"]


def test_clinical_criteria_met():
    result = check_clinical_criteria(
        "BAR001", "E66.01", {"bmi": 42.0, "months_conservative_treatment": 8}
    )
    assert result["met"] is True
    assert result["failed"] == []


def test_clinical_criteria_failed():
    low_bmi = check_clinical_criteria(
        "BAR001", "E66.01", {"bmi": 30.0, "months_conservative_treatment": 8}
    )
    assert low_bmi["met"] is False
    assert any("BMI" in f for f in low_bmi["failed"])

    wrong_dx = check_clinical_criteria(
        "BAR001", "Z00.0", {"bmi": 42.0, "months_conservative_treatment": 8}
    )
    assert wrong_dx["met"] is False


def test_registry_lists_all_tools():
    assert set(TOOLS) == {
        "check_eligibility", "check_coverage", "required_documents",
        "check_documents", "check_clinical_criteria",
    }
    for spec in TOOLS.values():
        assert callable(spec["func"])
        assert spec["description"]
