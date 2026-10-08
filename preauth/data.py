"""Synthetic reference data: members, coverage, clinical criteria, documents.

All of this is made up for the demo. In a real system it would come from the
member database, the plan's benefit rules, and published clinical guidelines. The
criteria below are illustrative, not medical advice.
"""

from __future__ import annotations

PROCEDURES: dict[str, str] = {
    "BAR001": "Bariatric surgery",
    "MRI001": "MRI of the lumbar spine",
    "PT001": "Physical therapy (extended course)",
}

# Members: active flag and plan.
MEMBERS: dict[str, dict] = {
    "M1001": {"active": True, "plan": "premium"},
    "M1002": {"active": True, "plan": "basic"},
    "M1003": {"active": False, "plan": "premium"},
}

# Which procedures each plan covers.
COVERAGE: dict[str, set[str]] = {
    "basic": {"MRI001", "PT001"},
    "premium": {"BAR001", "MRI001", "PT001"},
}

# Documents required per procedure.
REQUIRED_DOCS: dict[str, list[str]] = {
    "BAR001": ["bmi_report", "nutrition_evaluation", "psych_evaluation"],
    "MRI001": ["referral", "prior_imaging"],
    "PT001": ["referral"],
}

# Clinical criteria per procedure: an accepted diagnosis set plus numeric rules.
# A rule is (field, operator, value, description). Operators: >=, <=, >, <, ==.
CRITERIA: dict[str, dict] = {
    "BAR001": {
        "diagnoses": ["E66.01", "E66.2"],
        "rules": [
            ("bmi", ">=", 40.0, "BMI at least 40"),
            (
                "months_conservative_treatment",
                ">=",
                6.0,
                "at least 6 months of documented conservative treatment",
            ),
        ],
    },
    "MRI001": {
        "diagnoses": ["M54.5", "M51.26"],
        "rules": [
            ("weeks_of_symptoms", ">=", 6.0, "symptoms for at least 6 weeks"),
        ],
    },
    "PT001": {
        "diagnoses": ["M54.5", "M25.561"],
        "rules": [],
    },
}


SAMPLE_REQUESTS: list[dict] = [
    {  # clean approve
        "member_id": "M1001",
        "procedure_code": "BAR001",
        "diagnosis_code": "E66.01",
        "clinical_data": {"bmi": 42.0, "months_conservative_treatment": 8},
        "submitted_documents": ["bmi_report", "nutrition_evaluation", "psych_evaluation"],
    },
    {  # pend: missing a document
        "member_id": "M1001",
        "procedure_code": "BAR001",
        "diagnosis_code": "E66.01",
        "clinical_data": {"bmi": 41.0, "months_conservative_treatment": 7},
        "submitted_documents": ["bmi_report", "nutrition_evaluation"],
    },
    {  # deny: criteria not met (BMI too low)
        "member_id": "M1001",
        "procedure_code": "BAR001",
        "diagnosis_code": "E66.01",
        "clinical_data": {"bmi": 34.0, "months_conservative_treatment": 8},
        "submitted_documents": ["bmi_report", "nutrition_evaluation", "psych_evaluation"],
    },
    {  # deny: procedure not covered by the basic plan
        "member_id": "M1002",
        "procedure_code": "BAR001",
        "diagnosis_code": "E66.01",
        "clinical_data": {"bmi": 45.0, "months_conservative_treatment": 12},
        "submitted_documents": ["bmi_report", "nutrition_evaluation", "psych_evaluation"],
    },
]
