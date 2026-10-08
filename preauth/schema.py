"""Request and decision models for prior authorization.

A prior-authorization ("pre-auth") request asks an insurer to approve a
procedure before it happens. The agent gathers the facts with tools, recommends
an outcome, writes a rationale, and then a human decides. The models here carry
that request and the decision packet the agent produces.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Recommendation = Literal["approve", "deny", "pend"]
Status = Literal["pending_human_approval", "needs_info", "approved", "denied"]


class PriorAuthRequest(BaseModel):
    member_id: str
    procedure_code: str
    diagnosis_code: str
    clinical_data: dict[str, float] = Field(default_factory=dict)
    submitted_documents: list[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "member_id": "M1001",
                "procedure_code": "BAR001",
                "diagnosis_code": "E66.01",
                "clinical_data": {"bmi": 42.0, "months_conservative_treatment": 8},
                "submitted_documents": [
                    "bmi_report", "nutrition_evaluation", "psych_evaluation"
                ],
            }
        }
    }


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict


class HumanDecision(BaseModel):
    decision: Literal["approved", "denied"]
    reviewer: str
    note: str = ""
    decided_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DecisionPacket(BaseModel):
    request_id: str
    member_id: str
    procedure_code: str
    diagnosis_code: str
    recommendation: Recommendation
    rationale: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    missing_documents: list[str] = Field(default_factory=list)
    status: Status
    human_decision: HumanDecision | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
