from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from preauth.agent import PreAuthAgent
from preauth.llm.fake import FakeLLM
from preauth.main import app
from preauth.schema import PriorAuthRequest
from preauth.service import PreAuthService

APPROVE = {
    "member_id": "M1001", "procedure_code": "BAR001", "diagnosis_code": "E66.01",
    "clinical_data": {"bmi": 42.0, "months_conservative_treatment": 8},
    "submitted_documents": ["bmi_report", "nutrition_evaluation", "psych_evaluation"],
}
PEND = {**APPROVE, "submitted_documents": ["bmi_report"]}
DENY_CLINICAL = {**APPROVE, "clinical_data": {"bmi": 30.0, "months_conservative_treatment": 8}}
DENY_COVERAGE = {**APPROVE, "member_id": "M1002"}


def _agent():
    return PreAuthAgent(rationale_mode="template")


def test_agent_recommends_approve():
    packet = _agent().run(PriorAuthRequest(**APPROVE), "r1")
    assert packet.recommendation == "approve"
    assert packet.status == "pending_human_approval"
    assert len(packet.tool_results) == 4


def test_agent_pends_on_missing_docs():
    packet = _agent().run(PriorAuthRequest(**PEND), "r2")
    assert packet.recommendation == "pend"
    assert packet.status == "needs_info"
    assert "psych_evaluation" in packet.missing_documents


def test_agent_denies_on_clinical_and_coverage():
    assert _agent().run(PriorAuthRequest(**DENY_CLINICAL), "r3").recommendation == "deny"
    assert _agent().run(PriorAuthRequest(**DENY_COVERAGE), "r4").recommendation == "deny"


def test_agent_never_auto_approves():
    # Even a clean approve is only a recommendation; status waits for a human.
    packet = _agent().run(PriorAuthRequest(**APPROVE), "r5")
    assert packet.status != "approved"
    assert packet.human_decision is None


def test_agent_uses_llm_rationale_when_provided():
    agent = PreAuthAgent(llm=FakeLLM("LLM-written rationale."))
    packet = agent.run(PriorAuthRequest(**APPROVE), "r6")
    assert packet.rationale == "LLM-written rationale."


def test_service_approval_flow():
    service = PreAuthService()
    packet = service.submit(PriorAuthRequest(**APPROVE))
    assert packet.status == "pending_human_approval"
    final = service.approve(packet.request_id, "approved", "dr.smith", "ok")
    assert final.status == "approved"
    assert final.human_decision.reviewer == "dr.smith"


def test_service_cannot_approve_pending_info():
    service = PreAuthService()
    packet = service.submit(PriorAuthRequest(**PEND))
    with pytest.raises(ValueError):
        service.approve(packet.request_id, "approved", "dr.smith")


def test_service_get_unknown():
    assert PreAuthService().get("nope") is None


client = TestClient(app)


def test_api_flow():
    submit = client.post("/preauth", json=APPROVE)
    assert submit.status_code == 200
    request_id = submit.json()["request_id"]
    assert submit.json()["recommendation"] == "approve"

    got = client.get(f"/preauth/{request_id}")
    assert got.status_code == 200

    approved = client.post(
        f"/preauth/{request_id}/approve",
        json={"decision": "approved", "reviewer": "dr.smith"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"


def test_api_404_and_409():
    assert client.get("/preauth/nope").status_code == 404
    pend = client.post("/preauth", json=PEND).json()
    conflict = client.post(
        f"/preauth/{pend['request_id']}/approve",
        json={"decision": "approved", "reviewer": "x"},
    )
    assert conflict.status_code == 409
