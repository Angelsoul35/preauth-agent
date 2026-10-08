"""PreAuth Agent HTTP API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .schema import DecisionPacket, PriorAuthRequest
from .service import PreAuthService

app = FastAPI(
    title="PreAuth Agent",
    version="1.0.0",
    summary="Prior-authorization agent: rule-based checks via tools, with a human approval step.",
)

service = PreAuthService()


class ApprovalRequest(BaseModel):
    decision: str  # "approved" | "denied"
    reviewer: str
    note: str = ""


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/preauth", response_model=DecisionPacket, tags=["preauth"])
def submit(request: PriorAuthRequest) -> DecisionPacket:
    return service.submit(request)


@app.get("/preauth/{request_id}", response_model=DecisionPacket, tags=["preauth"])
def get(request_id: str) -> DecisionPacket:
    packet = service.get(request_id)
    if packet is None:
        raise HTTPException(status_code=404, detail="request not found")
    return packet


@app.post("/preauth/{request_id}/approve", response_model=DecisionPacket, tags=["preauth"])
def approve(request_id: str, approval: ApprovalRequest) -> DecisionPacket:
    try:
        packet = service.approve(request_id, approval.decision, approval.reviewer, approval.note)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if packet is None:
        raise HTTPException(status_code=404, detail="request not found")
    return packet
