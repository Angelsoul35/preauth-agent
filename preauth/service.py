"""The service: run the agent, store packets, and apply the human decision."""

from __future__ import annotations

import uuid

from .agent import PreAuthAgent
from .config import Settings, get_settings
from .llm import LLMClient
from .schema import DecisionPacket, HumanDecision, PriorAuthRequest


class PreAuthService:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.agent = PreAuthAgent(
            rationale_mode=self.settings.rationale_mode,
            llm=llm,
            provider_spec=self.settings.llm_providers,
        )
        self._store: dict[str, DecisionPacket] = {}

    def submit(self, request: PriorAuthRequest) -> DecisionPacket:
        request_id = str(uuid.uuid4())
        packet = self.agent.run(request, request_id)
        self._store[request_id] = packet
        return packet

    def get(self, request_id: str) -> DecisionPacket | None:
        return self._store.get(request_id)

    def approve(
        self, request_id: str, decision: str, reviewer: str, note: str = ""
    ) -> DecisionPacket | None:
        """Apply a human decision. Only allowed while awaiting human approval."""
        packet = self._store.get(request_id)
        if packet is None:
            return None
        if packet.status != "pending_human_approval":
            raise ValueError(f"request {request_id} is '{packet.status}', not awaiting approval")
        if decision not in {"approved", "denied"}:
            raise ValueError("decision must be 'approved' or 'denied'")
        packet.human_decision = HumanDecision(decision=decision, reviewer=reviewer, note=note)
        packet.status = decision
        return packet
