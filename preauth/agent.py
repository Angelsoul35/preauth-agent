"""The pre-authorization agent.

The agent gathers facts by calling tools, then applies a fixed decision policy,
writes a rationale, and hands the case to a human. Two design choices matter:

* **The decision is rule-based, not left to the LLM.** Eligibility, coverage, and
  clinical criteria are deterministic checks. A clinical/benefit decision must be
  explainable and reproducible, so the LLM never makes it.
* **A human always decides.** The agent recommends approve, deny, or pend; the
  final status is set by a reviewer. The agent never approves on its own.
"""

from __future__ import annotations

from .llm import LLMClient, build_llm
from .schema import DecisionPacket, PriorAuthRequest, ToolResult
from .tools import (
    check_clinical_criteria,
    check_coverage,
    check_documents,
    check_eligibility,
)


class PreAuthAgent:
    def __init__(
        self,
        rationale_mode: str = "template",
        llm: LLMClient | None = None,
        provider_spec: str | None = None,
    ) -> None:
        self.rationale_mode = rationale_mode
        self._llm = llm
        self._provider_spec = provider_spec

    def run(self, request: PriorAuthRequest, request_id: str) -> DecisionPacket:
        results: list[ToolResult] = []
        missing: list[str] = []

        elig = check_eligibility(request.member_id)
        results.append(ToolResult(tool="check_eligibility", ok=elig["eligible"], data=elig))

        if not elig["eligible"]:
            recommendation = "deny"
        else:
            plan = elig["plan"]
            cov = check_coverage(request.procedure_code, plan)
            results.append(ToolResult(tool="check_coverage", ok=cov["covered"], data=cov))

            docs = check_documents(request.procedure_code, request.submitted_documents)
            results.append(ToolResult(tool="check_documents", ok=docs["complete"], data=docs))
            missing = docs["missing"]

            clin = check_clinical_criteria(
                request.procedure_code, request.diagnosis_code, request.clinical_data
            )
            results.append(ToolResult(tool="check_clinical_criteria", ok=clin["met"], data=clin))

            if not cov["covered"]:
                recommendation = "deny"
            elif not docs["complete"]:
                recommendation = "pend"
            elif not clin["met"]:
                recommendation = "deny"
            else:
                recommendation = "approve"

        status = "needs_info" if recommendation == "pend" else "pending_human_approval"
        rationale = self._rationale(request, results, recommendation, missing)

        return DecisionPacket(
            request_id=request_id,
            member_id=request.member_id,
            procedure_code=request.procedure_code,
            diagnosis_code=request.diagnosis_code,
            recommendation=recommendation,
            rationale=rationale,
            tool_results=results,
            missing_documents=missing,
            status=status,
        )

    # ---------------------------------------------------------------- rationale
    def _rationale(self, request, results, recommendation, missing) -> str:
        if self.rationale_mode == "llm" or self._llm is not None:
            try:
                return self._llm_rationale(request, results, recommendation)
            except Exception:  # pragma: no cover - falls back when no provider
                pass
        return self._template_rationale(results, recommendation, missing)

    @staticmethod
    def _template_rationale(results, recommendation, missing) -> str:
        lines = [f"Recommendation: {recommendation.upper()}.", "Findings:"]
        for r in results:
            mark = "OK" if r.ok else "FAIL"
            reason = r.data.get("reason") or r.data.get("detail") or ""
            lines.append(f"  [{mark}] {r.tool}: {reason}")
        if recommendation == "pend" and missing:
            lines.append(f"Missing documents: {', '.join(missing)}.")
        lines.append("A human reviewer must confirm this recommendation.")
        return "\n".join(lines)

    def _llm_rationale(self, request, results, recommendation) -> str:
        client = self._llm or (
            build_llm(self._provider_spec) if self._provider_spec else build_llm()
        )
        facts = "\n".join(f"- {r.tool}: ok={r.ok}, {r.data}" for r in results)
        system = (
            "You are a clinical prior-authorization assistant. Using ONLY the findings "
            "given, write a short, neutral rationale (80-140 words) for the recommendation. "
            "Do not invent facts. State that a human reviewer must confirm the decision."
        )
        prompt = (
            f"Procedure: {request.procedure_code}. Diagnosis: {request.diagnosis_code}.\n"
            f"Recommendation: {recommendation}.\nFindings:\n{facts}\n\nWrite the rationale."
        )
        return client.complete(system, prompt, max_tokens=300).text.strip()
