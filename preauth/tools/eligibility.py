"""Tool: check member eligibility."""

from __future__ import annotations

from preauth.data import MEMBERS


def check_eligibility(member_id: str) -> dict:
    """Return whether a member is active and which plan they have."""
    member = MEMBERS.get(member_id)
    if member is None:
        return {"eligible": False, "plan": None, "reason": f"member {member_id} not found"}
    if not member["active"]:
        return {"eligible": False, "plan": member["plan"], "reason": "membership is not active"}
    return {"eligible": True, "plan": member["plan"], "reason": "active member"}
