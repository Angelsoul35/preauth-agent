"""PreAuth Agent command-line demo."""

from __future__ import annotations

import argparse

from .data import SAMPLE_REQUESTS
from .schema import PriorAuthRequest
from .service import PreAuthService


def demo() -> None:
    service = PreAuthService()
    print("Running the sample pre-authorization requests:\n")
    first_approval_id = None
    for i, raw in enumerate(SAMPLE_REQUESTS, start=1):
        packet = service.submit(PriorAuthRequest(**raw))
        print(f"--- Request {i}: {packet.procedure_code} for {packet.member_id} ---")
        print(f"recommendation : {packet.recommendation.upper()}")
        print(f"status         : {packet.status}")
        if packet.missing_documents:
            print(f"missing docs   : {', '.join(packet.missing_documents)}")
        print(packet.rationale)
        print()
        if first_approval_id is None and packet.status == "pending_human_approval":
            first_approval_id = packet.request_id

    if first_approval_id:
        print("A human reviewer confirms the first case ...")
        final = service.approve(first_approval_id, "approved", reviewer="dr.review", note="agrees")
        print(f"final status   : {final.status} by {final.human_decision.reviewer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PreAuth Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the sample requests end to end")
    args = parser.parse_args()
    if args.command == "demo":
        demo()


if __name__ == "__main__":
    main()
