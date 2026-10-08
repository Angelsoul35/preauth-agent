# ADR-0003: A human always makes the final decision

* Status: accepted
* Date: 2026-02-19

## Context

Even with correct rules, an automated system should not finalize a medical decision on its
own. Edge cases, missing context, and the weight of the outcome all call for a person.

## Decision

The agent only recommends (`approve`, `deny`, `pend`) and sets the status to
`pending_human_approval` (or `needs_info`). The final `approved` / `denied` status is set
only through the approval step, by a named reviewer. The service refuses to approve a
request that is not awaiting human approval.

## Consequences

* **Good:** the system speeds up the work without taking the decision away from a
  clinician. There is a clear, auditable approval event with a reviewer's name.
* **Cost:** a human step is still required, so this is assistance, not full automation. For
  this domain that is the right trade-off.
