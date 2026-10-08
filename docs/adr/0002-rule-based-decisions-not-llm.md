# ADR-0002: The decision is rule-based, not made by the LLM

* Status: accepted
* Date: 2026-02-18

## Context

A prior-authorization decision affects a patient's care and a member's money. It must be
explainable, reproducible, and auditable. An LLM is none of those by default: it can give
different answers to the same input and cannot be fully explained.

## Decision

Eligibility, coverage, and clinical criteria are deterministic checks in `tools/`. The
agent applies a fixed policy over their results. The LLM is used only to write the
rationale text, never to decide the outcome.

## Consequences

* **Good:** every decision can be explained line by line and reproduced exactly. The rules
  can be reviewed by clinical and compliance staff.
* **Cost:** the rules must be built and kept up to date. That is the correct place to spend
  effort for a health decision; it is not something to hide inside a model.
