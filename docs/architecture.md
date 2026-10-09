# Architecture

## Goal

Run the first pass of a prior-authorization request safely: gather facts with tools, apply
a rule-based policy, and hand the final decision to a human.

## Components

| Layer | Module | What it does |
|---|---|---|
| Schema | `schema.py` | Request and decision-packet models |
| Data | `data.py` | Synthetic members, coverage, criteria, documents |
| Tools | `tools/` | The four checks, each a focused function |
| Agent | `agent.py` | Gathers facts, applies the policy, writes the rationale |
| Service | `service.py` | Stores packets, applies the human decision |
| MCP | `mcp_server.py` | Exposes the tools over the Model Context Protocol |
| LLM | `llm/` | Provider-agnostic cascade (rationale text only) |
| Triage | `triage.py` | Learned first-pass fraud-risk model, alongside the rules |
| Real data | `real_data.py` | Loader, fetcher, and leakage-free target derivation |
| Benchmark | `real_benchmark.py` | Validates the triage model on real medical-claim data |
| API | `main.py` | FastAPI endpoints |

## The agent loop

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Service
    participant A as Agent
    participant T as Tools
    participant H as Human

    C->>S: POST /preauth {request}
    S->>A: run(request)
    A->>T: check_eligibility
    A->>T: check_coverage
    A->>T: check_documents
    A->>T: check_clinical_criteria
    A->>A: policy -> recommend (approve/deny/pend)
    A->>A: write rationale (template or LLM)
    A-->>S: decision packet (status: pending_human_approval or needs_info)
    S-->>C: packet
    C->>S: POST /preauth/{id}/approve {decision, reviewer}
    S->>H: applies the human decision
    S-->>C: final packet (approved / denied)
```

## Two safety rules (the point of the project)

**Rule-based decision.** Eligibility, coverage, and clinical criteria are deterministic
checks in `tools/`. They always give the same answer for the same input and can be
explained line by line. The LLM is never in the decision path — it only turns the findings
into readable text. In a health or benefit decision, "the model said so" is not acceptable.

**Human in the loop.** The agent sets the status to `pending_human_approval` (or
`needs_info` when documents are missing). The final `approved` / `denied` status can only be
set through the approval step, by a named reviewer. The service refuses to approve a request
that is not awaiting human approval.

## Tools and MCP

Each tool is a plain function with a clear input and a dict result. They live in `tools/`
and are collected in a registry. The agent calls them directly, and `mcp_server.py` exposes
the same functions over MCP, so one definition serves both the service and any external MCP
client.

## Learned first-pass triage, validated on real data

Alongside the rule engine, `triage.py` adds a learned first pass: a scikit-learn pipeline
(TF-IDF over the claim description plus the standardized requested amount and two intake
flags, into an L2-regularized logistic regression) that scores an incoming claim for fraud
risk. It never overrides the deterministic checks — it is a complementary, inspectable
signal that routes high-risk claims to a human sooner.

It is validated on a real, public dataset (`AniketAsla/adaption-medical-necessity-reviews`,
~35,976 records) using only intake-time, leakage-free features. `real_data.py` loads the
vendored balanced subset and derives the target (`fraud_label`: legitimate → approve,
fraudulent → deny/flag), excluding every post-adjudication field. `real_benchmark.py`
reports stratified cross-validation and a held-out split (macro-F1, accuracy, ROC-AUC), the
train/test gap, the majority baseline, a single-field separability probe, a regularization
path, and a shuffled-label control that collapses to chance. See the Spanish deep-dive in
[`estudio/04-evaluacion-datos-reales.md`](estudio/04-evaluacion-datos-reales.md).

## Provider-agnostic LLM

The rationale writer uses the `llm/` cascade: one `LLMClient` interface, adapters for
several providers, and a `CascadeLLM` that falls back on failure. It is only used for the
wording, and it falls back to a template if no provider is available, so the agent always
produces a rationale.

## Configuration

Settings use the `PA_` prefix (see `.env.example`): the rationale mode (`template` or
`llm`) and the provider order.
