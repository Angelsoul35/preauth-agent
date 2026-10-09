# Vendored real medical-claim data

This directory holds a small, balanced subset of a **real, public** medical-claim
adjudication dataset, used to validate the learned first-pass triage model
(`preauth/triage.py`) on genuine data. See the benchmark in
`preauth/real_benchmark.py` and the Spanish walkthrough in
[`docs/estudio/04-evaluacion-datos-reales.md`](../docs/estudio/04-evaluacion-datos-reales.md).

## Source

* **Dataset:** [`AniketAsla/adaption-medical-necessity-reviews`](https://huggingface.co/datasets/AniketAsla/adaption-medical-necessity-reviews)
  on the Hugging Face Hub (~35,976 records).
* **Access:** the public **datasets-server rows API**, no authentication:

  ```
  https://datasets-server.huggingface.co/rows?dataset=AniketAsla%2Fadaption-medical-necessity-reviews&config=default&split=train&offset=N&length=100
  ```

* **License:** refer to the dataset card on the Hub for the upstream license and
  terms. Only a small derived subset is vendored here, for reproducible
  benchmarking; the upstream dataset remains the source of record.

## File

`medical_necessity_reviews.balanced.jsonl` — 3,572 records, one JSON object per
line, **balanced** 1,786 fraudulent / 1,786 legitimate. Built from the first
10,000 training rows (offsets 0–9,999); the majority (legitimate) class is
randomly downsampled with `seed=42` to the size of the minority class, then
sorted by `claim_id` for a deterministic file. `claim_description` is truncated
to 600 characters to keep the file under ~3 MB.

Each line keeps only intake-time fields plus the target and one audit column:

| Field | Role |
|---|---|
| `claim_id` | identifier (audit) |
| `claim_description` | **feature** — free-text intake document (TF-IDF) |
| `claim_amount_requested_inr` | **feature** — amount requested |
| `icd_code_matches_procedure` | **feature** — intake coding check |
| `provider_blacklist_flag` | **feature** — intake provider-registry check |
| `fraud_label` | **target** — 0 legitimate, 1 fraudulent |
| `fraud_type` | audit only — **never** a feature (it defines the label) |

## How the target is derived

The decision target is `fraud_label`, mapped onto the project's vocabulary: a
**legitimate** claim (0) is on the *approve* track, a **fraudulent** claim (1) is
on the *deny / flag* track. The learned first pass predicts which track an
incoming claim belongs to.

The alternative "approve iff `claim_amount_approved_inr > 0`" is degenerate on
this data: the approved amount is positive for every record of both classes, so
it carries no signal. `fraud_label` is the substantive binary target.

## Leakage guard

Only information available **at intake** is used as a feature. Every field
decided *during or after* adjudication is excluded, because feeding it would
leak the answer:

* `fraud_type` equals `"legitimate"` exactly when `fraud_label == 0` — it defines
  the label. Fed as text it yields ROC-AUC ≈ 1.000, which is why it is excluded.
* `adjudication_reasoning` states the determination (`APPROVE` / `NOT
  SUBSTANTIATED`); `fraud_confidence` is the adjudicator's own confidence; the
  `enhanced_prompt` / `enhanced_completion` / `reasoning_trace` /
  `original_context` fields are generation artifacts describing the outcome;
  `claim_amount_approved_inr` is itself an adjudication result; `is_phantom` is a
  post-investigation flag.

No kept feature separates the classes on its own — the best single-field rule
reaches only ~0.63 accuracy, versus ~0.71 for the learned model — so the classes
are not trivially separable by a leaking field.

## Regenerate

```bash
python -m preauth.real_data        # fetches from the Hub and rewrites the JSONL
```

The fetcher (`preauth.real_data.fetch_claims`) uses `urllib` with bounded
exponential-backoff retries behind an injectable `urlopen`, so the network path
is testable offline.
