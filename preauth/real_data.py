"""Real medical-claim data: loader, fetcher, and target derivation.

This module grounds the project in a public, real-world dataset:
``AniketAsla/adaption-medical-necessity-reviews`` on the Hugging Face Hub
(~35,976 real medical-claim adjudication records). Each record is a claim that a
payer reviewed for medical necessity, labelled legitimate or fraudulent.

The data is read over the Hugging Face **datasets-server** rows API (no auth):

    https://datasets-server.huggingface.co/rows?dataset=...&offset=N&length=100

A balanced subset is vendored under ``datasets/`` so the benchmark runs offline
and reproducibly; :func:`fetch_claims` / :func:`regenerate` rebuild it from the
Hub.

Target derivation
-----------------
The learned target is ``fraud_label`` (0 = legitimate, 1 = fraudulent), mapped
onto the project's decision vocabulary: a **legitimate** claim is on the
*approve* track, a **fraudulent** claim is on the *deny / flag* track. So the
learned first pass predicts whether an incoming claim should be approved or
flagged for denial. The alternative target "approve iff ``claim_amount_approved_inr > 0``"
is degenerate here: the approved amount is positive for every record of both
classes, so it carries no decision signal. ``fraud_label`` is the substantive
binary target.

Leakage guard
-------------
Only **intake-time** fields are used as features -- the information that arrives
with a claim before any adjudication:

* ``claim_description`` (free text, via TF-IDF),
* ``claim_amount_requested_inr`` (the amount asked for),
* ``icd_code_matches_procedure`` (a deterministic coding check at intake),
* ``provider_blacklist_flag`` (a provider-registry check at intake).

Every field decided *during or after* adjudication is excluded (see
:data:`LEAKING_FIELDS`), because feeding it would leak the answer:
``fraud_type`` equals ``"legitimate"`` exactly when the label is 0 (it defines
the label); ``adjudication_reasoning`` spells out "APPROVE" / "NOT
SUBSTANTIATED"; ``fraud_confidence`` and the ``enhanced_*`` / ``reasoning_trace``
/ ``original_context`` fields are generation artifacts that describe the outcome;
``claim_amount_approved_inr`` is itself an adjudication result. None of the kept
features defines the label on its own (the best single-field rule reaches only
~0.63 accuracy), so the classes are not trivially separable by a leaking field.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

HF_DATASET = "AniketAsla/adaption-medical-necessity-reviews"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

TEXT_FEATURE = "claim_description"
NUMERIC_FEATURES = ("claim_amount_requested_inr",)
FLAG_FEATURES = ("icd_code_matches_procedure", "provider_blacklist_flag")
FEATURE_FIELDS = (TEXT_FEATURE, *NUMERIC_FEATURES, *FLAG_FEATURES)
TARGET_FIELD = "fraud_label"

#: Fields decided during or after adjudication. Using any of them as a feature
#: would leak the label, so they are never fed to the model.
LEAKING_FIELDS = (
    "fraud_type",
    "fraud_confidence",
    "claim_amount_approved_inr",
    "adjudication_reasoning",
    "enhanced_prompt",
    "enhanced_completion",
    "reasoning_trace",
    "original_context",
    "is_phantom",
)

MAX_DESC_CHARS = 600
_DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
DEFAULT_DATASET = _DATASET_DIR / "medical_necessity_reviews.balanced.jsonl"


@dataclass(frozen=True)
class ClaimRecord:
    """One real medical claim, reduced to intake-time features plus the target.

    ``fraud_type`` is kept for auditing and provenance only; it defines the label
    and must never be used as a feature.
    """

    claim_id: str
    claim_description: str
    claim_amount_requested_inr: int
    icd_code_matches_procedure: bool
    provider_blacklist_flag: bool
    label: int  # 1 = fraudulent (deny / flag), 0 = legitimate (approve)
    fraud_type: str  # audit only -- never a feature

    def features(self) -> dict:
        """Return the leakage-free feature view the model consumes."""
        return {
            "claim_description": self.claim_description,
            "claim_amount_requested_inr": self.claim_amount_requested_inr,
            "icd_code_matches_procedure": self.icd_code_matches_procedure,
            "provider_blacklist_flag": self.provider_blacklist_flag,
        }


def derive_label(raw: dict) -> int:
    """Derive the binary decision target from a raw dataset row.

    ``fraud_label`` 0 (legitimate) -> approve track -> 0;
    ``fraud_label`` 1 (fraudulent) -> deny / flag track -> 1.
    """
    return int(raw[TARGET_FIELD])


def _to_record(raw: dict) -> ClaimRecord:
    desc = (raw.get(TEXT_FEATURE) or "")[:MAX_DESC_CHARS]
    return ClaimRecord(
        claim_id=str(raw.get("claim_id", "")),
        claim_description=desc,
        claim_amount_requested_inr=int(raw.get("claim_amount_requested_inr", 0)),
        icd_code_matches_procedure=bool(raw.get("icd_code_matches_procedure", False)),
        provider_blacklist_flag=bool(raw.get("provider_blacklist_flag", False)),
        label=derive_label(raw),
        fraud_type=str(raw.get("fraud_type", "")),
    )


# --------------------------------------------------------------------- fetching
def _rows_url(offset: int, length: int) -> str:
    return (
        f"{ROWS_API}?dataset={quote(HF_DATASET, safe='')}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )


def _get_json(
    url: str,
    *,
    retries: int = 4,
    timeout: float = 60.0,
    opener: Callable | None = None,
) -> dict:
    """GET a JSON document with bounded exponential-backoff retries.

    ``opener`` defaults to the module-level :func:`urllib.request.urlopen` and is
    injectable (or monkeypatchable via ``preauth.real_data.urlopen``) so tests
    can exercise the fetch path without touching the network.
    """
    if opener is None:
        opener = urlopen
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "preauth-agent/real_data"})
            with opener(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network, decode, or bad status -> retry
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to GET {url} after {retries} attempts: {last}")


def fetch_rows(
    offset: int,
    length: int = PAGE_SIZE,
    *,
    retries: int = 4,
    timeout: float = 60.0,
    opener: Callable | None = None,
) -> list[dict]:
    """Fetch one page of raw rows from the datasets-server rows API."""
    data = _get_json(_rows_url(offset, length), retries=retries, timeout=timeout, opener=opener)
    return [item["row"] for item in data.get("rows", [])]


def fetch_claims(
    n_pages: int,
    *,
    page_size: int = PAGE_SIZE,
    retries: int = 4,
    timeout: float = 60.0,
    opener: Callable | None = None,
) -> list[dict]:
    """Fetch ``n_pages`` contiguous pages of raw rows from the Hub."""
    rows: list[dict] = []
    for page in range(n_pages):
        rows.extend(
            fetch_rows(
                page * page_size,
                page_size,
                retries=retries,
                timeout=timeout,
                opener=opener,
            )
        )
    return rows


# ---------------------------------------------------------------- balancing / IO
def build_balanced_subset(raw_rows: Iterable[dict], *, seed: int = 42) -> list[ClaimRecord]:
    """Downsample the majority class to build a 50/50 balanced subset.

    A balanced benchmark makes both classes learnable and the metrics directly
    interpretable (chance accuracy is 0.5). The majority (legitimate) class is
    randomly downsampled with a fixed seed to the size of the minority (fraud)
    class; the result is sorted by ``claim_id`` for a deterministic file.
    """
    import random

    records = [_to_record(r) for r in raw_rows]
    fraud = [r for r in records if r.label == 1]
    legit = [r for r in records if r.label == 0]
    rng = random.Random(seed)
    keep = min(len(fraud), len(legit))
    rng.shuffle(fraud)
    rng.shuffle(legit)
    balanced = fraud[:keep] + legit[:keep]
    balanced.sort(key=lambda r: r.claim_id)
    return balanced


def write_jsonl(records: Sequence[ClaimRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(
                json.dumps(
                    {
                        "claim_id": r.claim_id,
                        "claim_description": r.claim_description,
                        "claim_amount_requested_inr": r.claim_amount_requested_inr,
                        "icd_code_matches_procedure": r.icd_code_matches_procedure,
                        "provider_blacklist_flag": r.provider_blacklist_flag,
                        "fraud_label": r.label,
                        "fraud_type": r.fraud_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_claims(path: str | Path = DEFAULT_DATASET) -> list[ClaimRecord]:
    """Load vendored claims from a JSONL file into :class:`ClaimRecord` objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored dataset not found at {path}; regenerate it with "
            "`python -m preauth.real_data` (needs network access)."
        )
    records: list[ClaimRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(_to_record(json.loads(line)))
    return records


def regenerate(
    path: str | Path = DEFAULT_DATASET,
    *,
    n_pages: int = 100,
    seed: int = 42,
    opener: Callable | None = None,
) -> list[ClaimRecord]:
    """Rebuild the vendored balanced subset from the Hub and write it to ``path``."""
    raw = fetch_claims(n_pages, opener=opener)
    records = build_balanced_subset(raw, seed=seed)
    write_jsonl(records, path)
    return records


def main() -> None:  # pragma: no cover - network side effect
    records = regenerate()
    print(f"wrote {len(records)} balanced records to {DEFAULT_DATASET}")


if __name__ == "__main__":  # pragma: no cover
    main()
