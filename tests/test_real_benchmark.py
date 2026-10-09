"""Functional tests for the real-data loader, triage model, and benchmark.

These assert real properties, not surface shape: the vendored data is the real
balanced subset with the derived target; the learned model beats the right
baseline on held-out data and generalizes (CV agrees with held-out); a
shuffled-label control collapses to chance; and the network fetch path works
behind an injectable ``urlopen``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from preauth import real_benchmark, real_data
from preauth.real_benchmark import (
    CVScore,
    best_single_field_rule,
    classify_fit,
    cross_validate_model,
    format_report,
    holdout_scores,
    majority_baseline,
    run_benchmark,
    shuffled_control,
)
from preauth.real_data import (
    FEATURE_FIELDS,
    LEAKING_FIELDS,
    ClaimRecord,
    build_balanced_subset,
    derive_label,
    fetch_claims,
    load_claims,
    regenerate,
    write_jsonl,
)
from preauth.triage import ClaimTriageModel, FeatureBuilder


@pytest.fixture(scope="module")
def claims() -> list[ClaimRecord]:
    return load_claims()


@pytest.fixture(scope="module")
def balanced_sample(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    """A smaller balanced slice for the faster cross-validation tests."""
    fraud = [r for r in claims if r.label == 1][:500]
    legit = [r for r in claims if r.label == 0][:500]
    return fraud + legit


# ------------------------------------------------------------------- the loader
def test_loader_returns_real_balanced_data(claims: list[ClaimRecord]):
    assert len(claims) == 3572
    n_fraud = sum(1 for r in claims if r.label == 1)
    n_legit = sum(1 for r in claims if r.label == 0)
    assert n_fraud == n_legit == 1786  # balanced by construction
    assert {r.label for r in claims} == {0, 1}
    # Real intake text is present and non-trivial.
    assert all(len(r.claim_description) > 30 for r in claims)
    assert all(r.claim_id for r in claims)
    # The requested amount is a real positive integer.
    assert all(isinstance(r.claim_amount_requested_inr, int) for r in claims)
    assert all(r.claim_amount_requested_inr > 0 for r in claims)


def test_target_derivation_matches_fraud_type(claims: list[ClaimRecord]):
    # Derivation invariant: fraud_type == "legitimate" exactly for the approve
    # class (label 0); any other fraud_type belongs to the deny/flag class.
    for r in claims:
        assert (r.fraud_type == "legitimate") == (r.label == 0)
    assert derive_label({"fraud_label": 1}) == 1
    assert derive_label({"fraud_label": 0}) == 0


def test_features_are_leakage_free(claims: list[ClaimRecord]):
    feats = claims[0].features()
    # Exactly the four intake-time features, nothing decided at adjudication.
    assert set(feats) == set(FEATURE_FIELDS)
    for leak in LEAKING_FIELDS:
        assert leak not in feats
    # fraud_type defines the label, so it must never be part of the feature view.
    assert "fraud_type" not in feats
    assert "fraud_label" not in feats


# ----------------------------------------------------------------- the baseline
def test_majority_baseline_is_chance(claims: list[ClaimRecord]):
    base = majority_baseline(claims)
    assert base["accuracy"] == pytest.approx(0.5, abs=0.01)  # balanced data
    assert base["roc_auc"] == 0.5
    assert base["macro_f1"] < 0.4


def test_no_single_field_trivially_separates(claims: list[ClaimRecord]):
    # If any kept feature leaked the label this would approach 1.0; it does not.
    sep = best_single_field_rule(claims)
    assert sep["best_accuracy"] < 0.75
    assert sep["best_accuracy"] > 0.5  # still carries some genuine signal


# ------------------------------------------------------- the model generalizes
def test_model_beats_baseline_on_holdout(claims: list[ClaimRecord]):
    base = majority_baseline(claims)
    h = holdout_scores(claims)
    # Held-out test clears the majority baseline by a clear margin...
    assert h["test"]["macro_f1"] > base["macro_f1"] + 0.2
    assert h["test"]["accuracy"] > base["accuracy"] + 0.15
    # ...and ranks well above chance.
    assert h["test"]["roc_auc"] > 0.7
    # Not overfit: train does not tower over held-out.
    assert h["gap"] < 0.15


def test_cross_validation_agrees_with_holdout(balanced_sample: list[ClaimRecord]):
    base = majority_baseline(balanced_sample)
    cv = cross_validate_model(balanced_sample, n_splits=3)
    h = holdout_scores(balanced_sample)
    assert cv.macro_f1 > base["macro_f1"] + 0.2
    assert cv.roc_auc > 0.7
    # CV and held-out tell the same story -> the model generalizes.
    assert abs(cv.macro_f1 - h["test"]["macro_f1"]) < 0.1
    verdict = classify_fit(cv, h, base)
    assert verdict.startswith("well-fit")


def test_shuffled_label_control_collapses(balanced_sample: list[ClaimRecord]):
    real = cross_validate_model(balanced_sample, n_splits=3)
    control = shuffled_control(balanced_sample, n_splits=3)
    # Permuting the labels destroys the signal: scores fall back to chance.
    assert control["roc_auc"] < 0.6
    assert control["macro_f1"] < 0.6
    # The real model is far above its own shuffled control.
    assert real.roc_auc > control["roc_auc"] + 0.15


def test_triage_model_recommends_in_project_vocabulary(balanced_sample: list[ClaimRecord]):
    model = ClaimTriageModel().fit(balanced_sample)
    recs = {model.recommend(r) for r in balanced_sample}
    assert recs <= {"approve", "pend", "deny"}
    # A trained model does not collapse every claim into a single bucket.
    assert len(recs) >= 2
    # Fraud claims score higher on average than legitimate ones.
    proba = model.fraud_proba(balanced_sample)
    labels = np.array([r.label for r in balanced_sample])
    assert proba[labels == 1].mean() > proba[labels == 0].mean()


# ------------------------------------------------------------- the whole report
def test_run_benchmark_report_contains_the_evidence(balanced_sample: list[ClaimRecord]):
    res = run_benchmark(balanced_sample, n_splits=3, with_reg_path=True)
    assert res.n == len(balanced_sample)
    assert res.fit_label.startswith("well-fit")
    assert len(res.reg_path) == 3
    report = format_report(res)
    for marker in [
        "macro-F1",
        "ROC-AUC",
        "Majority-class baseline",
        "Shuffled-label control",
        "leakage",
        "Regularization path",
        "Verdict:",
    ]:
        assert marker in report


# ---------------------------------------------------------------- the fetch path
class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _raw_row(claim_id: str, label: int, *, icd_match: bool, blacklist: bool) -> dict:
    return {
        "row": {
            "claim_id": claim_id,
            "claim_description": f"Discharge abstract for claim {claim_id} with clinical notes.",
            "claim_amount_requested_inr": 50000 + label,
            "icd_code_matches_procedure": icd_match,
            "provider_blacklist_flag": blacklist,
            "fraud_label": label,
            "fraud_type": "legitimate" if label == 0 else "bill_inflation",
        }
    }


def test_fetch_path_with_injected_urlopen(monkeypatch, tmp_path):
    pages = {
        0: {"rows": [_raw_row(f"A{i}", 1, icd_match=False, blacklist=True) for i in range(3)]},
        100: {"rows": [_raw_row(f"B{i}", 0, icd_match=True, blacklist=False) for i in range(5)]},
    }

    def fake_urlopen(req, timeout=60.0):
        offset = int(req.full_url.split("offset=")[1].split("&")[0])
        return _FakeResponse(pages[offset])

    monkeypatch.setattr(real_data, "urlopen", fake_urlopen)

    raw = fetch_claims(2)
    assert len(raw) == 8  # 3 + 5 rows across two pages
    records = build_balanced_subset(raw)
    # Balanced down to the minority (3 fraud) -> 3 + 3.
    assert len(records) == 6
    assert sum(r.label for r in records) == 3

    out = tmp_path / "sub.jsonl"
    write_jsonl(records, out)
    reloaded = load_claims(out)
    assert len(reloaded) == 6
    assert {r.claim_id for r in reloaded} == {r.claim_id for r in records}


def test_fetch_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(real_data.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky_urlopen(req, timeout=60.0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient network error")
        return _FakeResponse({"rows": [_raw_row("X0", 1, icd_match=True, blacklist=False)]})

    monkeypatch.setattr(real_data, "urlopen", flaky_urlopen)
    rows = real_data.fetch_rows(0, retries=4)
    assert calls["n"] == 3  # failed twice, succeeded on the third
    assert rows[0]["claim_id"] == "X0"


def test_fetch_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setattr(real_data.time, "sleep", lambda *_: None)

    def always_fail(req, timeout=60.0):
        raise OSError("down")

    monkeypatch.setattr(real_data, "urlopen", always_fail)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        real_data.fetch_rows(0, retries=4)


def test_load_claims_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_claims(tmp_path / "does_not_exist.jsonl")


def test_regenerate_fetches_balances_and_writes(monkeypatch, tmp_path):
    pages = {
        0: {"rows": [_raw_row(f"A{i}", 1, icd_match=False, blacklist=True) for i in range(4)]},
        100: {"rows": [_raw_row(f"B{i}", 0, icd_match=True, blacklist=False) for i in range(9)]},
    }

    def fake_urlopen(req, timeout=60.0):
        offset = int(req.full_url.split("offset=")[1].split("&")[0])
        return _FakeResponse(pages[offset])

    monkeypatch.setattr(real_data, "urlopen", fake_urlopen)
    out = tmp_path / "regen.jsonl"
    records = regenerate(out, n_pages=2)
    assert out.exists()
    assert sum(r.label for r in records) == len(records) // 2  # balanced 4+4
    assert len(records) == 8


# ----------------------------------------------------------- triage internals
def test_feature_builder_text_only_drops_flag_columns(claims: list[ClaimRecord]):
    feats = [r.features() for r in claims[:200]]
    with_flags = FeatureBuilder(use_flags=True).fit(feats).transform(feats)
    text_only = FeatureBuilder(use_flags=False).fit(feats).transform(feats)
    # The flag build adds exactly three dense columns (amount + two flags).
    assert with_flags.shape[1] == text_only.shape[1] + 3


def test_triage_predict_and_recommend_bands(balanced_sample: list[ClaimRecord]):
    model = ClaimTriageModel().fit(balanced_sample)
    preds = model.predict(balanced_sample)
    assert set(preds.tolist()) <= {0, 1}
    # Drive each probability band deterministically through recommend().
    sample = balanced_sample[0]
    model.fraud_proba = lambda records: np.array([0.05])
    assert model.recommend(sample) == "approve"
    model.fraud_proba = lambda records: np.array([0.5])
    assert model.recommend(sample) == "pend"
    model.fraud_proba = lambda records: np.array([0.95])
    assert model.recommend(sample) == "deny"


# ------------------------------------------------------------ fit classifier
def _cv(macro_f1, train, auc):
    return CVScore(
        macro_f1=macro_f1,
        macro_f1_std=0.01,
        accuracy=macro_f1,
        roc_auc=auc,
        roc_auc_std=0.01,
        train_macro_f1=train,
    )


def test_classify_fit_underfit():
    base = {"macro_f1": 0.33, "accuracy": 0.5, "roc_auc": 0.5}
    cv = _cv(0.35, 0.36, 0.55)  # margin < 0.05
    holdout = {"train": {"macro_f1": 0.36}, "test": {"macro_f1": 0.35}}
    assert classify_fit(cv, holdout, base).startswith("underfit")


def test_classify_fit_overfit():
    base = {"macro_f1": 0.33, "accuracy": 0.5, "roc_auc": 0.5}
    cv = _cv(0.60, 0.85, 0.80)  # big train/CV gap, strong AUC
    holdout = {"train": {"macro_f1": 0.90}, "test": {"macro_f1": 0.60}}  # gap 0.30
    assert classify_fit(cv, holdout, base).startswith("overfit")


def test_main_prints_report(monkeypatch, capsys, balanced_sample: list[ClaimRecord]):
    res = run_benchmark(balanced_sample, n_splits=3, with_reg_path=False)
    monkeypatch.setattr(real_benchmark, "run_benchmark", lambda: res)
    real_benchmark.main()
    out = capsys.readouterr().out
    assert "Verdict:" in out
    assert "ROC-AUC" in out
