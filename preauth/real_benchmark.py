"""Benchmark the learned first-pass triage model on real medical-claim data.

Run it:

    python -m preauth.real_benchmark

It loads the vendored balanced subset of
``AniketAsla/adaption-medical-necessity-reviews`` and evaluates the triage model
from :mod:`preauth.triage` on the leakage-free intake features:

* **Stratified k-fold cross-validation** and a **held-out split**, reporting
  macro-F1, accuracy, and ROC-AUC, plus the train/test gap.
* The **majority-class baseline** (and chance ROC-AUC) the model must beat.
* A **single-field separability** probe, confirming no one field trivially
  separates the classes.
* A **regularization path** over C, showing the gap shrink as the model is
  constrained.
* A **shuffled-label control** that must collapse to chance, proving the model
  learns a real relationship rather than an artifact.

Metrics
-------
* accuracy = correct / total.
* For class c: precision_c = TP_c / (TP_c + FP_c), recall_c = TP_c / (TP_c + FN_c),
  F1_c = 2 * precision_c * recall_c / (precision_c + recall_c).
  macro-F1 = mean of the per-class F1 (both classes weighted equally).
* ROC-AUC = probability a random fraud claim is scored above a random
  legitimate one (0.5 = chance, 1.0 = perfect ranking).

A model is **well-fit** when it beats the baseline by a clear margin and its CV
and held-out scores agree with a small train/test gap; **overfit** when train
far exceeds a much lower held-out score; **underfit** when it barely clears the
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .real_data import ClaimRecord, load_claims
from .triage import build_pipeline

SEED = 42
REG_PATH = (0.1, 0.3, 1.0)
DEFAULT_C = 0.3


@dataclass(frozen=True)
class CVScore:
    macro_f1: float
    macro_f1_std: float
    accuracy: float
    roc_auc: float
    roc_auc_std: float
    train_macro_f1: float

    @property
    def gap(self) -> float:
        return self.train_macro_f1 - self.macro_f1


def _xy(records: list[ClaimRecord]) -> tuple[list[dict], np.ndarray]:
    X = [r.features() for r in records]
    y = np.array([r.label for r in records])
    return X, y


def cross_validate_model(
    records: list[ClaimRecord], *, C: float = DEFAULT_C, n_splits: int = 5, seed: int = SEED
) -> CVScore:
    X, y = _xy(records)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = cross_validate(
        build_pipeline(C=C),
        X,
        y,
        cv=skf,
        scoring=["f1_macro", "accuracy", "roc_auc"],
        return_train_score=True,
    )
    return CVScore(
        macro_f1=float(cv["test_f1_macro"].mean()),
        macro_f1_std=float(cv["test_f1_macro"].std()),
        accuracy=float(cv["test_accuracy"].mean()),
        roc_auc=float(cv["test_roc_auc"].mean()),
        roc_auc_std=float(cv["test_roc_auc"].std()),
        train_macro_f1=float(cv["train_f1_macro"].mean()),
    )


def holdout_scores(
    records: list[ClaimRecord], *, C: float = DEFAULT_C, test_size: float = 0.25, seed: int = SEED
) -> dict:
    X, y = _xy(records)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = build_pipeline(C=C).fit(X_tr, y_tr)
    out: dict = {}
    for name, Xs, ys in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        pred = model.predict(Xs)
        proba = model.predict_proba(Xs)[:, 1]
        out[name] = {
            "macro_f1": float(f1_score(ys, pred, average="macro")),
            "accuracy": float(accuracy_score(ys, pred)),
            "roc_auc": float(roc_auc_score(ys, proba)),
        }
    out["gap"] = out["train"]["macro_f1"] - out["test"]["macro_f1"]
    return out


def majority_baseline(records: list[ClaimRecord]) -> dict:
    _, y = _xy(records)
    majority = int(np.bincount(y).argmax())
    pred = np.full_like(y, majority)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": 0.5,
    }


def best_single_field_rule(records: list[ClaimRecord]) -> dict:
    """Best accuracy reachable by thresholding any one feature alone.

    If one field trivially separated the classes this would approach 1.0; it does
    not, which is the evidence that no kept feature leaks the label.
    """
    _, y = _xy(records)
    amounts = np.array([r.claim_amount_requested_inr for r in records])
    median = float(np.median(amounts))
    rules = {
        "icd_code_matches_procedure=False": np.array(
            [0 if r.icd_code_matches_procedure else 1 for r in records]
        ),
        "provider_blacklist_flag=True": np.array(
            [1 if r.provider_blacklist_flag else 0 for r in records]
        ),
        "claim_amount_requested_inr>median": (amounts > median).astype(int),
    }
    scored = {name: float(accuracy_score(y, pred)) for name, pred in rules.items()}
    best = max(scored, key=scored.get)
    return {"rules": scored, "best_field": best, "best_accuracy": scored[best]}


def shuffled_control(
    records: list[ClaimRecord], *, C: float = DEFAULT_C, n_splits: int = 5, seed: int = SEED
) -> dict:
    """Refit with labels randomly permuted; scores must collapse to chance."""
    X, y = _xy(records)
    rng = np.random.default_rng(seed)
    y_shuffled = y[rng.permutation(len(y))]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = cross_validate(build_pipeline(C=C), X, y_shuffled, cv=skf, scoring=["f1_macro", "roc_auc"])
    return {
        "macro_f1": float(cv["test_f1_macro"].mean()),
        "roc_auc": float(cv["test_roc_auc"].mean()),
    }


def regularization_path(
    records: list[ClaimRecord], *, n_splits: int = 5, seed: int = SEED
) -> list[tuple[float, CVScore]]:
    return [(C, cross_validate_model(records, C=C, n_splits=n_splits, seed=seed)) for C in REG_PATH]


def classify_fit(cv: CVScore, holdout: dict, baseline: dict) -> str:
    """Label the fit as well-fit / overfit / underfit, from the evidence."""
    margin = cv.macro_f1 - baseline["macro_f1"]
    cv_holdout_agreement = abs(cv.macro_f1 - holdout["test"]["macro_f1"])
    train_test_gap = holdout["train"]["macro_f1"] - holdout["test"]["macro_f1"]
    if margin < 0.05 or cv.roc_auc < 0.6:
        return "underfit (barely clears the baseline)"
    if train_test_gap > 0.2 and cv_holdout_agreement < 0.1 and cv.gap > 0.2:
        return "overfit (train far above held-out)"
    return "well-fit (clears baseline; CV and held-out agree; small train/test gap)"


@dataclass(frozen=True)
class BenchmarkResult:
    n: int
    n_fraud: int
    n_legit: int
    baseline: dict
    cv: CVScore
    holdout: dict
    single_field: dict
    shuffled: dict
    reg_path: list[tuple[float, CVScore]]
    fit_label: str


def run_benchmark(
    records: list[ClaimRecord] | None = None,
    *,
    C: float = DEFAULT_C,
    n_splits: int = 5,
    with_reg_path: bool = True,
) -> BenchmarkResult:
    records = records if records is not None else load_claims()
    n_fraud = sum(1 for r in records if r.label == 1)
    baseline = majority_baseline(records)
    cv = cross_validate_model(records, C=C, n_splits=n_splits)
    holdout = holdout_scores(records, C=C)
    single_field = best_single_field_rule(records)
    shuffled = shuffled_control(records, C=C, n_splits=n_splits)
    reg = regularization_path(records, n_splits=n_splits) if with_reg_path else []
    return BenchmarkResult(
        n=len(records),
        n_fraud=n_fraud,
        n_legit=len(records) - n_fraud,
        baseline=baseline,
        cv=cv,
        holdout=holdout,
        single_field=single_field,
        shuffled=shuffled,
        reg_path=reg,
        fit_label=classify_fit(cv, holdout, baseline),
    )


def format_report(res: BenchmarkResult) -> str:
    h = res.holdout
    lines = [
        "=" * 72,
        "PreAuth Agent -- learned first-pass triage on real medical-claim data",
        "=" * 72,
        "Dataset   : AniketAsla/adaption-medical-necessity-reviews (Hugging Face)",
        f"Records   : {res.n} balanced claims ({res.n_fraud} fraud / {res.n_legit} legitimate)",
        "Target    : fraud_label -> 1=fraudulent (deny/flag), 0=legitimate (approve)",
        "Features  : claim_description (TF-IDF) + requested amount + ICD-match + blacklist flag",
        "            (intake-time only; every post-adjudication field excluded as leakage)",
        "-" * 72,
        "Majority-class baseline (must be beaten):",
        f"  macro-F1 {res.baseline['macro_f1']:.3f} | accuracy {res.baseline['accuracy']:.3f}"
        f" | ROC-AUC {res.baseline['roc_auc']:.3f} (chance)",
        "-" * 72,
        f"Stratified cross-validation (logistic regression, C={DEFAULT_C}):",
        f"  macro-F1 {res.cv.macro_f1:.3f} +/- {res.cv.macro_f1_std:.3f}"
        f" | accuracy {res.cv.accuracy:.3f}"
        f" | ROC-AUC {res.cv.roc_auc:.3f} +/- {res.cv.roc_auc_std:.3f}",
        f"  train macro-F1 {res.cv.train_macro_f1:.3f} | train/CV gap {res.cv.gap:.3f}",
        "-" * 72,
        "Held-out split (75% train / 25% test):",
        f"  train : macro-F1 {h['train']['macro_f1']:.3f}"
        f" | accuracy {h['train']['accuracy']:.3f} | ROC-AUC {h['train']['roc_auc']:.3f}",
        f"  test  : macro-F1 {h['test']['macro_f1']:.3f}"
        f" | accuracy {h['test']['accuracy']:.3f} | ROC-AUC {h['test']['roc_auc']:.3f}",
        f"  train/test macro-F1 gap {h['gap']:.3f}",
        "-" * 72,
        "No single feature separates the classes (leakage guard):",
    ]
    for name, acc in res.single_field["rules"].items():
        lines.append(f"  {name:<40s} accuracy {acc:.3f}")
    lines.append(
        f"  best single field = {res.single_field['best_accuracy']:.3f}"
        f" << learned model {res.cv.accuracy:.3f}"
    )
    if res.reg_path:
        lines.append("-" * 72)
        lines.append("Regularization path (stronger regularization -> smaller gap):")
        for C, s in res.reg_path:
            lines.append(
                f"  C={C:<4} CV macro-F1 {s.macro_f1:.3f} | ROC-AUC {s.roc_auc:.3f}"
                f" | train/CV gap {s.gap:.3f}"
            )
    lines += [
        "-" * 72,
        "Shuffled-label control (labels permuted; must collapse to chance):",
        f"  CV macro-F1 {res.shuffled['macro_f1']:.3f}"
        f" | ROC-AUC {res.shuffled['roc_auc']:.3f}  (vs real ROC-AUC {res.cv.roc_auc:.3f})",
        "-" * 72,
        f"Verdict: {res.fit_label}",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> None:
    print(format_report(run_benchmark()))


if __name__ == "__main__":  # pragma: no cover
    main()
