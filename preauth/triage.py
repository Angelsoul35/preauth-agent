"""A learned first-pass triage model, alongside the rule engine.

The project's authorization decision stays rule-based and human-confirmed (see
:mod:`preauth.agent`). This module adds a complementary *learned* first pass: a
classifier that scores an incoming claim for fraud risk from its intake-time
information, so high-risk claims can be routed to a human reviewer sooner. It
never overrides the deterministic checks -- it is an additional, validated
signal.

The model is a scikit-learn pipeline:

1. :class:`FeatureBuilder` turns a claim into a sparse feature matrix --
   TF-IDF over ``claim_description`` combined with the standardized requested
   amount and two intake flags (ICD/procedure match, provider blacklist).
2. :class:`~sklearn.linear_model.LogisticRegression` (L2-regularized, class
   weighting for balance) makes an interpretable, reproducible prediction.

Logistic regression is deliberate: like the rule engine, its decision is
inspectable (per-feature coefficients), which fits a domain where a reviewer
must be able to see *why* a claim was flagged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .real_data import ClaimRecord
from .schema import Recommendation

# Probability bands that map a fraud score onto the project's decision vocabulary.
APPROVE_BELOW = 0.33
DENY_ABOVE = 0.66


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn claim feature-dicts into a leakage-free sparse matrix.

    Input ``X`` is a sequence of dicts with the intake-time fields (as produced
    by :meth:`preauth.real_data.ClaimRecord.features`). Text goes through TF-IDF;
    the requested amount is log-scaled and standardized; the two flags pass
    through as 0/1. Nothing decided during adjudication is read.
    """

    def __init__(self, min_df: int = 3, ngram_max: int = 2, use_flags: bool = True) -> None:
        self.min_df = min_df
        self.ngram_max = ngram_max
        self.use_flags = use_flags

    def _texts(self, X: Sequence[dict]) -> list[str]:
        return [str(d.get("claim_description") or "") for d in X]

    def _amounts(self, X: Sequence[dict]) -> np.ndarray:
        return np.log1p([max(int(d.get("claim_amount_requested_inr", 0)), 0) for d in X])

    def fit(self, X: Sequence[dict], y=None) -> FeatureBuilder:
        self.vectorizer_ = TfidfVectorizer(
            min_df=self.min_df, ngram_range=(1, self.ngram_max), sublinear_tf=True
        )
        self.vectorizer_.fit(self._texts(X))
        amt = self._amounts(X)
        self.amount_mean_ = float(amt.mean())
        self.amount_std_ = float(amt.std()) or 1.0
        return self

    def transform(self, X: Sequence[dict]):
        text = self.vectorizer_.transform(self._texts(X))
        if not self.use_flags:
            return text
        amt = (self._amounts(X) - self.amount_mean_) / self.amount_std_
        icd = np.array([1.0 if d.get("icd_code_matches_procedure") else 0.0 for d in X])
        blacklist = np.array([1.0 if d.get("provider_blacklist_flag") else 0.0 for d in X])
        dense = csr_matrix(np.column_stack([amt, icd, blacklist]))
        return hstack([text, dense]).tocsr()


def build_pipeline(
    C: float = 0.3, min_df: int = 3, ngram_max: int = 2, use_flags: bool = True
) -> Pipeline:
    """Build the triage pipeline (feature builder + logistic regression)."""
    return Pipeline(
        [
            ("features", FeatureBuilder(min_df=min_df, ngram_max=ngram_max, use_flags=use_flags)),
            (
                "classifier",
                LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
            ),
        ]
    )


class ClaimTriageModel:
    """App-facing wrapper over the triage pipeline.

    Trains on :class:`~preauth.real_data.ClaimRecord` objects and scores new
    claims, exposing the fraud probability and a recommendation in the project's
    ``approve`` / ``pend`` / ``deny`` vocabulary.
    """

    def __init__(self, C: float = 0.3, min_df: int = 3, ngram_max: int = 2) -> None:
        self.pipeline = build_pipeline(C=C, min_df=min_df, ngram_max=ngram_max)

    def fit(self, records: Sequence[ClaimRecord]) -> ClaimTriageModel:
        X = [r.features() for r in records]
        y = np.array([r.label for r in records])
        self.pipeline.fit(X, y)
        return self

    def fraud_proba(self, records: Sequence[ClaimRecord]) -> np.ndarray:
        X = [r.features() for r in records]
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, records: Sequence[ClaimRecord]) -> np.ndarray:
        X = [r.features() for r in records]
        return self.pipeline.predict(X)

    def recommend(self, record: ClaimRecord) -> Recommendation:
        """Map the fraud probability onto a first-pass recommendation.

        Low risk -> ``approve`` track; high risk -> ``deny`` / flag; the uncertain
        middle band -> ``pend`` for human review. This is a triage aid; the final
        authorization decision remains rule-based and human-confirmed.
        """
        p = float(self.fraud_proba([record])[0])
        if p < APPROVE_BELOW:
            return "approve"
        if p > DENY_ABOVE:
            return "deny"
        return "pend"
