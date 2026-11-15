"""Tier 1a: body anomaly detector (ADVISORY ONLY).

Learns what a normal day looks like across several body signals at once,
for one crew population, and scores each day with a strangeness value in
[0, 1]. It is meant to catch slow multi-signal drift that a single
threshold can miss (AI plan, Tier 1a). It can post AI notices through the
notice gate; it can never create an official alert.

Features: per signal, the personal z of the EWMA tracker (fast view) and
the personal z of the trailing 14-day mean (slow view). All features are
relative to the person's own baseline - the same mathematics Tier 0 uses
(aura/features.py), so rules and AI interpret identical quantities.

Models: IsolationForest as the primary, a small autoencoder (16-4-16 MLP)
as the comparison, per the AI plan ("use a simple model first... the small
neural network called an autoencoder as the comparison").
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .sim import BODY_SIGNALS

SIGNAL_PHRASES = {
    "resting_hr": "resting heart rate",
    "hrv_rmssd": "heart rate variability",
    "sleep_hours": "sleep duration",
    "active_min": "activity",
    "skin_temp_c": "skin temperature",
    "mood_score": "mood",
    "fatigue_score": "fatigue",
}


def body_feature_matrix(tracks: dict) -> tuple:
    """(X (n_days, 2*7), names, per-day count of usable signals)."""
    n = len(next(iter(tracks.values())).raw)
    X = np.full((n, 2 * len(BODY_SIGNALS)), np.nan)
    names = []
    ok_counts = np.zeros(n, dtype=int)
    for j, sig in enumerate(BODY_SIGNALS):
        tr = tracks.get(sig)
        if tr is None:
            names += [f"{sig}_fast", f"{sig}_slow"]
            continue
        X[:, 2 * j] = tr.z
        X[:, 2 * j + 1] = tr.z_slow
        ok_counts += tr.ok.astype(int)
        names += [f"{sig}_fast", f"{sig}_slow"]
    return X, names, ok_counts


def day_explanation(row: np.ndarray, names: list, top_k: int = 3) -> str:
    """Plain-language 'why': the features that moved the most that day."""
    items = []
    for v, nm in zip(row, names):
        if not np.isfinite(v) or abs(v) < 1.0:
            continue
        sig, view = nm.rsplit("_", 1)
        phrase = SIGNAL_PHRASES.get(sig, sig)
        direction = "above" if v > 0 else "below"
        view_txt = "recent" if view == "fast" else "14-day"
        items.append((abs(v), f"{phrase} {direction} your usual "
                              f"(z {v:+.1f}, {view_txt} view)"))
    items.sort(reverse=True)
    if not items:
        return "Several small changes together; AI suggestion, not a diagnosis."
    top = "; ".join(t for _, t in items[:top_k])
    return (f"Signals that moved the most: {top}. "
            f"AI suggestion, not a diagnosis. Consider a quick check-in.")


class BodyAnomalyDetector:
    """IsolationForest (primary) + small autoencoder (comparison)."""

    def __init__(self, contamination: float = 0.02, seed: int = 0):
        self.contamination = contamination
        self.seed = seed
        self.high_thresh_if = None
        self.high_thresh_ae = None

    def fit(self, X_normal: np.ndarray, high_quantile: float = 0.95) -> "BodyAnomalyDetector":
        X_normal = np.asarray(X_normal, dtype=float)
        self.scaler = StandardScaler().fit(X_normal)
        Xs = self.scaler.transform(X_normal)

        self.iforest = IsolationForest(n_estimators=300, contamination=self.contamination,
                                       random_state=self.seed).fit(Xs)
        if_scores = self.iforest.score_samples(Xs)          # higher = more normal
        self._if_sorted = np.sort(if_scores)

        self.autoencoder = MLPRegressor(hidden_layer_sizes=(16, 4, 16),
                                        activation="tanh", solver="adam",
                                        max_iter=1000, random_state=self.seed)
        self.autoencoder.fit(Xs, Xs)
        ae_err = ((self.autoencoder.predict(Xs) - Xs) ** 2).mean(axis=1)
        self._ae_sorted = np.sort(ae_err)

        # thresholds: the (1 - q) tail of the training-normal distribution
        st_if, st_ae = self.strangeness(X_normal)
        self.high_thresh_if = float(np.quantile(st_if, high_quantile))
        self.high_thresh_ae = float(np.quantile(st_ae, high_quantile))
        return self

    def strangeness(self, X: np.ndarray) -> tuple:
        """Return (strangeness_if, strangeness_ae), each in [0, 1].

        Strangeness = fraction of training-normal days that look MORE normal
        than this day. 0 = utterly normal, 1 = stranger than every normal day.
        """
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        Xs = self.scaler.transform(X)

        s = self.iforest.score_samples(Xs)
        n = len(self._if_sorted)
        st_if = 1.0 - np.searchsorted(self._if_sorted, s, side="right") / n

        err = ((self.autoencoder.predict(Xs) - Xs) ** 2).mean(axis=1)
        st_ae = np.searchsorted(self._ae_sorted, err, side="right") / n
        return st_if, st_ae


def run_body_scoring(detector: BodyAnomalyDetector, tracks: dict,
                     cold_start: int = 14, min_ok_signals: int = 5) -> dict:
    """Score every day of one astronaut. Days with too many failed signals,
    or before cold start, get NaN (no notice can come from them)."""
    X, names, ok_counts = body_feature_matrix(tracks)
    n = X.shape[0]
    days = np.arange(n)
    scoreable = (ok_counts >= min_ok_signals) & (days >= cold_start)

    st_if, st_ae = detector.strangeness(X)
    st_if = np.where(scoreable, st_if, np.nan)
    st_ae = np.where(scoreable, st_ae, np.nan)

    explanations = []
    for d in range(n):
        explanations.append(day_explanation(X[d], names) if scoreable[d] else "")
    return {"strangeness_if": st_if, "strangeness_ae": st_ae,
            "explanations": explanations, "features": X, "feature_names": names,
            "scoreable": scoreable}
