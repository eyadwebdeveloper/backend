"""Tier 1b: cabin anomaly detector (ADVISORY ONLY).

Same idea as Tier 1a but for the station environment: it learns the
station's own normal cabin conditions and flags unusual states. It emits
CONTEXT LABELS (for example "high CO2 exposure"), never medical
conclusions. The labels help avoid blaming the body for a problem caused
by the cabin (AI plan, Tier 1b). The radiation dose rate is used as cabin
context only and is never a personal dose.

The context-label thresholds live in config/tier0_rules.yaml under
`context_labels` (engineering placeholders; the medics own the wording).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .sim import CABIN_SIGNALS
from .tier0 import _runs


def cabin_feature_matrix(tracks: dict) -> tuple:
    n = len(next(iter(tracks.values())).raw)
    X = np.full((n, 2 * len(CABIN_SIGNALS)), np.nan)
    names = []
    ok_counts = np.zeros(n, dtype=int)
    for j, sig in enumerate(CABIN_SIGNALS):
        tr = tracks.get(sig)
        if tr is None:
            names += [f"{sig}_fast", f"{sig}_slow"]
            continue
        X[:, 2 * j] = tr.z
        X[:, 2 * j + 1] = tr.z_slow
        ok_counts += tr.ok.astype(int)
        names += [f"{sig}_fast", f"{sig}_slow"]
    return X, names, ok_counts


class StationAnomalyDetector:
    """IsolationForest over cabin z features, per station."""

    def __init__(self, contamination: float = 0.02, seed: int = 0):
        self.contamination = contamination
        self.seed = seed
        self.high_thresh = None

    def fit(self, X_normal: np.ndarray, high_quantile: float = 0.95) -> "StationAnomalyDetector":
        X_normal = np.asarray(X_normal, dtype=float)
        self.scaler = StandardScaler().fit(X_normal)
        Xs = self.scaler.transform(X_normal)
        self.iforest = IsolationForest(n_estimators=300, contamination=self.contamination,
                                       random_state=self.seed).fit(Xs)
        self._scores = np.sort(self.iforest.score_samples(Xs))
        st = self.strangeness(X_normal)
        self.high_thresh = float(np.quantile(st, high_quantile))
        return self

    def strangeness(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        Xs = self.scaler.transform(X)
        s = self.iforest.score_samples(Xs)
        n = len(self._scores)
        return 1.0 - np.searchsorted(self._scores, s, side="right") / n


def run_station_scoring(detector: StationAnomalyDetector, tracks: dict,
                        cold_start: int = 14, min_ok_signals: int = 4) -> dict:
    X, names, ok_counts = cabin_feature_matrix(tracks)
    n = X.shape[0]
    days = np.arange(n)
    scoreable = (ok_counts >= min_ok_signals) & (days >= cold_start)
    st = detector.strangeness(X)
    st = np.where(scoreable, st, np.nan)
    moved = []
    for d in range(n):
        if not scoreable[d]:
            moved.append("")
            continue
        items = [(abs(X[d, 2 * j]), CABIN_SIGNALS[j]) for j in range(len(CABIN_SIGNALS))
                 if np.isfinite(X[d, 2 * j])]
        items.sort(reverse=True)
        txt = ", ".join(f"{sig.replace('_', ' ')} unusual for this station" for _, sig in items[:2])
        moved.append(f"Cabin environment unusual: {txt}. Context information, not a medical result.")
    return {"strangeness": st, "explanations": moved, "features": X, "scoreable": scoreable}


def context_label_episodes(tracks: dict, context_rules: dict) -> list:
    """Advisory context labels from cabin z thresholds (config-owned)."""
    labels = []
    for name, r in context_rules.items():
        tr = tracks.get(r["signal"])
        if tr is None:
            continue
        cond = np.isfinite(tr.z) & (tr.z >= float(r["z_min"]))
        for s0, e0 in _runs(cond, int(r["min_days"])):
            labels.append({"label": name, "signal": r["signal"],
                           "day": int(s0 + int(r["min_days"]) - 1), "end_day": int(e0),
                           "text": r["text"]})
    return labels
