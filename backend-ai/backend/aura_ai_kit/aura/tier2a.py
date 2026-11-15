"""Tier 2a: cardiovascular deconditioning risk classifier (ADVISORY ONLY).

Sorts each astronaut-day into a risk band (low / moderate / elevated) for
cardiovascular deconditioning, from heart-rate and HRV features relative
to the person's own baseline. It is always labeled "not a diagnosis."

HONEST CIRCULARITY, STATED UP FRONT: in this kit the training labels are
defined by the same simulator injection that drives the input features.
This validates the pipeline mechanics (features -> model -> band ->
confidence) only. It does NOT validate medical risk prediction. The open
datasets checked in the project's dataset notes do not link continuous
wearable features to deconditioning outcomes, so Tier 2a stays a design
demonstration on simulated data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import trailing_ols_slope

FEATURES = ["hr_z14", "hrv_z14", "sleep_z14", "active_z14",
            "hr_slope14", "hrv_slope14"]
BANDS = ["none", "early", "established"]


def build_decond_table(tracks_by_astro: dict, episodes: pd.DataFrame,
                       n_days: int, cold_start: int = 14) -> pd.DataFrame:
    """Feature rows + simulator labels for every scoreable astronaut-day."""
    rows = []
    for astro_id, tracks in tracks_by_astro.items():
        eps = episodes[(episodes["astro_id"] == astro_id)
                       & (episodes["problem"] == "deconditioning")]
        z_hr = tracks["resting_hr"].z_slow
        z_hrv = tracks["hrv_rmssd"].z_slow
        z_sleep = tracks["sleep_hours"].z_slow
        z_active = tracks["active_min"].z_slow
        sl_hr = trailing_ols_slope(tracks["resting_hr"].z, 14)
        sl_hrv = trailing_ols_slope(tracks["hrv_rmssd"].z, 14)
        for day in range(cold_start, n_days):
            feats = [z_hr[day], z_hrv[day], z_sleep[day], z_active[day],
                     sl_hr[day], sl_hrv[day]]
            if not all(np.isfinite(f) for f in feats):
                continue
            rows.append({"astro_id": astro_id, "day": day,
                         **dict(zip(FEATURES, [float(f) for f in feats])),
                         "label": _label_day(eps, day)})
    return pd.DataFrame(rows)


def _label_day(eps: pd.DataFrame, day: int) -> str:
    for _, ep in eps.iterrows():
        if ep["onset_day"] <= day <= ep["end_day"]:
            progress = min(1.0, (day - ep["onset_day"]) / max(float(ep["ramp_days"]), 1.0))
            if progress >= 0.5 and day >= ep["onset_day"] + 14:
                return "established"
            return "early"
    return "none"


class DeconditioningRiskClassifier:
    """Multinomial logistic regression: none / early / established."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.clf = LogisticRegression(class_weight="balanced",
                                      max_iter=3000, C=1.0,
                                      random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeconditioningRiskClassifier":
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X: np.ndarray):
        """Returns (bands, confidences). Confidence = max class probability."""
        proba = self.clf.predict_proba(np.asarray(X, dtype=float))
        idx = np.argmax(proba, axis=1)
        bands = np.array(self.clf.classes_)[idx]
        conf = proba[np.arange(len(idx)), idx]
        return bands, conf

    @property
    def n_parameters(self) -> int:
        return int(np.sum([c.size for c in self.clf.coef_]) + self.clf.intercept_.size)
