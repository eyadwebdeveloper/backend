"""Tier 3: Game-session AI (ADVISORY ONLY).

Reads one 40-second Astro-Shield run (the raw SessionLog the game records)
and answers four questions for the crew member and the flight surgeon:

  1. What state does this run look like?   nominal / fatigue / tension / low_vigor / cabin_co2
     (multinomial logistic regression, probabilities + confidence)
  2. How unusual is it?                    strangeness in [0, 1]
     (IsolationForest fitted on nominal runs only - same scheme as Tier 1a)
  3. Is it me or the cabin?                posterior probability of the cabin_co2 state
     (slow, fatigue-like runs that coincide with high simulated cabin CO2)
  4. Is it a pattern?                      personal-baseline z-scores and a
     notice gate (N consecutive non-nominal runs), same idea as notices.py

HONEST CIRCULARITY, STATED UP FRONT: like Tier 2a, the training sessions
come from a simulator (`simulate_session`) whose latent states are defined
by the same effects the features measure. This validates the mechanics
(log -> features -> model -> explanation) and NOT medical validity.
Output is decision support, never a diagnosis.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

STATES = ["nominal", "fatigue", "tension", "low_vigor", "cabin_co2"]
STATE_READINESS = {"nominal": 1.0, "tension": 0.62, "low_vigor": 0.55,
                   "fatigue": 0.48, "cabin_co2": 0.5}
STATE_NAMES = {"nominal": "nominal", "fatigue": "fatigue", "tension": "tension",
               "low_vigor": "low-vigor", "cabin_co2": "cabin-CO2"}
NORMAL_CO2 = 3.4          # mmHg, simulated cabin reference
RUN_MS = 40_000

# name -> (plain-language label, "higher is worse" direction text pair)
FEATURES = {
    "median_rt":      ("median reaction time", "slower", "faster"),
    "lapse_rate":     ("lapse rate", "higher", "lower"),
    "rt_cv":          ("reaction-time variability", "higher", "lower"),
    "radar_rt":       ("peripheral-ping reaction time", "slower", "faster"),
    "false_starts":   ("false starts", "more", "fewer"),
    "friendly_fire":  ("friendly-fire blasts", "more", "fewer"),
    "collisions":     ("collisions", "more", "fewer"),
    "adapt_s":        ("time to adapt to inverted controls", "longer", "shorter"),
    "reversal_errors": ("control-reversal errors", "more", "fewer"),
    "engagement":     ("late-run pickup rate vs early", "lower", "higher"),
    "post_error":     ("post-error slowing", "stronger", "weaker"),
    "input_rate":     ("input rate", "higher", "lower"),
    "decision_s":     ("route decision time", "slower", "faster"),
    "co2":            ("cabin CO2 (simulated)", "higher", "lower"),
}
FEATURE_NAMES = list(FEATURES)
# only these features describe the person; CO2 is cabin context
BODY_FEATURES = [f for f in FEATURE_NAMES if f != "co2"]

_IMPUTE = {"median_rt": 650.0, "radar_rt": 900.0, "adapt_s": 6.0, "engagement": 1.0,
           "post_error": 1.0, "decision_s": 2.0}


def _f(x, default):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ feature extract
def extract_features(log: dict, co2: float = NORMAL_CO2) -> dict:
    """Raw SessionLog (as the game records it) -> named feature dict.
    Works on partial logs: anything missing falls back to a neutral value."""
    trials = log.get("trials") or []
    pvt = [t for t in trials if t.get("kind") == "pvt"]
    radar = [t for t in trials if t.get("kind") == "radar"]
    rts = [float(t["rt"]) for t in pvt if t.get("rt") is not None]
    radar_rts = [float(t["rt"]) for t in radar if t.get("rt") is not None]
    lapses = sum(1 for t in pvt if t.get("outcome") in ("lapse", "miss"))
    duration = max(_f(log.get("durationMs"), RUN_MS), 1.0)

    med = float(np.median(rts)) if rts else _IMPUTE["median_rt"]
    cv = float(np.std(rts) / np.mean(rts)) if len(rts) >= 2 else 0.2

    zones = log.get("magnetic") or []
    adapt = [z["adaptMs"] for z in zones if z.get("adaptMs") is not None]
    adapt_s = float(np.mean(adapt)) / 1000.0 if adapt else (
        _IMPUTE["adapt_s"] if zones else 1.5)

    half = duration / 2.0
    first = _f(log.get("pickupsFirstHalf"), 0) / half
    second = _f(log.get("pickupsSecondHalf"), 0) / half
    engagement = float(np.clip(second / first, 0.0, 2.5)) if first > 0 else 1.0

    inputs = sorted(i["at"] for i in (log.get("inputs") or []) if "at" in i)
    gaps = [b - a for a, b in zip(inputs, inputs[1:]) if 0 < b - a < 4000]
    mean_gap = float(np.mean(gaps)) if gaps else 800.0
    ratios = []
    for t in pvt:
        if t.get("outcome") == "miss":
            nxt = next((i for i in inputs if i > t["onsetAt"]), None)
            if nxt is not None and 0 < nxt - t["onsetAt"] < 6000:
                ratios.append((nxt - t["onsetAt"]) / mean_gap)
    route = log.get("route") or {}

    return {
        "median_rt": med,
        "lapse_rate": lapses / len(pvt) if pvt else 0.0,
        "rt_cv": cv,
        "radar_rt": float(np.median(radar_rts)) if radar_rts else _IMPUTE["radar_rt"],
        "false_starts": _f(log.get("falseStarts"), 0),
        "friendly_fire": _f(log.get("friendlyFire"), 0),
        "collisions": _f(log.get("collisions"), 0),
        "adapt_s": adapt_s,
        "reversal_errors": float(sum(_f(z.get("reversalErrors"), 0) for z in zones)),
        "engagement": engagement,
        "post_error": float(np.mean(ratios)) if ratios else 1.0,
        "input_rate": len(inputs) / duration * 60_000.0,
        "decision_s": _f(route.get("decisionMs"), 2000.0) / 1000.0,
        "co2": _f(co2, NORMAL_CO2),
    }


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[n] for n in FEATURE_NAMES], dtype=float)


# ------------------------------------------------------------------ simulator
# Latent-state effects on the *run*. Magnitudes are engineering placeholders.
_STATE_FX = {
    #            rt_mult lapse_p  fs   ff  adapt  eng   post  jitter
    "nominal":   (1.00, 0.04, 0.4, 0.2, 1.8, 1.00, 1.0, 0.12),
    "fatigue":   (1.38, 0.30, 0.2, 0.2, 3.4, 0.85, 1.9, 0.26),
    "tension":   (0.86, 0.06, 2.4, 1.5, 2.6, 1.00, 1.1, 0.16),
    "low_vigor": (1.20, 0.18, 0.2, 0.1, 2.6, 0.38, 1.7, 0.20),
    # healthy person, poisoned air: fatigue-like performance + high CO2
    "cabin_co2": (1.30, 0.22, 0.3, 0.2, 3.0, 0.90, 1.6, 0.22),
}


def simulate_session(state: str, rng: np.random.Generator, co2: float | None = None,
                     person_rt: float | None = None) -> tuple[dict, float]:
    """Make a SessionLog-shaped dict for one virtual run in a latent state."""
    rt_mult, lapse_p, fs, ff, adapt, eng, post, jit = _STATE_FX[state]
    if co2 is None:
        co2 = float(rng.uniform(5.0, 6.2)) if state == "cabin_co2" \
            else float(np.clip(rng.normal(3.6, 0.8), 2.3, 6.2))
    co2 = float(co2)
    # cabin confound: high CO2 slows everyone a little and adds lapses
    co2_excess = max(0.0, co2 - 4.4)
    base = person_rt if person_rt is not None else float(np.clip(rng.normal(300, 35), 220, 400))
    rt_scale = base * rt_mult * (1.0 + 0.10 * co2_excess)
    lapse_p = min(0.9, lapse_p + 0.05 * co2_excess)

    n_pvt = int(rng.integers(6, 11))
    trials, t_at = [], 4500.0
    for _ in range(n_pvt):
        t_at += rng.uniform(1800, 5000)
        if t_at > RUN_MS - 2000:
            break
        if rng.random() < lapse_p:
            if rng.random() < 0.35:
                trials.append({"kind": "pvt", "onsetAt": t_at, "rt": None, "outcome": "miss"})
            else:
                trials.append({"kind": "pvt", "onsetAt": t_at,
                               "rt": float(rng.uniform(510, 900)), "outcome": "lapse"})
        else:
            rt = float(np.clip(rng.lognormal(math.log(rt_scale), jit), 150, 499))
            trials.append({"kind": "pvt", "onsetAt": t_at, "rt": rt, "outcome": "hit"})
    for at in (11_500.0, 28_000.0):
        if rng.random() < lapse_p * 0.9:
            trials.append({"kind": "radar", "onsetAt": at, "rt": None, "outcome": "miss"})
        else:
            trials.append({"kind": "radar", "onsetAt": at, "outcome": "hit",
                           "rt": float(rng.lognormal(math.log(rt_scale * 1.5), jit))})

    zones = []
    for start in (14_000.0, 31_000.0):
        zones.append({"startedAt": start, "endedAt": start + 5000,
                      "adaptMs": float(np.clip(rng.normal(adapt * 1000, 500), 400, 5000))
                      if rng.random() > lapse_p * 0.4 else None,
                      "reversalErrors": int(rng.poisson(1.0 + 1.5 * (rt_mult - 1.0) + (adapt - 1.8) * 0.5)),
                      "collisions": int(rng.poisson(0.3 + 0.4 * (adapt - 1.8)))})
    inputs, at = [], 600.0
    rate = {"nominal": 1.0, "fatigue": 1.45, "tension": 1.5, "low_vigor": 0.6, "cabin_co2": 1.3}[state]
    while at < RUN_MS:
        at += rng.exponential(780 / rate)
        inputs.append({"at": at, "kind": "steer"})
    # post-error slowing: after a missed stimulus the crew member pauses
    for t in trials:
        if t["kind"] == "pvt" and t["outcome"] == "miss":
            delay = (post - 1.0) * 780.0
            inputs = [i for i in inputs if not (t["onsetAt"] < i["at"] < t["onsetAt"] + delay)]
    first = max(1, int(rng.poisson(7)))
    second = int(rng.binomial(first + 2, min(1.0, 0.5 * eng)))
    return ({
        "durationMs": float(RUN_MS), "trials": trials,
        "falseStarts": int(rng.poisson(fs)), "friendlyFire": int(rng.poisson(ff)),
        "collisions": int(rng.poisson(0.8 + 0.7 * (rt_mult - 1.0) * 3)) +
                      sum(z["collisions"] for z in zones),
        "magnetic": zones,
        "route": {"chosen": "storm" if (state == "tension" and rng.random() < .6) else "safe",
                  "decisionMs": float(np.clip(rng.normal(1200 * (1 + (rt_mult - 1.0)), 350), 300, 2000))},
        "inputs": inputs, "pickupsFirstHalf": first, "pickupsSecondHalf": second,
    }, co2)


# ---------------------------------------------------------------------- model
@dataclass
class GameAIModel:
    scaler: StandardScaler
    clf: LogisticRegression
    iforest: IsolationForest
    if_sorted: np.ndarray
    norm_center: np.ndarray       # nominal-run median per feature
    norm_scale: np.ndarray        # nominal-run robust sd per feature
    n_train: int
    train_accuracy: float
    holdout_accuracy: float
    co2_index: int

    @classmethod
    def train(cls, n_per_state: int = 500, seed: int = 2026) -> "GameAIModel":
        rng = np.random.default_rng(seed)
        X, y = [], []
        for state in STATES:
            for _ in range(n_per_state):
                log, co2 = simulate_session(state, rng)
                X.append(feature_vector(extract_features(log, co2)))
                y.append(state)
        X, y = np.asarray(X), np.asarray(y)
        order = rng.permutation(len(X))
        split = int(len(X) * 0.8)
        tr, te = order[:split], order[split:]

        scaler = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=3000, C=0.7, class_weight="balanced",
                                     random_state=seed).fit(scaler.transform(X[tr]), y[tr])
        acc_tr = float(clf.score(scaler.transform(X[tr]), y[tr]))
        acc_te = float(clf.score(scaler.transform(X[te]), y[te]))

        # final fit on everything
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=0.7, class_weight="balanced",
                                 random_state=seed).fit(scaler.transform(X), y)
        normal = scaler.transform(X[y == "nominal"])
        iforest = IsolationForest(n_estimators=300, contamination=0.02,
                                  random_state=seed).fit(normal)
        if_sorted = np.sort(iforest.score_samples(normal))

        nominal_raw = X[y == "nominal"]
        center = np.median(nominal_raw, axis=0)
        mad = np.median(np.abs(nominal_raw - center), axis=0) * 1.4826
        scale = np.where(mad > 1e-9, mad, np.maximum(np.std(nominal_raw, axis=0), 1e-6))
        return cls(scaler, clf, iforest, if_sorted, center, scale, len(X),
                   acc_tr, acc_te, FEATURE_NAMES.index("co2"))

    # -- inference ---------------------------------------------------------
    def _proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(x.reshape(1, -1)))[0]

    def _strangeness(self, x: np.ndarray) -> float:
        s = self.iforest.score_samples(self.scaler.transform(x.reshape(1, -1)))[0]
        return float(1.0 - np.searchsorted(self.if_sorted, s, side="right") / len(self.if_sorted))

    def _drivers(self, x: np.ndarray, state: str, top_k: int = 3) -> list[dict]:
        zs = self.scaler.transform(x.reshape(1, -1))[0]
        idx = list(self.clf.classes_).index(state)
        contrib = self.clf.coef_[idx] * zs
        out = []
        for j in np.argsort(-np.abs(contrib))[:top_k + 1]:
            name = FEATURE_NAMES[j]
            if name == "co2" and state != "nominal":
                continue
            label, up, down = FEATURES[name]
            z = float((x[j] - self.norm_center[j]) / self.norm_scale[j])
            if abs(z) < 0.5:
                continue
            out.append({"feature": name, "label": label,
                        "direction": up if z > 0 else down,
                        "z_vs_typical": round(z, 1),
                        "weight": round(float(contrib[j]), 2)})
        return out[:top_k]

    def assess(self, log: dict, co2: float, history: list[dict],
               history_co2: list[float] | None = None) -> dict:
        co2 = _f(co2, NORMAL_CO2)
        feats = extract_features(log, co2)
        x = feature_vector(feats)
        proba = self._proba(x)
        classes = list(self.clf.classes_)
        top = int(np.argmax(proba))
        state, conf = str(classes[top]), float(proba[top])
        p_nominal = float(proba[classes.index("nominal")])
        # strangeness describes the person's run, so score it at normal cabin CO2
        x_norm = x.copy()
        x_norm[self.co2_index] = NORMAL_CO2
        strange = self._strangeness(x_norm)
        # expected "fitness for duty" under the state posterior, trimmed by oddness
        expected = float(sum(p * STATE_READINESS[c] for c, p in zip(classes, proba)))
        readiness = int(np.clip(round(100 * expected * (1.0 - 0.25 * strange)), 15, 100))

        # ---- Is it me or the cabin? posterior of the cabin_co2 state ------------
        p_cabin = float(proba[classes.index("cabin_co2")])
        if state == "nominal":
            verdict, cabin_text = "none", "Run looks nominal; no cabin attribution needed."
        elif state == "cabin_co2" or p_cabin >= 0.40:
            verdict = "cabin"
            cabin_text = (f"Slow, fatigue-like run while simulated cabin CO2 is {co2:.1f} mmHg "
                          f"(cabin-cause probability {p_cabin * 100:.0f}%). Check scrubber load and "
                          f"ventilation before attributing this to the crew member.")
        elif p_cabin >= 0.15:
            verdict = "mixed"
            cabin_text = (f"The cabin may be part of it (cabin-cause probability "
                          f"{p_cabin * 100:.0f}%, CO2 {co2:.1f} mmHg). Both the cabin and the "
                          f"crew member may contribute.")
        else:
            verdict = "person"
            cabin_text = (f"Cabin CO2 is {co2:.1f} mmHg and the pattern does not look cabin-driven "
                          f"(cabin-cause probability {p_cabin * 100:.0f}%).")

        # ---- personal baseline from the person's own prior runs ----------------
        hist_feats = [extract_features(h, c) for h, c in
                      zip(history, (history_co2 or [NORMAL_CO2] * len(history)))]
        baseline = self._baseline(feats, hist_feats)

        # ---- notice gate: N consecutive non-nominal runs (incl. this one) -------
        hist_states = []
        for hf in hist_feats:
            hp = self._proba(feature_vector(hf))
            hist_states.append(str(classes[int(np.argmax(hp))]))
        streak = 0
        for s in reversed(hist_states + [state]):
            if s == "nominal":
                break
            streak += 1

        notices = self._notices(state, conf, streak, baseline, verdict, cabin_text, strange)
        trend = self._trend(history, hist_feats, feats)

        return {
            "simulated_training": True,
            "model": {
                "name": "aura-game-ai", "version": "1.0",
                "classifier": "logistic_regression", "anomaly": "isolation_forest",
                "n_train_runs": self.n_train,
                "holdout_accuracy": round(self.holdout_accuracy, 3),
                "note": "Trained on simulated runs. Decision support, not a diagnosis.",
            },
            "state": state,
            "confidence": round(conf, 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
            "strangeness": round(strange, 3),
            "readiness": readiness,
            "drivers": self._drivers(x, state) if state != "nominal" else [],
            "cabin": {"co2": round(co2, 1), "verdict": verdict, "text": cabin_text,
                      "cabin_probability": round(p_cabin, 3)},
            "baseline": baseline,
            "pattern": {"consecutive_non_nominal": streak, "history_states": [str(h) for h in hist_states]},
            "notices": notices,
            "trend": trend,
            "features": {k: round(v, 3) for k, v in feats.items()},
        }

    def _baseline(self, feats: dict, hist: list[dict]) -> dict:
        n = len(hist)
        if n < 3:
            return {"prior_runs": n, "ready": False, "z": {}, "worse_than_own_range": False,
                    "note": f"Need 3 prior runs for a personal baseline ({n} so far); "
                            f"comparing against the typical-crew range instead."}
        z, worse = {}, []
        for name in BODY_FEATURES:
            vals = np.array([h[name] for h in hist], dtype=float)
            med = float(np.median(vals))
            sd = 1.4826 * float(np.median(np.abs(vals - med)))
            j = FEATURE_NAMES.index(name)
            # floor the spread so count features with MAD=0 don't explode
            sd = max(sd, 0.5 * float(self.norm_scale[j]), 0.05 * abs(med), 1e-3)
            zz = float(np.clip((feats[name] - med) / sd, -6.0, 6.0))
            z[name] = round(zz, 1)
            # direction of "worse" per feature
            if name in ("engagement",):
                zz = -zz
            if zz >= 1.5 and name in ("median_rt", "lapse_rate", "radar_rt", "post_error", "rt_cv",
                                      "engagement", "adapt_s"):
                worse.append(name)
        return {"prior_runs": n, "ready": True, "z": z,
                "worse_than_own_range": len(worse) >= 2, "worse_features": worse,
                "note": "Compared with your own last runs (robust z)."}

    @staticmethod
    def _trend(history: list[dict], hist_feats: list[dict], feats: dict) -> dict | None:
        """Tiny linear trend of median RT across runs, projected one run ahead."""
        series = [h["median_rt"] for h in hist_feats] + [feats["median_rt"]]
        if len(series) < 4:
            return None
        k = np.arange(len(series))
        slope, intercept = np.polyfit(k, series, 1)
        resid = np.std(np.array(series) - (slope * k + intercept))
        nxt = float(slope * len(series) + intercept)
        return {"signal": "median_rt_ms", "slope_ms_per_run": round(float(slope), 1),
                "next_run_estimate": round(nxt), "band": round(float(1.64 * resid + 15)),
                "direction": "slowing" if slope > 6 else "speeding" if slope < -6 else "steady"}

    @staticmethod
    def _notices(state, conf, streak, baseline, verdict, cabin_text, strange) -> list[dict]:
        out = []
        if state != "nominal" and streak >= 2:
            out.append({"level": "act", "title": "Pattern across consecutive runs",
                        "text": f"{streak} runs in a row look non-nominal (latest: {STATE_NAMES[state]}). "
                                f"AI suggestion, not a diagnosis. Consider workload relief and a quick check-in."})
        elif state not in ("nominal", "cabin_co2") and conf >= 0.55:
            out.append({"level": "watch", "title": f"Run looks like a {STATE_NAMES[state]} pattern",
                        "text": "One run only; a second consecutive one would raise a pattern notice."})
        if baseline.get("worse_than_own_range"):
            out.append({"level": "watch", "title": "Outside your own range",
                        "text": "Several measures were worse than your recent runs "
                                f"({', '.join(FEATURES[f][0] for f in baseline.get('worse_features', [])[:3])})."})
        if verdict in ("cabin", "mixed") and state != "nominal":
            out.append({"level": "act" if verdict == "cabin" else "watch",
                        "title": "Is it me or the cabin?", "text": cabin_text})
        if strange >= 0.95 and state == "nominal":
            out.append({"level": "info", "title": "Unusual but not concerning",
                        "text": "This run is stranger than almost all nominal runs, "
                                "yet no known pattern matched. Probably an off day."})
        return out


_MODEL: GameAIModel | None = None


def get_model() -> GameAIModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = GameAIModel.train()
    return _MODEL
