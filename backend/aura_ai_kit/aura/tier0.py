"""Tier 0: the rules engine. THE ONLY ALERT AUTHORITY.

Implements the AI plan's one rule: "The models advise. The rules decide."
Only this module can create official alert levels (informational, watch,
elevated, urgent_review). All rule values come from config/tier0_rules.yaml,
which the medical teammates own; the code only reads it.

Per astronaut and per station:
  1. data-quality gate (range, stuck, gaps) -> sensor messages; bad values
     never reach the health rules (never a fake health alert)
  2. personal baselines (EWMA + robust z vs a mission-start reference)
  3. rules: absolute levels, personal z with duration, rate-of-change,
     slow-drift-vs-reference, check-in thresholds
  4. a 30-day straight-line trend note (informational "days until your
     personal watch level"), which is a note, never an alert

Alerts are recorded as EPISODES: one row when a rule activates, with the
end day when the condition clears. This keeps counting honest (no daily
re-alerting of the same condition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SignalTrack, make_tracks, trailing_ols_slope

LEVELS = ["informational", "watch", "elevated", "urgent_review"]

FRIENDLY = {
    "resting_hr": "Resting heart rate", "hrv_rmssd": "Heart rate variability (RMSSD)",
    "sleep_hours": "Sleep duration", "active_min": "Activity",
    "skin_temp_c": "Skin temperature", "mood_score": "Mood (check-in)",
    "fatigue_score": "Fatigue (check-in)", "co2_ppm": "Cabin CO2",
    "cabin_temp_c": "Cabin temperature", "rh_pct": "Cabin humidity",
    "noise_db": "Cabin noise", "dose_usv_day": "Cabin radiation dose rate",
}


def _beyond(direction: str, x, thr: float):
    """Directional threshold test on an array of z values."""
    if direction == "upper":
        return x > thr
    if direction == "lower":
        return x < -thr
    return np.abs(x) > thr


def _runs(cond: np.ndarray, min_len: int):
    """Runs of True of length >= min_len. Returns list of (start, end_inclusive)."""
    runs = []
    n = len(cond)
    t = 0
    while t < n:
        if cond[t]:
            s0 = t
            while t < n and cond[t]:
                t += 1
            if t - s0 >= min_len:
                runs.append((s0, t - 1))
        else:
            t += 1
    return runs


class Tier0Engine:
    def __init__(self, rules: dict):
        self.rules = rules
        self.settings = rules["settings"]

    # ---------------- body (per astronaut) ----------------

    def run_body(self, astro_id: str, station_id: str, df_a: pd.DataFrame) -> dict:
        alerts, notes, msgs = [], [], []
        signals = {s: df_a[s].to_numpy(dtype=float) for s in df_a.columns
                   if s in self.rules["body_signals"]}
        tracks = make_tracks(signals, self.rules["body_signals"], self.settings)
        cold = int(self.settings.get("cold_start_days", 14))
        n = len(df_a)
        days = np.arange(n)

        for sig, track in tracks.items():
            cfg = self.rules["body_signals"][sig]
            friendly = FRIENDLY.get(sig, sig)
            for ev in track.sensor_events:
                msgs.append({"scope": "body", "astro_id": astro_id,
                             "station_id": station_id, **ev})

            ok = track.ok
            finite_z = np.isfinite(track.z)
            personal = cfg.get("personal") or {}

            # absolute level rules (raw value, usable days only)
            if cfg.get("absolute"):
                for level, bnd in cfg["absolute"].items():
                    lo = bnd.get("low")
                    hi = bnd.get("high")
                    cond = np.zeros(n, dtype=bool)
                    if lo is not None:
                        cond |= ok & (track.raw < float(lo))
                    if hi is not None:
                        cond |= ok & (track.raw > float(hi))
                    for s0, e0 in _runs(cond, 1):
                        v = track.raw[s0]
                        alerts.append(self._alert(
                            "body", astro_id, station_id, sig, "absolute", level,
                            s0, e0,
                            f"{friendly} {v:.1f} is outside the absolute limit for "
                            f"{level.replace('_', ' ')} (limit "
                            f"{'below ' + str(lo) if lo is not None else ''}"
                            f"{' above ' + str(hi) if hi is not None else ''})."))

            if not personal:
                continue

            direction = personal.get("direction", "both")

            # personal z rule with duration (never on a day the signal failed DQ:
            # the EWMA carries stale values across dead-sensor days, so finiteness
            # of z alone is NOT enough - require the day itself to be usable)
            z_w = float(personal.get("z_watch", 2.0))
            dur = int(personal.get("duration_days", 3))
            cond = _beyond(direction, np.nan_to_num(track.z, nan=0.0), z_w) \
                & finite_z & ok & (days >= cold)
            for s0, e0 in _runs(cond, dur):
                alerts.append(self._alert(
                    "body", astro_id, station_id, sig, "personal_z", "watch",
                    s0 + dur - 1, e0,
                    f"{friendly} has been {self._word(direction)} your personal "
                    f"normal for {dur} days or more (z {track.z[s0 + dur - 1]:.1f}, "
                    f"value {track.raw[s0 + dur - 1]:.1f} vs your reference "
                    f"{track.reference:.1f})."))

            # rate-of-change rule (week-over-week 7-day means)
            r_z = personal.get("rate_7d_z")
            if r_z:
                m7 = pd.Series(np.where(ok, track.raw, np.nan)).rolling(7, min_periods=4).mean()
                m7_prev = m7.shift(7)
                sd_diff = track.sigma * np.sqrt(2.0 / 7.0)
                zr = (m7 - m7_prev) / sd_diff
                cond = _beyond(direction, np.nan_to_num(zr.to_numpy(), nan=0.0), float(r_z)) \
                    & np.isfinite(zr.to_numpy()) & ok & (days >= cold)
                for s0, e0 in _runs(cond, 1):
                    alerts.append(self._alert(
                        "body", astro_id, station_id, sig, "rate_of_change", "watch",
                        s0, e0,
                        f"{friendly} changed faster than usual over the last week "
                        f"(week-over-week change {zr.iloc[s0]:.1f} robust sigmas)."))

            # slow-drift-vs-reference rule (14-day mean vs mission-start reference)
            d_z = personal.get("drift_ref_z")
            if d_z:
                d_dur = int(personal.get("drift_duration_days", 5))
                zs = np.nan_to_num(track.z_slow, nan=0.0)
                cond = _beyond(direction, zs, float(d_z)) \
                    & np.isfinite(track.z_slow) & ok & (days >= cold)
                for s0, e0 in _runs(cond, d_dur):
                    alerts.append(self._alert(
                        "body", astro_id, station_id, sig, "slow_drift", "watch",
                        s0 + d_dur - 1, e0,
                        f"{friendly} has drifted {self._word(direction)} your "
                        f"mission-start level for {d_dur} days or more "
                        f"(14-day mean z {track.z_slow[s0 + d_dur - 1]:.1f})."))

            # 30-day trend note (informational only, never an alert)
            if personal.get("z_watch"):
                slope_z = trailing_ols_slope(track.z, int(self.settings.get("trend_window_days", 30)))
                t = n - 1
                if np.isfinite(slope_z[t]) and np.isfinite(track.z[t]) and ok[t]:
                    z_now, sl = track.z[t], slope_z[t]
                    if direction == "upper" and sl > 0:
                        cross = (z_w - z_now) / sl
                    elif direction == "lower" and sl < 0:
                        cross = (-z_w - z_now) / sl
                    else:
                        cross = np.nan
                    if np.isfinite(cross) and 0 < cross <= 30:
                        notes.append({
                            "scope": "body", "astro_id": astro_id, "station_id": station_id,
                            "signal": sig, "day": t,
                            "text": (f"If the last 30 days' trend continues, {friendly.lower()} "
                                     f"reaches your personal watch level in about {cross:.0f} days "
                                     f"(informational note, not an alert).")})

        # check-in absolute backstop (raw self-report)
        for sig, cfg in (self.rules.get("checkin_signals") or {}).items():
            if sig not in df_a.columns:
                continue
            raw = df_a[sig].to_numpy(dtype=float)
            finite = np.isfinite(raw)
            scale = cfg.get("scale") or [None, None]
            in_scale = finite & (raw >= float(scale[0])) & (raw <= float(scale[1]))
            if cfg.get("watch_below") is not None:
                cond = in_scale & (raw <= float(cfg["watch_below"]))
                word = "at or below the low threshold"
            elif cfg.get("watch_above") is not None:
                cond = in_scale & (raw >= float(cfg["watch_above"]))
                word = "at or above the high threshold"
            else:
                continue
            dur = int(cfg.get("duration_days", 3))
            for s0, e0 in _runs(cond, dur):
                alerts.append(self._alert(
                    "body", astro_id, station_id, sig, "checkin_threshold", "watch",
                    s0 + dur - 1, e0,
                    f"{FRIENDLY.get(sig, sig)} has been {word} for {dur} days or more "
                    f"(latest {raw[s0 + dur - 1]:.1f} on a scale {cfg.get('scale')})."))

        return {"alerts": alerts, "sensor_messages": msgs, "trend_notes": notes,
                "tracks": tracks}

    # ---------------- cabin (per station) ----------------

    def run_station(self, station_id: str, df_s: pd.DataFrame) -> dict:
        alerts, msgs = [], []
        signals = {s: df_s[s].to_numpy(dtype=float) for s in df_s.columns
                   if s in self.rules["cabin_signals"]}
        tracks = make_tracks(signals, self.rules["cabin_signals"], self.settings)
        cold = int(self.settings.get("cold_start_days", 14))
        n = len(df_s)
        days = np.arange(n)

        for sig, track in tracks.items():
            cfg = self.rules["cabin_signals"][sig]
            friendly = FRIENDLY.get(sig, sig)
            for ev in track.sensor_events:
                msgs.append({"scope": "station", "astro_id": None,
                             "station_id": station_id, **ev})
            ok = track.ok

            if cfg.get("absolute"):
                for level, bnd in cfg["absolute"].items():
                    lo, hi = bnd.get("low"), bnd.get("high")
                    cond = np.zeros(n, dtype=bool)
                    if lo is not None:
                        cond |= ok & (track.raw < float(lo))
                    if hi is not None:
                        cond |= ok & (track.raw > float(hi))
                    for s0, e0 in _runs(cond, 1):
                        alerts.append(self._alert(
                            "station", None, station_id, sig, "absolute", level, s0, e0,
                            f"{friendly} {track.raw[s0]:.0f} is outside the cabin limit "
                            f"for {level.replace('_', ' ')}."))

            personal = cfg.get("personal") or {}
            if personal:
                direction = personal.get("direction", "both")
                z_w = float(personal.get("z_watch", 2.5))
                dur = int(personal.get("duration_days", 2))
                cond = _beyond(direction, np.nan_to_num(track.z, nan=0.0), z_w) \
                    & np.isfinite(track.z) & ok & (days >= cold)
                for s0, e0 in _runs(cond, dur):
                    alerts.append(self._alert(
                        "station", None, station_id, sig, "cabin_personal_z", "watch",
                        s0 + dur - 1, e0,
                        f"{friendly} has been {self._word(direction)} this station's "
                        f"usual range for {dur} days or more "
                        f"(z {track.z[s0 + dur - 1]:.1f})."))
        return {"alerts": alerts, "sensor_messages": msgs, "tracks": tracks}

    # ---------------- helpers ----------------

    @staticmethod
    def _word(direction: str) -> str:
        return {"upper": "above", "lower": "below"}.get(direction, "away from")

    @staticmethod
    def _alert(scope, astro_id, station_id, sig, rule, level, day, end_day, explanation):
        return {"scope": scope, "astro_id": astro_id, "station_id": station_id,
                "signal": sig, "rule": rule, "level": level,
                "alert_day": int(day), "end_day": int(end_day),
                "explanation": explanation}


def alerts_to_df(alerts: list) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(columns=["scope", "astro_id", "station_id", "signal",
                                     "rule", "level", "alert_day", "end_day", "explanation"])
    df = pd.DataFrame(alerts)
    df["level"] = pd.Categorical(df["level"], categories=LEVELS, ordered=True)
    return df
