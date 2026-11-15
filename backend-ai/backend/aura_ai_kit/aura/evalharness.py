"""Evaluation harness (AI plan, section 7).

Metrics, in the plan's own words:
  - share of real problems caught           -> detection table per problem type
  - share of alerts and notices that were real -> precision
  - false alarms per week (clean astronauts)   -> false-alarm stats
  - days until a problem is noticed           -> detection delay
  - model size, memory, speed                 -> size/latency report (capped-hardware
       benchmark under a 2-core / 2 GB cap is a pending item; numbers here are
       from the development machine and labeled as such)

Also implements the plan's stress test (extra noise + extra dropouts on the
held-out astronauts, detectors unchanged) and the data-quality integrity
check (no health alert and no notice may come from a failed-sensor day).
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from .notices import Notice, notice_episodes

ALERT_LEVELS = ("watch", "elevated", "urgent_review")


# ---------------------------------------------------------------- utilities

def body_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes[episodes["applies_to"] == "body"].copy()


def station_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes[episodes["applies_to"] == "station"].copy()


def clean_astro_ids(episodes: pd.DataFrame, astro_ids, stations_by_astro: dict) -> list:
    """Test astronauts with no body episode and no CO2 fault on their station."""
    fault_stations = set(station_episodes(episodes)["station_id"])
    used = set(body_episodes(episodes)["astro_id"])
    return [a for a in astro_ids
            if a not in used and stations_by_astro[a] not in fault_stations]


def clean_station_ids(episodes: pd.DataFrame, station_ids) -> list:
    fault = set(station_episodes(episodes)["station_id"])
    return [s for s in station_ids if s not in fault]


def model_size_kb(obj) -> float:
    return len(pickle.dumps(obj)) / 1024.0


def time_ms(fn, *args, repeat: int = 3) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    return (time.perf_counter() - t0) / repeat * 1000.0


# ------------------------------------------------------------- detection

def _affected(ep) -> set:
    """Signals an episode actually affects (for honest detection credit)."""
    aff = ep.get("affected") if hasattr(ep, "get") else None
    if isinstance(aff, str) and aff:
        return set(aff.split(","))
    if isinstance(aff, (list, tuple, set)):
        return set(aff)
    return None      # unknown -> do not filter by signal


def body_detection_table(alerts_df: pd.DataFrame, notices_if: dict, notices_ae: dict,
                         episodes: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Per body episode: detection delay in days (None = missed) per detector.

    Tier 0 credit requires the alert's signal to be one the problem actually
    affects (an unrelated false alert must not count as detection). AI
    notices are multi-signal by design, so they are credited on any notice.
    """
    rows = []
    a = alerts_df
    for _, ep in body_episodes(episodes).iterrows():
        astro, onset, end = ep["astro_id"], int(ep["onset_day"]), int(ep["end_day"])
        rec = {"problem": ep["problem"], "astro_id": astro,
               "onset_day": onset, "end_day": end}

        sel = a[(a["scope"] == "body") & (a["astro_id"] == astro)
                & (a["level"].astype(str).isin(ALERT_LEVELS))
                & (a["alert_day"] >= onset) & (a["alert_day"] <= end + horizon)]
        aff = _affected(ep)
        if aff is not None:
            sel = sel[sel["signal"].isin(aff)]
        rec["tier0"] = int(sel["alert_day"].min() - onset) if len(sel) else None

        for name, src in (("tier1a_if", notices_if), ("tier1a_ae", notices_ae)):
            days = [n.day for n in src.get(astro, [])
                    if onset <= n.day <= end + horizon]
            rec[name] = int(min(days) - onset) if days else None
        rows.append(rec)
    return pd.DataFrame(rows)


def station_detection_table(alerts_df: pd.DataFrame, station_notices: dict,
                            context_labels: dict, episodes: pd.DataFrame,
                            horizon: int = 7) -> pd.DataFrame:
    rows = []
    a = alerts_df
    for _, ep in station_episodes(episodes).iterrows():
        sid, onset, end = ep["station_id"], int(ep["onset_day"]), int(ep["end_day"])
        rec = {"problem": ep["problem"], "station_id": sid,
               "onset_day": onset, "end_day": end}
        sel = a[(a["scope"] == "station") & (a["station_id"] == sid)
                & (a["signal"] == "co2_ppm")
                & (a["level"].astype(str).isin(ALERT_LEVELS))
                & (a["alert_day"] >= onset) & (a["alert_day"] <= end + horizon)]
        rec["tier0_cabin"] = int(sel["alert_day"].min() - onset) if len(sel) else None

        days = [n.day for n in station_notices.get(sid, [])
                if onset <= n.day <= end + horizon]
        rec["tier1b_notice"] = int(min(days) - onset) if days else None

        lab_days = [l["day"] for l in context_labels.get(sid, [])
                    if l["label"] == "high_co2_exposure" and onset <= l["day"] <= end + horizon]
        rec["context_label"] = int(min(lab_days) - onset) if lab_days else None
        rows.append(rec)
    return pd.DataFrame(rows)


def detection_summary(det_table: pd.DataFrame, detectors: list) -> dict:
    out = {}
    for det in detectors:
        if det not in det_table:
            continue
        caught = det_table[det].notna()
        delays = det_table.loc[caught, det].astype(int)
        out[det] = {
            "share_caught": float(caught.mean()) if len(det_table) else None,
            "median_delay_days": float(delays.median()) if len(delays) else None,
        }
    return out


# --------------------------------------------------------- false alarms

def false_alarm_stats(alerts_df, notices_if, notices_ae, clean_astros, n_days) -> dict:
    weeks = max(len(clean_astros) * n_days / 7.0, 1e-9)
    a = alerts_df
    t0 = a[(a["scope"] == "body") & (a["astro_id"].isin(clean_astros))
           & (a["level"].astype(str).isin(ALERT_LEVELS))]
    by_rule = t0.groupby("rule").size().sort_values(ascending=False).to_dict()
    n_if = sum(len(notice_episodes(notices_if.get(astro, []))) for astro in clean_astros)
    n_ae = sum(len(notice_episodes(notices_ae.get(astro, []))) for astro in clean_astros)
    return {
        "n_clean_astronauts": len(clean_astros),
        "person_weeks": round(len(clean_astros) * n_days / 7.0, 1),
        "tier0_false_alert_episodes": int(len(t0)),
        "tier0_false_alerts_per_week": float(len(t0) / weeks),
        "tier0_false_alerts_by_rule": {str(k): int(v) for k, v in by_rule.items()},
        "tier1a_if_false_notice_episodes": int(n_if),
        "tier1a_if_false_notices_per_week": float(n_if / weeks),
        "tier1a_ae_false_notice_episodes": int(n_ae),
        "tier1a_ae_false_notices_per_week": float(n_ae / weeks),
    }


def station_false_alarm_stats(alerts_df, station_notices, clean_stations, n_days) -> dict:
    weeks = max(len(clean_stations) * n_days / 7.0, 1e-9)
    a = alerts_df
    t0 = a[(a["scope"] == "station") & (a["station_id"].isin(clean_stations))
           & (a["level"].astype(str).isin(ALERT_LEVELS))]
    n_1b = sum(len(notice_episodes(station_notices.get(s, []))) for s in clean_stations)
    return {
        "n_clean_stations": len(clean_stations),
        "station_weeks": round(len(clean_stations) * n_days / 7.0, 1),
        "tier0_cabin_false_alerts": int(len(t0)),
        "tier0_cabin_false_alerts_per_week": float(len(t0) / weeks),
        "tier1b_false_notice_episodes": int(n_1b),
        "tier1b_false_notices_per_week": float(n_1b / weeks),
    }


# ------------------------------------------------------------ precision

def precision_stats(alerts_df, notices_if, notices_ae, episodes,
                    margin_before: int = 3, horizon: int = 7) -> dict:
    eps = body_episodes(episodes)

    def real_for(astro, day, signal=None):
        sel = eps[(eps["astro_id"] == astro)]
        for _, ep in sel.iterrows():
            if ep["onset_day"] - margin_before <= day <= ep["end_day"] + horizon:
                aff = _affected(ep)
                if signal is not None and aff is not None and signal not in aff:
                    continue
                return True
        return False

    a = alerts_df[(alerts_df["scope"] == "body")
                  & (alerts_df["level"].astype(str).isin(ALERT_LEVELS))]
    t0_real = sum(real_for(r.astro_id, r.alert_day, r.signal) for r in a.itertuples())

    n_if_real = n_ae_real = n_if_tot = n_ae_tot = 0
    for astro, ns in notices_if.items():
        for n in notice_episodes(ns):
            n_if_tot += 1
            n_if_real += real_for(astro, n[0].day)
    for astro, ns in notices_ae.items():
        for n in notice_episodes(ns):
            n_ae_tot += 1
            n_ae_real += real_for(astro, n[0].day)
    return {
        "tier0_alerts_total": int(len(a)),
        "tier0_share_real": float(t0_real / len(a)) if len(a) else None,
        "tier1a_if_notice_episodes": int(n_if_tot),
        "tier1a_if_share_real": float(n_if_real / n_if_tot) if n_if_tot else None,
        "tier1a_ae_notice_episodes": int(n_ae_tot),
        "tier1a_ae_share_real": float(n_ae_real / n_ae_tot) if n_ae_tot else None,
    }


# ------------------------------------------------- data-quality integrity

def dq_integrity_check(alerts_df, tracks_body) -> dict:
    """A health alert must never sit on a day where that signal failed DQ."""
    bad = 0
    for r in alerts_df[alerts_df["scope"] == "body"].itertuples():
        tr = tracks_body.get(r.astro_id, {}).get(r.signal)
        if tr is None:
            continue
        if not bool(tr.ok[r.alert_day]):
            bad += 1
    return {"alerts_on_failed_sensor_days": int(bad),
            "assertion_passed": bad == 0}


# ------------------------------------------------------- context events

def storm_context_stats(context_labels: dict, context_events: pd.DataFrame,
                        test_stations) -> dict:
    evs = context_events[context_events["station_id"].isin(test_stations)]
    covered = 0
    for _, ev in evs.iterrows():
        labels = context_labels.get(ev["station_id"], [])
        hit = any(l["label"] == "elevated_radiation_context"
                  and l["day"] <= ev["end"] + 1 and l["end_day"] >= ev["start"] - 1
                  for l in labels)
        covered += hit
    false_rad = 0
    for sid in test_stations:
        for l in context_labels.get(sid, []):
            if l["label"] != "elevated_radiation_context":
                continue
            overlap = any(l["day"] <= ev["end"] + 1 and l["end_day"] >= ev["start"] - 1
                          for _, ev in evs[evs["station_id"] == sid].iterrows())
            if not overlap:
                false_rad += 1
    return {"storms_total": int(len(evs)),
            "storms_labeled": int(covered),
            "radiation_labels_without_storm": int(false_rad)}


# ------------------------------------------------------------- stress

def perturb_df(df: pd.DataFrame, rng, noise_mult: float, dropout_p: float,
               body_sigmas: dict, cabin_sigmas: dict) -> pd.DataFrame:
    """Add extra noise and dropouts to a simulated df (stress test).

    The harness knows the generator's noise sigmas; in a real deployment
    noise levels are unknown (documented limitation of the stress test).
    """
    df = df.copy()
    for sig, sd in {**body_sigmas, **cabin_sigmas}.items():
        if sig not in df.columns:
            continue
        vals = df[sig].to_numpy(dtype=float, copy=True)
        finite = np.isfinite(vals)
        noise = rng.normal(0.0, noise_mult * sd, size=len(vals))
        vals[finite] += noise[finite]
        drop = rng.random(len(vals)) < dropout_p
        vals[drop] = np.nan
        df[sig] = vals
    return df
