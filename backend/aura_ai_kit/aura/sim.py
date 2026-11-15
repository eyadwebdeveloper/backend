"""Virtual astronaut simulator (AURA AI kit).

Generates per-person daily health series and per-station cabin environment
series with weekly cycles, autocorrelated noise, dropouts and sensor
faults, then injects known problems at known times. The injection table is
the ground truth used by the evaluation harness (AI plan, section 7).

Effect DIRECTIONS follow the published findings named in
config/problem_profiles.yaml. Magnitudes and prevalences are engineering
placeholders pending medical review.

Deliberate constraints (documented, to keep evaluation interpretable):
  - at most one slow-drift problem per astronaut (deconditioning XOR
    behavioral decline); the short infection-like event may co-occur
  - a fixed fraction of astronauts is kept problem-free (clean controls)
  - solar-storm dose spikes are recorded as context events, not problems
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BODY_SIGNALS = ["resting_hr", "hrv_rmssd", "sleep_hours", "active_min",
                "skin_temp_c", "mood_score", "fatigue_score"]
CABIN_SIGNALS = ["co2_ppm", "cabin_temp_c", "rh_pct", "noise_db", "dose_usv_day"]

# physical plausibility bounds used only to draw corrupt spike values
PHYS_BOUNDS = {
    "resting_hr": (5.0, 240.0), "hrv_rmssd": (2.0, 400.0),
    "sleep_hours": (-3.0, 22.0), "active_min": (0.0, 900.0),
    "skin_temp_c": (25.0, 42.0), "mood_score": (-2.0, 9.0),
    "fatigue_score": (-2.0, 9.0), "co2_ppm": (50.0, 16000.0),
    "cabin_temp_c": (5.0, 45.0), "rh_pct": (0.0, 100.0),
    "noise_db": (10.0, 120.0), "dose_usv_day": (-100.0, 20000.0),
}

# generation-time plausibility clips for NORMAL data (spikes are added later
# and must survive as corrupt values, so they are never clipped)
CLIP_BOUNDS = {
    "resting_hr": (35, 160), "hrv_rmssd": (10, 150),
    "sleep_hours": (3, 12), "active_min": (10, 300),
    "skin_temp_c": (31.5, 35.5), "mood_score": (1, 5),
    "fatigue_score": (1, 5),
}

# daily noise sigmas used by the generator (native units). Also used by the
# evaluation harness stress test, which needs to know the noise scale.
BODY_STRESS_SD = {
    "resting_hr": 1.8, "hrv_rmssd": 2.7, "sleep_hours": 0.55,
    "active_min": 14.0, "skin_temp_c": 0.15,
    "mood_score": 0.25, "fatigue_score": 0.25,
}
CABIN_STRESS_SD = {
    "co2_ppm": 90.0, "cabin_temp_c": 0.35, "rh_pct": 2.5,
    "noise_db": 2.5, "dose_usv_day": 25.0,
}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ar1(n: int, phi: float, sigma: float, rng) -> np.ndarray:
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _episode_envelope(n_days: int, onset: int, ramp_days: int,
                      hold_days: int, recovery_days: int = 0) -> np.ndarray:
    """Effect envelope in [0, 1]: linear ramp up (smoothed), hold, optional recovery."""
    env = np.zeros(n_days)
    for t in range(n_days):
        d = t - onset
        if d < 0:
            v = 0.0
        elif d < ramp_days:
            v = d / max(ramp_days, 1)
        elif d < ramp_days + hold_days:
            v = 1.0
        elif recovery_days > 0 and d < ramp_days + hold_days + recovery_days:
            v = 1.0 - (d - ramp_days - hold_days) / recovery_days
        else:
            v = 0.0
        env[t] = _smoothstep(np.array(v))
    return env


def _sample_person_base(rng) -> dict:
    return {
        "resting_hr": float(np.clip(rng.normal(62, 6), 50, 78)),
        "hrv_rmssd": float(np.clip(np.exp(rng.normal(np.log(38.0), 0.30)), 22, 75)),
        "sleep_hours": float(np.clip(rng.normal(7.2, 0.5), 5.8, 8.6)),
        "active_min": float(np.clip(rng.normal(115, 18), 70, 160)),
        "skin_temp_c": float(np.clip(rng.normal(33.6, 0.4), 32.5, 34.8)),
        "mood_score": float(np.clip(rng.normal(3.8, 0.45), 2.8, 4.6)),
        "fatigue_score": float(np.clip(rng.normal(2.2, 0.45), 1.3, 3.3)),
    }


def _gen_body_normal(base: dict, n_days: int, rng) -> dict:
    days = np.arange(n_days)
    ph = rng.uniform(0, 2 * np.pi)
    wk = np.sin(2 * np.pi * days / 7.0 + ph)
    s = {}
    s["resting_hr"] = base["resting_hr"] + 0.8 * wk + _ar1(n_days, 0.5, 1.8, rng)
    s["hrv_rmssd"] = np.exp(np.log(base["hrv_rmssd"]) + 0.04 * wk + _ar1(n_days, 0.5, 0.07, rng))
    s["sleep_hours"] = base["sleep_hours"] + 0.25 * wk + _ar1(n_days, 0.5, 0.55, rng)
    s["active_min"] = base["active_min"] + 8.0 * wk + _ar1(n_days, 0.5, 14.0, rng)
    s["skin_temp_c"] = base["skin_temp_c"] + 0.05 * wk + _ar1(n_days, 0.5, 0.15, rng)
    s["mood_score"] = base["mood_score"] + 0.10 * wk + _ar1(n_days, 0.5, 0.25, rng)
    s["fatigue_score"] = base["fatigue_score"] - 0.08 * wk + _ar1(n_days, 0.5, 0.25, rng)
    for k, (lo, hi) in CLIP_BOUNDS.items():
        s[k] = np.clip(s[k], lo, hi)
    return s


def _gen_cabin_normal(n_days: int, rng) -> dict:
    days = np.arange(n_days)
    ph = rng.uniform(0, 2 * np.pi)
    wk = np.sin(2 * np.pi * days / 7.0 + ph)
    s = {}
    s["co2_ppm"] = 2700.0 + rng.normal(0, 150) + 120.0 * wk + _ar1(n_days, 0.6, 90.0, rng)
    s["cabin_temp_c"] = 22.5 + 0.3 * wk + _ar1(n_days, 0.6, 0.35, rng)
    s["rh_pct"] = 45.0 + 1.5 * wk + _ar1(n_days, 0.6, 2.5, rng)
    noise = 57.0 + _ar1(n_days, 0.5, 2.5, rng)
    for t in range(n_days):
        if rng.random() < 0.04:
            noise[t] += rng.uniform(4.0, 9.0)
    s["noise_db"] = noise
    s["dose_usv_day"] = 400.0 + _ar1(n_days, 0.7, 25.0, rng)
    return s


def _add_storm_events(dose: np.ndarray, rng):
    """Solar-storm-like context events: dose spikes for 1-2 days. Not problems."""
    events = []
    n = len(dose)
    if n > 40 and rng.random() < 0.6:
        start = int(rng.integers(20, n - 4))
        length = int(rng.integers(1, 3))
        mult = rng.uniform(1.8, 2.8)
        dose[start:start + length] *= mult
        events.append((start, min(start + length, n) - 1))
    return dose, events


def _apply_effects(signals: dict, effects: dict, env: np.ndarray) -> None:
    for sig, spec in effects.items():
        if "mult" in spec:
            signals[sig] = signals[sig] * (1.0 + (spec["mult"] - 1.0) * env)
        elif "add" in spec:
            signals[sig] = signals[sig] + spec["add"] * env


def _assign_body_problems(profiles: dict, rng, n_astronauts: int, clean_idx: set,
                          n_days: int) -> dict:
    assigns = {i: [] for i in range(n_astronauts)}
    for i in range(n_astronauts):
        if i in clean_idx:
            continue
        got_slow = False
        for ptype in ("deconditioning", "infection_like", "behavioral_decline"):
            p = profiles["problems"][ptype]
            if rng.random() >= float(p["prevalence"]):
                continue
            if ptype in ("deconditioning", "behavioral_decline"):
                if got_slow:            # at most one slow-drift problem per person
                    continue
                got_slow = True
            onset = _draw_onset(p, rng, n_days)
            dur = int(rng.integers(p["duration_days"][0], p["duration_days"][1] + 1))
            assigns[i].append({
                "problem": ptype, "onset": onset, "duration": dur,
                "ramp_days": int(p["ramp_days"]),
                "recovery_days": int(p["recovery_days"]),
                "effects": p.get("body_effects", {}),
                "affected": sorted(p.get("body_effects", {}).keys()),
            })
    return assigns


def _draw_onset(p: dict, rng, n_days: int) -> int:
    """Draw an onset day, clamped so the episode starts inside the mission."""
    lo = int(p["onset_range"]["min_day"])
    hi = min(int(p["onset_range"]["max_day"]), n_days - 5)
    if hi < lo:
        hi = lo
    return int(rng.integers(lo, hi + 1))


def _assign_station_problems(profiles: dict, rng, station_ids: list, n_days: int) -> dict:
    out = {}
    p = profiles["problems"]["co2_scrubber_fault"]
    for sid in station_ids:
        if rng.random() >= float(p["prevalence"]):
            continue
        onset = _draw_onset(p, rng, n_days)
        dur = int(rng.integers(p["duration_days"][0], p["duration_days"][1] + 1))
        out[sid] = {
            "problem": "co2_scrubber_fault", "onset": onset, "duration": dur,
            "ramp_days": int(p["ramp_days"]), "recovery_days": int(p["recovery_days"]),
            "station_effects": p.get("station_effects", {}),
            "body_effects": p.get("body_effects", {}),
        }
    return out


def _inject_dq_body(signals: dict, rng, dq_cfg: dict, n_days: int, astro_id: str) -> list:
    """Corrupt body values in place. Returns verification-only event records."""
    events = []
    for sig in BODY_SIGNALS:
        arr = signals[sig]
        miss = rng.random(n_days) < float(dq_cfg["missing_days"]["per_day_prob"])
        spk = rng.random(n_days) < float(dq_cfg["spike_bursts"]["per_day_prob"])
        lo, hi = PHYS_BOUNDS[sig]
        span = hi - lo
        n_miss = int(miss.sum())
        n_spk = int((spk & ~miss).sum())
        for t in np.nonzero(miss)[0]:
            arr[t] = np.nan
        for t in np.nonzero(spk & ~miss)[0]:
            if rng.random() < 0.5:
                arr[t] = rng.uniform(hi + 0.2 * span, hi + 0.6 * span)
            else:
                arr[t] = rng.uniform(lo - 0.6 * span, lo - 0.2 * span)
        if n_miss:
            events.append({"astro_id": astro_id, "signal": sig, "kind": "missing_days", "count": n_miss})
        if n_spk:
            events.append({"astro_id": astro_id, "signal": sig, "kind": "spikes", "count": n_spk})

    dead = dq_cfg["dead_sensor"]
    if rng.random() < float(dead["prevalence"]):
        sig = BODY_SIGNALS[int(rng.integers(0, len(BODY_SIGNALS)))]
        length = int(rng.integers(dead["length_days"][0], dead["length_days"][1] + 1))
        start = int(rng.integers(15, max(16, n_days - length)))
        signals[sig][start:start + length] = np.nan
        events.append({"astro_id": astro_id, "signal": sig, "kind": "dead_sensor",
                       "start": start, "end": start + length - 1})

    stuck = dq_cfg["stuck_value"]
    if rng.random() < float(stuck["prevalence"]):
        sig = BODY_SIGNALS[int(rng.integers(0, len(BODY_SIGNALS)))]
        length = int(rng.integers(stuck["length_days"][0], stuck["length_days"][1] + 1))
        start = int(rng.integers(15, max(16, n_days - length)))
        ref = signals[sig][start - 1] if start > 0 and np.isfinite(signals[sig][start - 1]) \
            else float(np.nanmedian(signals[sig]))
        signals[sig][start:start + length] = ref
        events.append({"astro_id": astro_id, "signal": sig, "kind": "stuck_value",
                       "start": start, "end": start + length - 1})
    return events


def simulate(profiles: dict, n_astronauts: int, n_days: int, rng,
             station_ids: list, astro_id_offset: int = 0,
             force_clean_fraction: float = 0.30,
             ensure_problems: tuple = (),
             ensure_station_problem: bool = False) -> dict:
    """Run one split (train or test). Returns dict with df, episodes,
    dq_events, context_events.

    ensure_problems / ensure_station_problem guarantee that the named problem
    types appear at least once (used on the test split so that every run
    exercises every detection path; the draw stays seeded and reproducible).
    """
    dq_cfg = profiles["dq_faults"]

    # --- stations ---
    stations = {sid: _gen_cabin_normal(n_days, rng) for sid in station_ids}
    context_events = []
    for sid in station_ids:
        dose, evs = _add_storm_events(stations[sid]["dose_usv_day"], rng)
        stations[sid]["dose_usv_day"] = dose
        context_events += [{"station_id": sid, "kind": "solar_storm",
                            "start": a, "end": b} for a, b in evs]

    station_problems = _assign_station_problems(profiles, rng, station_ids, n_days)
    if ensure_station_problem and not station_problems:
        p = profiles["problems"]["co2_scrubber_fault"]
        sid = station_ids[0]
        onset = _draw_onset(p, rng, n_days)
        dur = int(rng.integers(p["duration_days"][0], p["duration_days"][1] + 1))
        station_problems[sid] = {
            "problem": "co2_scrubber_fault", "onset": onset, "duration": dur,
            "ramp_days": int(p["ramp_days"]), "recovery_days": int(p["recovery_days"]),
            "station_effects": p.get("station_effects", {}),
            "body_effects": p.get("body_effects", {}),
        }
    station_env = {}
    for sid, pr in station_problems.items():
        env = _episode_envelope(n_days, pr["onset"], pr["ramp_days"],
                                max(pr["duration"] - pr["ramp_days"], 1), pr["recovery_days"])
        _apply_effects(stations[sid], pr["station_effects"], env)
        station_env[sid] = env

    # --- astronauts ---
    n_clean = int(np.ceil(force_clean_fraction * n_astronauts))
    clean_idx = set(rng.permutation(n_astronauts)[:n_clean].tolist())
    body_assign = _assign_body_problems(profiles, rng, n_astronauts, clean_idx, n_days)

    # guarantee the requested problem types appear at least once
    present = {pr["problem"] for lst in body_assign.values() for pr in lst}
    for ptype in ensure_problems:
        if ptype in present or ptype not in profiles["problems"]:
            continue
        p = profiles["problems"][ptype]
        cands = []
        for i in range(n_astronauts):
            if i in clean_idx:
                continue
            types = {pr["problem"] for pr in body_assign[i]}
            if ptype in types:
                continue
            if ptype in ("deconditioning", "behavioral_decline") \
                    and types & {"deconditioning", "behavioral_decline"}:
                continue
            cands.append(i)
        if not cands:
            cands = [i for i in range(n_astronauts) if i not in clean_idx]
        if not cands:
            continue
        i = int(rng.choice(cands))
        onset = _draw_onset(p, rng, n_days)
        dur = int(rng.integers(p["duration_days"][0], p["duration_days"][1] + 1))
        body_assign[i].append({
            "problem": ptype, "onset": onset, "duration": dur,
            "ramp_days": int(p["ramp_days"]),
            "recovery_days": int(p["recovery_days"]),
            "effects": p.get("body_effects", {}),
            "affected": sorted(p.get("body_effects", {}).keys()),
        })

    rows = []
    episodes = []
    dq_events = []
    for i in range(n_astronauts):
        astro_id = f"A{astro_id_offset + i:03d}"
        sid = station_ids[i % len(station_ids)]
        base = _sample_person_base(rng)
        signals = _gen_body_normal(base, n_days, rng)

        for pr in body_assign[i]:
            env = _episode_envelope(n_days, pr["onset"], pr["ramp_days"],
                                    max(pr["duration"] - pr["ramp_days"], 1), pr["recovery_days"])
            _apply_effects(signals, pr["effects"], env)
            episodes.append({"astro_id": astro_id, "station_id": sid,
                             "problem": pr["problem"], "applies_to": "body",
                             "onset_day": pr["onset"],
                             "end_day": min(pr["onset"] + pr["duration"] - 1, n_days - 1),
                             "ramp_days": pr["ramp_days"],
                             "affected": ",".join(pr.get("affected", []))})

        if sid in station_problems:
            _apply_effects(signals, station_problems[sid]["body_effects"], station_env[sid])

        dq_events += _inject_dq_body(signals, rng, dq_cfg, n_days, astro_id)

        cabin = stations[sid]
        for t in range(n_days):
            row = {"astro_id": astro_id, "station_id": sid, "day": t}
            row.update({k: float(v[t]) for k, v in signals.items()})
            row.update({k: float(cabin[k][t]) for k in CABIN_SIGNALS})
            rows.append(row)

    for sid, pr in station_problems.items():
        episodes.append({"astro_id": None, "station_id": sid,
                         "problem": "co2_scrubber_fault", "applies_to": "station",
                         "onset_day": pr["onset"],
                         "end_day": min(pr["onset"] + pr["duration"] - 1, n_days - 1),
                         "ramp_days": pr["ramp_days"]})

    df = pd.DataFrame(rows)
    ep_cols = ["astro_id", "station_id", "problem", "applies_to",
               "onset_day", "end_day", "ramp_days", "affected"]
    return {
        "df": df,
        "episodes": pd.DataFrame(episodes, columns=ep_cols),
        "dq_events": pd.DataFrame(dq_events),
        "context_events": pd.DataFrame(context_events,
                                       columns=["station_id", "kind", "start", "end"]),
    }
