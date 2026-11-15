"""End-to-end pipeline for the AURA AI kit.

    python run_pipeline.py

Steps:
  1. simulate train and test splits (disjoint astronauts AND stations)
  2. Tier 0 rules on both splits (no training; rules come from YAML)
  3. Tier 1a / 1b: fit on normal days of TRAIN split, score TEST split,
     fire AI notices through the notice gate
  4. Tier 2a: fit on simulator-labeled train days, predict test days
  5. Tier 2b: fit baseline + learned forecaster on train windows,
     evaluate MAE and days-to-limit on held-out astronauts
  6. stress test: perturbed test split, same fitted models
  7. metrics.json + report.md + charts into results/

All output is simulated-data-only with engineering-placeholder rules.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score

from aura import __version__
from aura.evalharness import (
    body_detection_table, clean_astro_ids, detection_summary, dq_integrity_check,
    false_alarm_stats, model_size_kb, precision_stats, station_detection_table,
    station_false_alarm_stats, storm_context_stats, time_ms, perturb_df)
from aura.notices import fire_notices
from aura.sim import (BODY_SIGNALS, BODY_STRESS_SD, CABIN_SIGNALS, CABIN_STRESS_SD,
                      simulate)
from aura.tier0 import Tier0Engine, alerts_to_df
from aura.tier1a import (BodyAnomalyDetector, body_feature_matrix,
                         run_body_scoring)
from aura.tier1b import (StationAnomalyDetector, cabin_feature_matrix,
                         context_label_episodes, run_station_scoring)
from aura.tier2a import BANDS, FEATURES as T2A_FEATURES, \
    DeconditioningRiskClassifier, build_decond_table
from aura.tier2b import (FORECAST_SIGNALS, TrendBaseline, build_windows_from_tracks,
                         crossing_day, make_learned_forecaster)
from aura.web_snapshot import build_web_snapshots

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def confusion_md(cm, labels):
    lines = ["| actual \\ predicted | " + " | ".join(labels) + " |",
             "|---|" + "---|" * len(labels)]
    for i, lab in enumerate(labels):
        lines.append(f"| {lab} | " + " | ".join(str(cm[i, j]) for j in range(len(labels))) + " |")
    return "\n".join(lines)


def mae_over_windows(mus, tgts, masks) -> dict:
    out = {h: {sig: [] for sig in FORECAST_SIGNALS} for h in (7, 14)}
    for mu, tg, mk in zip(mus, tgts, masks):
        for h in (7, 14):
            for j, sig in enumerate(FORECAST_SIGNALS):
                if mk[h - 1, j] and np.isfinite(mu[h - 1, j]):
                    out[h][sig].append(abs(mu[h - 1, j] - tg[h - 1, j]))
    return {h: {sig: (float(np.mean(v)) if v else float("nan"))
                for sig, v in d.items()} for h, d in out.items()}


def run_tier0(engine: Tier0Engine, df: pd.DataFrame):
    alerts, msgs, tracks_body, tracks_station = [], [], {}, {}
    for astro_id, g in df.groupby("astro_id", sort=True):
        sid = g["station_id"].iloc[0]
        res = engine.run_body(astro_id, sid, g)
        alerts += res["alerts"]
        msgs += res["sensor_messages"]
        tracks_body[astro_id] = res["tracks"]
        tracks_body[astro_id]["__station__"] = sid
    for sid, g in df.groupby("station_id", sort=True):
        g_s = g.sort_values("day").drop_duplicates("day")
        res = engine.run_station(sid, g_s)
        alerts += res["alerts"]
        msgs += res["sensor_messages"]
        tracks_station[sid] = res["tracks"]
    return alerts_to_df(alerts), msgs, tracks_body, tracks_station


def episode_bad_days(episodes: pd.DataFrame, n_days: int,
                     astro_id=None, station_id=None,
                     body_margin: int = 3, station_margin: int = 3) -> np.ndarray:
    """Days to EXCLUDE from 'normal' training data (problem windows)."""
    bad = np.zeros(n_days, dtype=bool)
    eps = episodes
    if astro_id is not None:
        eps = eps[eps["astro_id"] == astro_id]
    if station_id is not None:
        eps = eps[(eps["station_id"] == station_id) & (eps["applies_to"] == "station")]
        astro_id = None
    for _, ep in eps.iterrows():
        m = int(body_margin if ep["applies_to"] == "body" else station_margin)
        bad[max(0, int(ep["onset_day"]) - m):min(n_days, int(ep["end_day"]) + m + 1)] = True
    return bad


def main():
    t_start = time.time()
    base = Path(__file__).resolve().parent
    cfg = load_yaml(base / "config" / "pipeline.yaml")
    rules = load_yaml(base / "config" / "tier0_rules.yaml")
    profiles = load_yaml(base / "config" / "problem_profiles.yaml")
    seed = int(cfg["seed"])
    n_days = int(cfg["mission"]["n_days"])
    cold = int(rules["settings"]["cold_start_days"])
    n_train = int(cfg["mission"]["n_train_astronauts"])
    n_test = int(cfg["mission"]["n_test_astronauts"])
    n_st = int(cfg["mission"]["stations_per_split"])
    cons = int(cfg["notices"]["consecutive_days"])
    cool = int(cfg["notices"]["cooldown_days"])
    high_q = float(cfg["notices"]["high_quantile"])
    results_dir = base / cfg["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    # ---------------- 1. simulate ----------------
    print("[1/7] simulating virtual astronauts ...", flush=True)
    train_stations = [f"S{i}" for i in range(n_st)]
    test_stations = [f"S{i}" for i in range(n_st, 2 * n_st)]
    train = simulate(profiles, n_train, n_days, np.random.default_rng(seed),
                     train_stations, astro_id_offset=0,
                     force_clean_fraction=float(cfg["mission"]["force_clean_fraction"]))
    ensure = tuple(cfg["mission"].get("ensure_test_problems", []))
    ensure_station = bool(cfg["mission"].get("ensure_test_station_problem", False))
    test = simulate(profiles, n_test, n_days, np.random.default_rng(seed + 1),
                    test_stations, astro_id_offset=100,
                    force_clean_fraction=float(cfg["mission"]["force_clean_fraction"]),
                    ensure_problems=ensure,
                    ensure_station_problem=ensure_station)
    test_astro_ids = sorted(test["df"]["astro_id"].unique())

    # ---------------- 2. Tier 0 ----------------
    print("[2/7] Tier 0 rules engine ...", flush=True)
    engine = Tier0Engine(rules)
    t0_train, msgs_train, tracks_body_train, tracks_st_train = run_tier0(engine, train["df"])
    t0_test, msgs_test, tracks_body_test, tracks_st_test = run_tier0(engine, test["df"])

    # ---------------- 3. Tier 1a / 1b ----------------
    print("[3/7] Tier 1a body anomaly detector ...", flush=True)
    rows = []
    for astro_id, tracks in tracks_body_train.items():
        X, _, ok_counts = body_feature_matrix(tracks)
        bad = episode_bad_days(train["episodes"], n_days, astro_id=astro_id)
        bad |= episode_bad_days(train["episodes"], n_days,
                                station_id=tracks.get("__station__"))
        valid = (ok_counts == len(BODY_SIGNALS)) & (np.arange(n_days) >= cold) & ~bad
        valid &= np.isfinite(X).all(axis=1)
        if valid.any():
            rows.append(X[valid])
    X_normal = np.vstack(rows)
    print(f"      training days (normal only): {len(X_normal)}", flush=True)
    det1a = BodyAnomalyDetector(
        contamination=float(cfg["models"]["tier1a"]["contamination"]),
        seed=seed).fit(X_normal, high_quantile=high_q)

    notices_if, notices_ae, scores_test = {}, {}, {}
    for astro_id in test_astro_ids:
        sc = run_body_scoring(det1a, tracks_body_test[astro_id], cold_start=cold)
        scores_test[astro_id] = sc
        notices_if[astro_id] = fire_notices(
            sc["strangeness_if"], sc["explanations"], det1a.high_thresh_if,
            consecutive=cons, cooldown=cool, source="tier1a_isolation_forest")
        notices_ae[astro_id] = fire_notices(
            sc["strangeness_ae"], sc["explanations"], det1a.high_thresh_ae,
            consecutive=cons, cooldown=cool, source="tier1a_autoencoder")

    print("[3/7] Tier 1b cabin anomaly detector ...", flush=True)
    rows = []
    for sid, tracks in tracks_st_train.items():
        X, _, ok_counts = cabin_feature_matrix(tracks)
        bad = episode_bad_days(train["episodes"], n_days, station_id=sid)
        for _, ev in train["context_events"].iterrows():
            if ev["station_id"] == sid:
                bad[max(0, int(ev["start"]) - 2):int(ev["end"]) + 3] = True
        valid = (ok_counts == len(CABIN_SIGNALS)) & (np.arange(n_days) >= cold) & ~bad
        valid &= np.isfinite(X).all(axis=1)
        if valid.any():
            rows.append(X[valid])
    det1b = StationAnomalyDetector(
        contamination=float(cfg["models"]["tier1b"]["contamination"]),
        seed=seed).fit(np.vstack(rows), high_quantile=high_q)

    station_notices, context_labels = {}, {}
    for sid, tracks in tracks_st_test.items():
        sc = run_station_scoring(det1b, tracks, cold_start=cold)
        station_notices[sid] = fire_notices(
            sc["strangeness"], sc["explanations"], det1b.high_thresh,
            consecutive=cons, cooldown=cool, source="tier1b_station")
        context_labels[sid] = context_label_episodes(tracks, rules.get("context_labels") or {})

    # ---------------- 4. Tier 2a ----------------
    print("[4/7] Tier 2a deconditioning risk classifier ...", flush=True)
    tab_tr = build_decond_table(tracks_body_train, train["episodes"], n_days, cold)
    clf2a = DeconditioningRiskClassifier(seed=seed).fit(
        tab_tr[T2A_FEATURES].to_numpy(), tab_tr["label"].to_numpy())
    tab_te = build_decond_table(tracks_body_test, test["episodes"], n_days, cold)
    bands, conf = clf2a.predict(tab_te[T2A_FEATURES].to_numpy())
    tab_te["pred"] = bands
    tab_te["confidence"] = conf
    macro_f1 = float(f1_score(tab_te["label"], tab_te["pred"], average="macro"))
    cm = confusion_matrix(tab_te["label"], tab_te["pred"], labels=BANDS)
    # infection confusion: infection-episode days predicted as established
    inf_eps = test["episodes"][(test["episodes"]["problem"] == "infection_like")]
    n_inf_days = n_inf_est = 0
    for _, ep in inf_eps.iterrows():
        sel = tab_te[(tab_te["astro_id"] == ep["astro_id"])
                     & (tab_te["day"] >= ep["onset_day"])
                     & (tab_te["day"] <= ep["end_day"])]
        n_inf_days += len(sel)
        n_inf_est += int((sel["pred"] == "established").sum())

    # ---------------- 5. Tier 2b ----------------
    print("[5/7] Tier 2b forecaster ...", flush=True)
    W = int(cfg["models"]["tier2b"]["input_window_days"])
    H = int(cfg["models"]["tier2b"]["horizon_days"])
    win, tgt, msk, _ = build_windows_from_tracks(tracks_body_train, W=W, H=H, stride=1)
    print(f"      training windows: {len(win)}", flush=True)
    baseline_fb = TrendBaseline(W, H)
    learned_fb = make_learned_forecaster(
        cfg["models"]["tier2b"]["learned_backend"],
        input_window=W, horizon=H, seed=seed).fit(win, tgt, msk)
    wte, tte, mte, _ = build_windows_from_tracks(tracks_body_test, W=W, H=H, stride=7)
    mus_b = [baseline_fb.forecast(w)[0] for w in wte]
    mus_l = [learned_fb.forecast(w)[0] for w in wte]
    mae_b = mae_over_windows(mus_b, tte, mte)
    mae_l = mae_over_windows(mus_l, tte, mte)
    mean14 = lambda d: float(np.nanmean([v for v in d[14].values()]))
    verdict = bool(mean14(mae_l) < mean14(mae_b))

    # days-to-limit on deconditioning astronauts (snapshot at onset + 21)
    d2l = []
    df_by_astro = {a: g.sort_values("day") for a, g in test["df"].groupby("astro_id")}
    for _, ep in test["episodes"][test["episodes"]["problem"] == "deconditioning"].iterrows():
        astro, onset = ep["astro_id"], int(ep["onset_day"])
        snap = onset + 21
        if snap + H >= n_days or snap < W:
            continue
        tracks = tracks_body_test[astro]
        g = df_by_astro[astro]
        hist = np.column_stack([tracks[s].ewma for s in FORECAST_SIGNALS])[snap - W:snap]
        realized = g[list(FORECAST_SIGNALS)].to_numpy()[snap:snap + H]
        mu_b, _ = baseline_fb.forecast(hist)
        mu_l, _ = learned_fb.forecast(hist)

        def pred_cross(mu):
            best = None
            for j, sig in enumerate(FORECAST_SIGNALS):
                tr = tracks[sig]
                direction = "upper" if sig == "resting_hr" else "lower"
                for h in range(H):
                    if not np.isfinite(mu[h, j]):
                        continue
                    z = (mu[h, j] - tr.reference) / tr.sigma
                    hit = (direction == "upper" and z >= 2.0) or \
                          (direction == "lower" and z <= -2.0)
                    if hit:
                        d = snap + 1 + h
                        best = d if best is None else min(best, d)
                        break
            return best

        actual = None
        for sig, direction in (("hrv_rmssd", "lower"), ("resting_hr", "upper")):
            d = crossing_day(tracks[sig].z_slow, 2.0, direction, start=snap, limit=H)
            if d is not None:
                actual = d if actual is None else min(actual, d)
        if actual is None:
            continue
        d2l.append({"astro_id": astro, "actual": actual,
                    "baseline": pred_cross(mu_b), "learned": pred_cross(mu_l)})

    # ---------------- 6. metrics + stress ----------------
    print("[6/7] evaluation + stress test ...", flush=True)
    det_table = body_detection_table(t0_test, notices_if, notices_ae, test["episodes"])
    st_table = station_detection_table(t0_test, station_notices, context_labels,
                                       test["episodes"])
    stations_by_astro = {a: tracks_body_test[a]["__station__"] for a in test_astro_ids}
    clean_ids = clean_astro_ids(test["episodes"], test_astro_ids, stations_by_astro)
    clean_st = [s for s in test_stations
                if s not in set(test["episodes"][test["episodes"]["applies_to"] == "station"]["station_id"])]
    fa = false_alarm_stats(t0_test, notices_if, notices_ae, clean_ids, n_days)
    fa_st = station_false_alarm_stats(t0_test, station_notices, clean_st, n_days)
    prec = precision_stats(t0_test, notices_if, notices_ae, test["episodes"])
    dq = dq_integrity_check(t0_test, tracks_body_test)
    storm = storm_context_stats(context_labels, test["context_events"], test_stations)

    rng_stress = np.random.default_rng(seed + 2)
    df_stress = perturb_df(test["df"], rng_stress,
                           float(cfg["stress"]["noise_multiplier"]),
                           float(cfg["stress"]["extra_dropout_prob"]),
                           BODY_STRESS_SD, CABIN_STRESS_SD)
    t0_s, _, tracks_body_s, _ = run_tier0(engine, df_stress)
    notices_if_s, notices_ae_s = {}, {}
    for astro_id in test_astro_ids:
        sc = run_body_scoring(det1a, tracks_body_s[astro_id], cold_start=cold)
        notices_if_s[astro_id] = fire_notices(
            sc["strangeness_if"], sc["explanations"], det1a.high_thresh_if,
            consecutive=cons, cooldown=cool, source="tier1a_isolation_forest")
        notices_ae_s[astro_id] = fire_notices(
            sc["strangeness_ae"], sc["explanations"], det1a.high_thresh_ae,
            consecutive=cons, cooldown=cool, source="tier1a_autoencoder")
    det_table_s = body_detection_table(t0_s, notices_if_s, notices_ae_s, test["episodes"])
    fa_s = false_alarm_stats(t0_s, notices_if_s, notices_ae_s, clean_ids, n_days)
    dq_s = dq_integrity_check(t0_s, tracks_body_s)

    # sizes and latency
    sizes = {
        "tier1a_if+ae": {"kb": model_size_kb(det1a),
                         "note": "IsolationForest + 16-4-16 autoencoder + scaler"},
        "tier1b_if": {"kb": model_size_kb(det1b), "note": "IsolationForest, cabin"},
        "tier2a_logreg": {"kb": model_size_kb(clf2a), "params": clf2a.n_parameters,
                          "note": "multinomial logistic regression"},
        f"tier2b_{learned_fb.name}": {
            "kb": model_size_kb(learned_fb),
            "params": learned_fb.n_parameters() if hasattr(learned_fb, "n_parameters") else "-",
            "note": "learned forecaster" + ("" if learned_fb.name == "gru"
                                            else " (PyTorch unavailable fallback)")},
    }
    X_demo, _, _ = body_feature_matrix(tracks_body_test[test_astro_ids[0]])
    lat_1a = time_ms(lambda: det1a.strangeness(X_demo)) / len(X_demo)
    g0 = df_by_astro[test_astro_ids[0]]
    hist0 = g0[list(FORECAST_SIGNALS)].to_numpy()[:W]
    lat_2b = time_ms(lambda: learned_fb.forecast(hist0))

    det_summaries = {pt: detection_summary(g, ["tier0", "tier1a_if", "tier1a_ae"])
                     for pt, g in det_table.groupby("problem")}
    det_summaries_s = {pt: detection_summary(g, ["tier0", "tier1a_if", "tier1a_ae"])
                       for pt, g in det_table_s.groupby("problem")}

    metrics = {
        "seed": seed,
        "runtime_s": time.time() - t_start,
        "n_train_astronauts": n_train, "n_test_astronauts": n_test,
        "n_days": n_days,
        "body_det_table": det_table, "body_det_table_stress": det_table_s,
        "station_det_table": st_table,
        "det_summaries": det_summaries, "det_summaries_stress": det_summaries_s,
        "false_alarms": fa, "false_alarms_stress": fa_s,
        "station_false_alarms": fa_st,
        "precision": prec, "dq_integrity": dq, "dq_integrity_stress": dq_s,
        "storm_context": storm,
        "sensor_message_count": len(msgs_train) + len(msgs_test),
        "tier2a": {"macro_f1": macro_f1,
                   "confusion_md": confusion_md(cm, BANDS),
                   "infection_days_total": int(n_inf_days),
                   "infection_days_called_established": int(n_inf_est)},
        "tier2b": {"learned_name": learned_fb.name,
                   "learned_params": (learned_fb.n_parameters()
                                      if hasattr(learned_fb, "n_parameters") else "-"),
                   "baseline_mae": mae_b, "learned_mae": mae_l,
                   "baseline_mae14_mean": mean14(mae_b),
                   "learned_mae14_mean": mean14(mae_l),
                   "learned_beats_baseline": verdict,
                   "days_to_limit": d2l},
        "sizes": sizes,
        "latency": {"tier1a_ms_per_day": lat_1a, "tier2b_ms_per_forecast": lat_2b},
    }

    # ---------------- 7. report + charts ----------------
    print("[7/7] charts and report ...", flush=True)
    from aura import report as rp

    # showcase: a deconditioning astronaut detected by both Tier 0 and 1a
    decond_eps = test["episodes"][test["episodes"]["problem"] == "deconditioning"]
    show_astro, show_ep = None, None
    for _, ep in decond_eps.iterrows():
        det = det_table[(det_table["astro_id"] == ep["astro_id"])]
        if len(det) and det["tier0"].notna().any() and det["tier1a_if"].notna().any():
            show_astro, show_ep = ep["astro_id"], ep
            break
    if show_astro is None and len(decond_eps):
        show_ep = decond_eps.iloc[0]
        show_astro = show_ep["astro_id"]

    if show_astro is not None:
        eps_a = test["episodes"][test["episodes"]["astro_id"] == show_astro]
        alerts_a = t0_test[(t0_test["scope"] == "body") & (t0_test["astro_id"] == show_astro)]
        rp.fig_example_timeline(results_dir / "fig_example_timeline.png",
                                df_by_astro[show_astro], tracks_body_test[show_astro],
                                alerts_a, eps_a, show_astro)

        sc = scores_test[show_astro]
        pairs = [(sc["strangeness_if"], [n.day for n in notices_if[show_astro]])]
        aids, flags = [show_astro], [True]
        if clean_ids:
            ca = clean_ids[0]
            sca = scores_test[ca]
            pairs.append((sca["strangeness_if"], [n.day for n in notices_if[ca]]))
            aids.append(ca)
            flags.append(False)
        rp.fig_notices(results_dir / "fig_notices.png", pairs, det1a.high_thresh_if,
                       aids, flags)

        snap = int(show_ep["onset_day"]) + 21
        if snap + H < n_days and snap >= W:
            g = df_by_astro[show_astro]
            tracks_show = tracks_body_test[show_astro]
            hist = np.column_stack([tracks_show[s].ewma
                                    for s in FORECAST_SIGNALS])[snap - W:snap]
            realized = g[list(FORECAST_SIGNALS)].to_numpy()[snap:snap + H]
            realized_mask = np.column_stack([tracks_show[s].ok
                                             for s in FORECAST_SIGNALS])[snap:snap + H]
            hist_vals = np.column_stack([tracks_show[s].ewma
                                         for s in FORECAST_SIGNALS])[snap - W:snap]
            mu, band = learned_fb.forecast(hist)
            rp.fig_forecast(results_dir / "fig_forecast.png",
                            np.arange(snap - W, snap), hist_vals, mu, band, realized,
                            realized_mask, 1, "hrv_rmssd", "ms", snap, H)

    co2_eps = test["episodes"][test["episodes"]["problem"] == "co2_scrubber_fault"]
    if len(co2_eps):
        ep = co2_eps.iloc[0]
        sid = ep["station_id"]
        tr = tracks_st_test[sid]["co2_ppm"]
        rp.fig_cabin(results_dir / "fig_cabin.png", sid, tr.raw, tr.z,
                     t0_test[(t0_test["scope"] == "station") & (t0_test["station_id"] == sid)
                             & (t0_test["signal"] == "co2_ppm")],
                     context_labels.get(sid, []), ep, det1b.high_thresh)

    rp.fig_summary(results_dir / "fig_summary.png", det_summaries,
                   {"normal": fa, "stress": fa_s},
                   ["tier0", "tier1a_if", "tier1a_ae"],
                   sorted(det_table["problem"].unique().tolist()))

    rp.write_report(results_dir / "report.md", metrics, __version__)

    serializable = {k: v for k, v in metrics.items() if not isinstance(v, pd.DataFrame)}
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)

    snapshots = build_web_snapshots(
        astro_ids=test_astro_ids,
        current_day=n_days - 1,
        alerts=t0_test,
        tracks_by_astro=tracks_body_test,
        notices_if=notices_if,
        notices_ae=notices_ae,
        station_notices=station_notices,
        context_labels=context_labels,
        risk_table=tab_te,
        forecaster=learned_fb,
        input_window=W,
        horizon=H,
    )
    with open(results_dir / "aura_snapshots.json", "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2, allow_nan=False)

    print(f"\nDone in {time.time() - t_start:.0f}s. Results in {results_dir}/")
    print(f"  Tier 2b: learned ({learned_fb.name}) beats baseline: {verdict}")
    print(f"  DQ assertion (normal): {dq['assertion_passed']}, "
          f"(stress): {dq_s['assertion_passed']}")


if __name__ == "__main__":
    main()
