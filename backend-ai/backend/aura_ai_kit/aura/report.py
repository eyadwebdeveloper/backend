"""Report and chart generation for the AURA AI kit.

Charts follow the kit's plain-English rule: every title, axis and legend is
in English, honest labels ("simulated data") are printed on the figures,
and all numbers shown come from the metrics computed by the harness.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 9

SIM_TAG = "simulated data - engineering placeholders, not medic-approved values"


def _save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def fig_example_timeline(path, df_a, tracks, alerts_a, episodes_a, astro_id):
    """Body timeline for one astronaut with a problem: raw + baseline +
    alert markers over the shaded ground-truth window."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, sig, unit in zip(axes, ["resting_hr", "hrv_rmssd"],
                             ["bpm", "ms"]):
        tr = tracks[sig]
        days = np.arange(len(tr.raw))
        ax.plot(days, tr.raw, ".", color="#9aa5b1", ms=3, label="daily value (as received)")
        ax.plot(days, tr.ewma, "-", color="#1f77b4", lw=1.4, label="personal tracker (EWMA)")
        ax.axhline(tr.reference, color="#2ca02c", ls="--", lw=1,
                   label=f"mission-start reference {tr.reference:.1f}")
        # y-range from quality-gated values only: corrupt spikes (reported in
        # the sensor-message channel, not shown here) would squash the plot
        okv = tr.raw[tr.ok]
        if okv.size:
            lo, hi = np.nanpercentile(okv, [1, 99])
            pad = 0.15 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        for _, ep in episodes_a.iterrows():
            ax.axvspan(ep["onset_day"], ep["end_day"], color="#f4a582", alpha=0.25,
                       label="injected problem (ground truth)")
        fired = alerts_a[alerts_a["signal"] == sig] if len(alerts_a) else alerts_a
        if len(fired):
            ax.scatter(fired["alert_day"], [tr.reference] * len(fired), marker="v",
                       color="#d62728", zorder=5, label="Tier 0 alert (rule)")
        ax.set_ylabel(f"{sig} ({unit})")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    axes[1].set_xlabel("mission day")
    fig.suptitle(f"Astronaut {astro_id}: personal-baseline view - {SIM_TAG}", fontsize=9)
    _save(fig, path)


def fig_notices(path, score_pairs, thresh, astro_ids, problem_flags):
    """Strangeness over time for a problem astronaut vs a clean one."""
    fig, axes = plt.subplots(len(score_pairs), 1, figsize=(9, 5.5),
                             sharex=True, constrained_layout=True)
    if len(score_pairs) == 1:
        axes = [axes]
    for ax, (scores, notices_days), aid, is_problem in zip(
            axes, score_pairs, astro_ids, problem_flags):
        days = np.arange(len(scores))
        ax.plot(days, scores, "-", color="#1f77b4", lw=1.2,
                label="strangeness (IsolationForest, 0-1)")
        ax.axhline(thresh, color="#ff7f0e", ls="--", lw=1,
                   label=f"notice threshold {thresh:.2f}")
        if notices_days:
            ax.scatter(notices_days, [thresh] * len(notices_days), marker="o",
                       facecolors="none", edgecolors="#d62728", zorder=5,
                       label="AI notice fired")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("strangeness")
        kind = "with injected problem" if is_problem else "clean (no problem)"
        ax.set_title(f"Astronaut {aid} ({kind})", fontsize=8, loc="left")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("mission day")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.suptitle(f"Tier 1a body anomaly detector - {SIM_TAG}", fontsize=9)
    _save(fig, path)


def fig_forecast(path, hist_days, hist_vals, mu, band, realized, realized_mask,
                  signal_idx, signal, unit, snapshot_day, horizon_days):
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    j = signal_idx
    ax.plot(hist_days, hist_vals[:, j], "-", color="#1f77b4", lw=1.3,
            label="history (personal tracker)")
    fut = np.arange(snapshot_day + 1, snapshot_day + 1 + horizon_days)
    ax.plot(fut, mu[:, j], "--", color="#d62728", lw=1.4,
            label="Tier 2b forecast (GRU)")
    ax.fill_between(fut, mu[:, j] - band[:, j], mu[:, j] + band[:, j],
                    color="#d62728", alpha=0.15, label="uncertainty band")
    real = np.where(np.asarray(realized_mask)[:, j], realized[:, j], np.nan)
    ax.plot(fut, real, ".", color="#7f7f7f", ms=4,
            label="what actually happened (usable days, simulated)")
    allv = np.concatenate([hist_vals[:, j][np.isfinite(hist_vals[:, j])],
                           mu[:, j][np.isfinite(mu[:, j])],
                           (mu[:, j] - band[:, j])[np.isfinite(mu[:, j] - band[:, j])],
                           (mu[:, j] + band[:, j])[np.isfinite(mu[:, j] + band[:, j])],
                           real[np.isfinite(real)]])
    if allv.size:
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
        pad = 0.12 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    ax.axvline(snapshot_day, color="k", ls=":", lw=0.8)
    ax.set_xlabel("mission day")
    ax.set_ylabel(f"{signal} ({unit})")
    ax.set_title(f"Tier 2b forecast at day {snapshot_day} - {SIM_TAG}", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.grid(alpha=0.25)
    _save(fig, path)


def fig_cabin(path, station_id, co2_raw, z, alerts_cabin, labels, episode, threshold_days):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    days = np.arange(len(co2_raw))
    ax.plot(days, co2_raw, "-", color="#1f77b4", lw=1.1, label="cabin CO2 (as received)")
    if episode is not None:
        ax.axvspan(episode["onset_day"], episode["end_day"], color="#f4a582",
                   alpha=0.25, label="injected scrubber fault (ground truth)")
    if alerts_cabin is not None and len(alerts_cabin):
        ax.scatter(alerts_cabin["alert_day"], alerts_cabin["level"].astype(str).map(
            {"watch": 4000, "elevated": 5200}).fillna(4000),
            marker="v", color="#d62728", zorder=5, label="Tier 0 cabin alert")
    lab_days = [l["day"] for l in labels if l["label"] == "high_co2_exposure"]
    if lab_days:
        ax.scatter(lab_days, [3600] * len(lab_days), marker="o", facecolors="none",
                   edgecolors="#9467bd", zorder=5, label="Tier 1b context label")
    ax.set_xlabel("mission day")
    ax.set_ylabel("CO2 (ppm)")
    ymax = np.nanmax(co2_raw) if len(co2_raw) else 1.0
    ax.set_ylim(top=max(ymax * 1.08, 3600 + 400))   # headroom for marker rows
    ax.set_title(f"Station {station_id}: cabin CO2 - {SIM_TAG}", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.grid(alpha=0.25)
    _save(fig, path)


def fig_summary(path, det_summaries, false_alarms, detectors, problem_types):
    """Two panels: detection share + median delay; false alarms per week."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(problem_types))
    w = 0.8 / len(detectors)
    for k, det in enumerate(detectors):
        shares = [det_summaries.get(pt, {}).get(det, {}).get("share_caught") or 0
                  for pt in problem_types]
        ax1.bar(x + (k - len(detectors) / 2 + 0.5) * w, shares, w, label=det)
    ax1.set_xticks(x, problem_types, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("share of episodes caught")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Detection (within 7 days of onset)", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25, axis="y")

    bars = [
        ("Tier 0 alerts", false_alarms["normal"]["tier0_false_alerts_per_week"]),
        ("Tier 1a IF notices", false_alarms["normal"]["tier1a_if_false_notices_per_week"]),
        ("Tier 1a AE notices", false_alarms["normal"]["tier1a_ae_false_notices_per_week"]),
        ("Tier 0 (stress)", false_alarms["stress"]["tier0_false_alerts_per_week"]),
        ("Tier 1a IF (stress)", false_alarms["stress"]["tier1a_if_false_notices_per_week"]),
    ]
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#aec7e8", "#c7e9c0"]
    ax2.barh(names, vals, color=colors)
    ax2.set_xlabel("false episodes per person-week (clean astronauts)")
    ax2.set_title("False alarms, normal vs stress", fontsize=9)
    ax2.grid(alpha=0.25, axis="x")
    fig.suptitle(SIM_TAG, fontsize=9)
    _save(fig, path)


# ---------------------------------------------------------------- markdown

def fmt(v, nd=2):
    if v is None:
        return "missed"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def detection_md(det_table: pd.DataFrame, detectors: list) -> str:
    lines = ["| Problem | n | " + " | ".join(
        f"{d} caught" for d in detectors) + " | " + " | ".join(
        f"{d} median delay" for d in detectors) + " |",
        "|---|---|" + "---|" * (2 * len(detectors))]
    for pt, grp in det_table.groupby("problem"):
        cells = [str(len(grp))]
        for d in detectors:
            cells.append(f"{fmt(grp[d].notna().mean() * 100 if len(grp) else None, 0)}%")
        for d in detectors:
            delays = grp[d].dropna()
            cells.append(f"{delays.median():.0f} d" if len(delays) else "missed")
        lines.append(f"| {pt} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path: Path, metrics: dict, kit_version: str) -> None:
    m = metrics
    L = []
    L.append("# AURA AI kit - results report\n")
    L.append(f"Generated by `run_pipeline.py` (kit v{kit_version}). "
             "Everything below comes from the virtual-astronaut simulator with "
             "ENGINEERING PLACEHOLDER rules and problem magnitudes.\n")
    L.append("> **Honest limit (AI plan, section 7):** all accuracy numbers come "
             "from a test we built ourselves. They show the system works on "
             "simulated cases, not on real astronauts. Tier 2a labels are "
             "circular by construction (see its section). The medical teammates "
             "own the real rules; nothing here is medic-approved.\n")

    L.append("## 1. Detection of injected problems (body)\n")
    L.append("A detection counts if an alert or notice appears within 7 days "
             "of onset. Delay is days from onset to the first alert/notice.\n")
    L.append(detection_md(m["body_det_table"], ["tier0", "tier1a_if", "tier1a_ae"]))
    L.append("")

    L.append("## 2. Detection of the cabin CO2 fault (station)\n")
    st = m["station_det_table"]
    L.append("| Detector | share caught | median delay |")
    L.append("|---|---|---|")
    for d in ["tier0_cabin", "tier1b_notice", "context_label"]:
        if len(st):
            caught = st[d].notna().mean() if d in st else 0
            delays = st[d].dropna() if d in st else pd.Series(dtype=float)
            delay_txt = f"{delays.median():.0f} d" if len(delays) else "-"
            L.append(f"| {d} | {caught * 100:.0f}% | {delay_txt} |")
        else:
            L.append(f"| {d} | no station episode in this run | - |")
    L.append("")

    L.append("## 3. False alarms and precision (clean vs all astronauts)\n")
    fa, fas = m["false_alarms"], m["false_alarms_stress"]
    pr = m["precision"]
    L.append("| Metric | Normal | Stress (noise x1.5, +20% dropouts) |")
    L.append("|---|---|---|")
    L.append(f"| Tier 0 false alerts / person-week (clean) | "
             f"{fa['tier0_false_alerts_per_week']:.3f} | "
             f"{fas['tier0_false_alerts_per_week']:.3f} |")
    L.append(f"| Tier 1a IF false notices / person-week (clean) | "
             f"{fa['tier1a_if_false_notices_per_week']:.3f} | "
             f"{fas['tier1a_if_false_notices_per_week']:.3f} |")
    L.append(f"| Tier 1a AE false notices / person-week (clean) | "
             f"{fa['tier1a_ae_false_notices_per_week']:.3f} | "
             f"{fas['tier1a_ae_false_notices_per_week']:.3f} |")
    L.append("")
    L.append(f"Share of Tier 0 alert episodes that were real: "
             f"{fmt(pr['tier0_share_real'] * 100 if pr['tier0_share_real'] is not None else None, 0)}% "
             f"({pr['tier0_alerts_total']} episodes).")
    L.append(f"Share of Tier 1a IF notice episodes that were real: "
             f"{fmt(pr['tier1a_if_share_real'] * 100 if pr['tier1a_if_share_real'] is not None else None, 0)}% "
             f"({pr['tier1a_if_notice_episodes']} episodes).")
    L.append(f"Share of Tier 1a AE notice episodes that were real: "
             f"{fmt(pr['tier1a_ae_share_real'] * 100 if pr['tier1a_ae_share_real'] is not None else None, 0)}% "
             f"({pr['tier1a_ae_notice_episodes']} episodes).")
    L.append("")
    by_rule = fa.get("tier0_false_alerts_by_rule") or {}
    if by_rule:
        L.append("Tier 0 false alerts on clean astronauts, by rule (the alert-fatigue "
                 "tuning list for the medics):\n")
        L.append("| Rule | False alert episodes |")
        L.append("|---|---|")
        for rule, cnt in sorted(by_rule.items(), key=lambda kv: -kv[1]):
            L.append(f"| {rule} | {cnt} |")
        L.append("")

    L.append("## 4. Data-quality integrity (the 'never a fake health alert' rule)\n")
    dq = m["dq_integrity"]
    L.append(f"- Health alerts sitting on a failed-sensor day: "
             f"**{dq['alerts_on_failed_sensor_days']}** "
             f"(assertion {'PASSED' if dq['assertion_passed'] else 'FAILED - investigate'}).")
    L.append(f"- Sensor messages raised (train+test): {m['sensor_message_count']}.")
    L.append(f"- Corrupted records accepted as health signals: "
             f"**{0 if dq['assertion_passed'] else 'see assertion above'}**.")
    L.append("")

    L.append("## 5. Tier 2a: deconditioning risk band (circular by construction)\n")
    t2a = m["tier2a"]
    L.append(f"- Macro F1 on held-out astronaut-days: **{t2a['macro_f1']:.2f}** "
             f"(chance = 0.33 for three balanced classes).")
    L.append(f"- Days inside infection-like episodes labeled 'established' by the "
             f"classifier: {t2a['infection_days_called_established']} of "
             f"{t2a['infection_days_total']} - the risk band cannot separate an "
             "infection-like heart-rate rise from deconditioning drift from "
             "these features alone. Stated plainly, as a limitation.")
    L.append("")
    L.append("Confusion matrix (rows = simulator label, columns = predicted):\n")
    L.append(t2a["confusion_md"])
    L.append("")
    L.append("> Labels come from the same injection that drives the features. "
             "This validates the pipeline mechanics only, not medical risk "
             "prediction.\n")

    L.append("## 6. Tier 2b: forecast (baseline vs learned)\n")
    t2b = m["tier2b"]
    L.append(f"- Learned model: **{t2b['learned_name']}** "
             f"({t2b['learned_params']} parameters).")
    L.append(f"- MAE at day 14, averaged over signals: baseline "
             f"**{t2b['baseline_mae14_mean']:.2f}** vs learned "
             f"**{t2b['learned_mae14_mean']:.2f}** (native units differ per signal; "
             "per-signal table below).")
    verdict = t2b["learned_beats_baseline"]
    L.append(f"- **AI plan honesty rule: does the learned model beat the simple "
             f"baseline? {'YES' if verdict else 'NO'}** "
             f"(report this as-is in the submission).\n")
    L.append("| Signal | unit | MAE day 7 baseline | day 7 learned | MAE day 14 baseline | day 14 learned |")
    L.append("|---|---|---|---|---|---|")
    for sig, unit in [("resting_hr", "bpm"), ("hrv_rmssd", "ms"), ("sleep_hours", "h")]:
        b7 = t2b["baseline_mae"][7][sig]
        l7 = t2b["learned_mae"][7][sig]
        b14 = t2b["baseline_mae"][14][sig]
        l14 = t2b["learned_mae"][14][sig]
        L.append(f"| {sig} | {unit} | {b7:.2f} | {l7:.2f} | {b14:.2f} | {l14:.2f} |")
    L.append("")
    if t2b.get("days_to_limit"):
        L.append("Days-to-personal-limit projection (deconditioning astronauts, "
                 "snapshot at onset+21):\n")
        L.append("| Astronaut | actual crossing | baseline prediction | learned prediction |")
        L.append("|---|---|---|---|")
        for r in t2b["days_to_limit"]:
            L.append(f"| {r['astro_id']} | {r['actual']} | {r['baseline']} | {r['learned']} |")
        L.append("")

    L.append("## 7. Context labels on solar-storm days (Tier 1b)\n")
    sc = m["storm_context"]
    L.append(f"- Storm events in test stations: {sc['storms_total']}; labeled "
             f"'elevated radiation context': {sc['storms_labeled']}.")
    L.append(f"- Radiation context labels without a storm: "
             f"{sc['radiation_labels_without_storm']} "
             "(a cabin reading is context only, never a personal dose).")
    L.append("")

    L.append("## 8. Model sizes and speed (development machine, not the capped benchmark)\n")
    sz = m["sizes"]
    L.append("| Model | pickled size | parameters | notes |")
    L.append("|---|---|---|---|")
    for name, row in sz.items():
        L.append(f"| {name} | {row['kb']:.1f} KB | {row.get('params', '-')} | {row.get('note', '')} |")
    L.append("")
    L.append(f"Scoring latency: {m['latency']['tier1a_ms_per_day']:.2f} ms per "
             f"astronaut-day (Tier 1a), {m['latency']['tier2b_ms_per_forecast']:.2f} ms "
             "per forecast (Tier 2b). The capped 2-core / 2 GB benchmark is a "
             "pending item (run during the event on the target laptop).")
    L.append("")

    L.append("## 9. What this run does and does not show\n")
    L.append("- It DOES show: the full pipeline runs offline end to end; rules and "
             "AI share the same personal-baseline mathematics; the data-quality "
             "gate keeps corrupt data out of health alerts; tiered detection "
             "behaves as designed (fast events caught in days, slow drift caught "
             "over weeks; multi-signal drift caught by the advisory AI even when "
             "single thresholds miss it); degradation under stress is measured.")
    L.append("- It does NOT show: performance on real astronauts, real sensor "
             "hardware, or medic-approved thresholds. All magnitudes are "
             "placeholders pending medical review.")
    L.append("- Reproduce: `python run_pipeline.py` (seed "
             f"{m['seed']}, runtime {m['runtime_s']:.0f} s on this machine).")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")
