"""Build JSON snapshots for the local web showcase from pipeline outputs."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .tier2b import FORECAST_SIGNALS

CREW_TO_ASTRONAUT = {
    "cmdr": "A101",
    "med": "A102",
    "eng": "A108",
    "sci": "A106",
}

SIGNAL_UNITS = {
    "resting_hr": "bpm",
    "hrv_rmssd": "ms",
    "sleep_hours": "h",
}


def _finite_list(values: np.ndarray) -> list[float]:
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("A web snapshot contains a non-finite forecast value")
    return [float(value) for value in result]


def _history_values(values: np.ndarray) -> list[float]:
    series = pd.Series(np.asarray(values, dtype=float))
    clean = series.interpolate(limit_direction="both").ffill().bfill()
    if clean.isna().any():
        raise ValueError("A web snapshot has no usable forecast history")
    return [float(value) for value in clean.to_numpy()]


def _validate_finite_json(value, path: str = "snapshot") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"Non-finite value in web snapshot at {path}")


def _risk_drivers(row: pd.Series) -> list[str]:
    drivers = []
    for feature, label, direction in (
        ("hr_z14", "14-day mean resting heart rate", "above"),
        ("hrv_z14", "14-day mean heart rate variability", "below"),
        ("sleep_z14", "14-day mean sleep duration", "below"),
        ("active_z14", "14-day mean activity", "below"),
    ):
        value = float(row[feature])
        if (direction == "above" and value >= 0.5) or (
                direction == "below" and value <= -0.5):
            drivers.append(
                f"{label} is {direction} the personal reference "
                f"({value:+.1f} robust SD).")
    if not drivers:
        drivers.append(
            "The advisory band is based on simulated personal-baseline features.")
    return drivers


def build_web_snapshots(
    *,
    astro_ids: list[str],
    current_day: int,
    alerts: pd.DataFrame,
    tracks_by_astro: dict,
    notices_if: dict,
    notices_ae: dict,
    station_notices: dict,
    context_labels: dict,
    risk_table: pd.DataFrame,
    forecaster,
    input_window: int,
    horizon: int,
) -> dict[str, dict]:
    """Create one current simulated snapshot for each held-out astronaut."""
    generated_at = datetime.now(timezone.utc).isoformat()
    output = {}
    recent_from = max(0, current_day - 30)

    for astro_id in astro_ids:
        astronaut_tracks = tracks_by_astro[astro_id]
        station_id = astronaut_tracks["__station__"]

        relevant = alerts[
            (
                (alerts["scope"] == "body")
                & (alerts["astro_id"] == astro_id)
            )
            | (
                (alerts["scope"] == "station")
                & (alerts["station_id"] == station_id)
            )
        ]
        relevant = relevant[
            (relevant["alert_day"] <= current_day)
            & (relevant["end_day"] >= recent_from)
        ]
        tier0_alerts = []
        for item in relevant.to_dict("records"):
            tier0_alerts.append({
                "scope": str(item["scope"]),
                "astro_id": (
                    None if pd.isna(item["astro_id"])
                    else str(item["astro_id"])
                ),
                "station_id": str(item["station_id"]),
                "signal": str(item["signal"]),
                "rule": str(item["rule"]),
                "level": str(item["level"]),
                "alert_day": int(item["alert_day"]),
                "end_day": int(item["end_day"]),
                "explanation": str(item["explanation"]),
            })

        tier1_notices = []
        for notice in [*notices_if.get(astro_id, []),
                       *notices_ae.get(astro_id, []),
                       *station_notices.get(station_id, [])]:
            if recent_from <= notice.day <= current_day:
                tier1_notices.append(notice.as_dict())
        for context in context_labels.get(station_id, []):
            if recent_from <= int(context["day"]) <= current_day:
                tier1_notices.append({
                    "day": int(context["day"]),
                    "source": "tier1b_context",
                    "strangeness": 1.0,
                    "explanation": str(context["text"]),
                })
        tier1_notices.sort(key=lambda item: (item["day"], item["source"]))

        risk = None
        candidate_rows = risk_table[risk_table["astro_id"] == astro_id]
        if not candidate_rows.empty:
            row = candidate_rows.sort_values("day").iloc[-1]
            risk = {
                "band": str(row["pred"]),
                "confidence": float(row["confidence"]),
                "drivers": _risk_drivers(row),
            }

        history = np.column_stack([
            astronaut_tracks[signal].ewma for signal in FORECAST_SIGNALS
        ])[-input_window:]
        forecast_mean, forecast_band = forecaster.forecast(history)
        series = []
        for index, signal in enumerate(FORECAST_SIGNALS):
            series.append({
                "signal": signal,
                "unit": SIGNAL_UNITS[signal],
                "history": _history_values(history[:, index]),
                "mu": _finite_list(forecast_mean[:, index]),
                "band": _finite_list(forecast_band[:, index]),
            })

        output[astro_id] = {
            "astro_id": astro_id,
            "day": int(current_day),
            "generated_at": generated_at,
            "simulated": True,
            "tier0": {"alerts": tier0_alerts},
            "tier1": {"notices": tier1_notices},
            "tier2": {
                "risk": risk,
                "forecast": {
                    "horizon_days": int(horizon),
                    "backend": (
                        "ridge" if forecaster.name == "ridge_fallback"
                        else forecaster.name
                    ),
                    "series": series,
                },
            },
        }
        _validate_finite_json(output[astro_id], f"snapshot.{astro_id}")
    return output
