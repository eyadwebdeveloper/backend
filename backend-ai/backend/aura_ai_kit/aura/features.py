"""Shared personal-baseline mathematics (used by Tier 0 rules and AI features).

Pipeline per person (or station) and per signal:

    raw -> range check (dq_range)            out-of-range = corrupt, excluded
        -> stuck detection (rolling std)     frozen values = faulty, excluded
        -> 3-day trailing median             kills odd single readings
        -> EWMA fast tracker                 weights recent days more
        -> z vs mission-start reference      in robust-sigma units
        -> trailing 14-day-mean z            the slow-drift view

The mission-start reference exists so a slow decline is not silently
absorbed into the baseline (AI plan: "keep a long-term reference").
All comparisons are against the person's own history, never a population.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def robust_sd(x) -> float:
    """1.4826 x MAD; falls back to std when MAD collapses; NaN if unusable."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return 1.4826 * mad
    sd = float(np.std(x))
    return sd if sd > 1e-12 else float("nan")


def ewma_nan(values: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA that skips NaN days (state carried forward; NaN emitted those days)."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    s = None
    for i, v in enumerate(values):
        if np.isfinite(v):
            s = float(v) if s is None else alpha * float(v) + (1.0 - alpha) * s
            out[i] = s
    return out


def median3_nan(values: np.ndarray) -> np.ndarray:
    """Causal 3-day trailing median (removes single odd readings)."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(3, min_periods=1).median().to_numpy()


def trailing_mean(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(window, min_periods=min_periods).mean().to_numpy()


def trailing_ols_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Causal per-day OLS slope of the last `window` finite values (units/day)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for t in range(n - 1, -1, -1):
        lo = max(0, t - window + 1)
        seg = values[lo:t + 1]
        mask = np.isfinite(seg)
        k = int(mask.sum())
        if k >= max(3, window // 2):
            x = np.arange(lo, t + 1)[mask] - t  # days relative to t
            y = seg[mask]
            xm, ym = x.mean(), y.mean()
            denom = ((x - xm) ** 2).sum()
            if denom > 0:
                out[t] = float(((x - xm) * (y - ym)).sum() / denom)
    return out


@dataclass
class SignalTrack:
    """Baseline state for one signal of one person or station.

    After `process()`:
      ok        bool array - True where the day's value is usable
      z         EWMA z-score vs mission-start reference (fast view)
      z_slow    trailing 14-day-mean z-score (slow-drift view)
      reference mission-start reference (median of first clean window)
      sigma     robust sd of the reference window
      sensor_events  list of dicts for the app's sensor-message channel
    """

    name: str
    raw: np.ndarray
    dq_range: tuple | None = None
    settings: dict = field(default_factory=dict)

    ok: np.ndarray = None
    ewma: np.ndarray = None
    z: np.ndarray = None
    z_slow: np.ndarray = None
    reference: float = float("nan")
    sigma: float = float("nan")
    sensor_events: list = field(default_factory=list)

    def process(self) -> "SignalTrack":
        raw = np.asarray(self.raw, dtype=float)
        n = len(raw)
        finite = np.isfinite(raw)

        # 1. range check
        if self.dq_range is not None:
            lo, hi = float(self.dq_range[0]), float(self.dq_range[1])
            in_range = finite & (raw >= lo) & (raw <= hi)
        else:
            in_range = finite

        # 2. stuck detection. The tolerance sigma is estimated from the
        #    reference window only: a whole-history estimate would be inflated
        #    by any regime change and would then flag normal quiet windows as
        #    frozen. A sensor also has to stay frozen for >= 3 consecutive
        #    days before it is declared stuck (transient quiet windows happen).
        ref_win = int(self.settings.get("reference_window_days", 30))
        seed_vals = raw[in_range][:ref_win]
        if seed_vals.size < 10:
            seed_vals = raw[in_range][:80]
        sd0 = robust_sd(seed_vals)
        ser = pd.Series(np.where(in_range, raw, np.nan))
        fw = int(self.settings.get("flatline_window_days", 5))
        roll_std = ser.rolling(fw, min_periods=3).std().to_numpy()
        tol = self.flatline_tol * (sd0 if np.isfinite(sd0) else 0.0)
        frozen = in_range & np.isfinite(roll_std) & (roll_std < max(tol, 1e-12))
        stuck = np.zeros(n, dtype=bool)
        t = 0
        while t < n:
            if frozen[t]:
                s0 = t
                while t < n and frozen[t]:
                    t += 1
                if t - s0 >= 3:                 # persistent freeze only
                    stuck[s0:t] = True
            else:
                t += 1

        ok = in_range & ~stuck

        # 3. sensor events: out-of-range days, stuck runs, long unusable runs
        ev = []
        for t in np.nonzero(finite & ~in_range)[0]:
            ev.append({"day": int(t), "kind": "out_of_range",
                       "detail": f"{self.name}: value {raw[t]:.1f} is outside the plausible range"
                                 f" ({self.dq_range[0]} to {self.dq_range[1]}); ignored"})
        # stuck runs
        t = 0
        while t < n:
            if stuck[t]:
                s0 = t
                while t < n and stuck[t]:
                    t += 1
                ev.append({"day": s0, "kind": "stuck_sensor",
                           "detail": f"{self.name}: value frozen around {raw[s0]:.1f} for {t - s0} days; ignored"})
            else:
                t += 1
        # long unusable runs (missing + rejected)
        gap_max = int(self.settings.get("gap_max_days", 2))
        t = 0
        while t < n:
            if not ok[t]:
                s0 = t
                while t < n and not ok[t]:
                    t += 1
                if t - s0 > gap_max:
                    ev.append({"day": s0, "kind": "no_data",
                               "detail": f"{self.name}: no usable data for {t - s0} days"})
                if t < n:
                    ev.append({"day": t, "kind": "recovered",
                               "detail": f"{self.name}: data recovered"})
            else:
                t += 1

        # 4. baseline chain on usable values only
        clean = np.where(ok, raw, np.nan)
        smoothed = median3_nan(clean)
        alpha = float(self.settings.get("ewma_alpha", 0.2))
        self.ewma = ewma_nan(smoothed, alpha)

        ref_win = int(self.settings.get("reference_window_days", 30))
        min_ref = int(self.settings.get("min_reference_samples", 15))
        first_ok = raw[ok][:ref_win]
        if first_ok.size >= min_ref:
            self.reference = float(np.median(first_ok))
            self.sigma = robust_sd(first_ok)
            if not np.isfinite(self.sigma):
                # extremely quiet signal: fall back to a tiny epsilon so z stays finite
                self.sigma = max(float(np.std(first_ok)), 1e-6)
        else:
            self.reference = float("nan")
            self.sigma = float("nan")

        with np.errstate(invalid="ignore", divide="ignore"):
            self.z = (self.ewma - self.reference) / self.sigma
            m14 = trailing_mean(clean, 14, 10)
            self.z_slow = (m14 - self.reference) / self.sigma

        self.ok = ok
        self.raw = raw
        self.sensor_events = ev
        return self

    @property
    def flatline_tol(self) -> float:
        return float(self.settings.get("flatline_tol_frac", 0.15))

    def first_valid_day(self) -> int:
        z = self.z
        idx = np.nonzero(np.isfinite(z))[0]
        return int(idx[0]) if idx.size else -1


def make_tracks(signals: dict, rules: dict, settings: dict) -> dict:
    """Build and process a SignalTrack per signal defined in the rules."""
    tracks = {}
    for name, cfg in rules.items():
        if name not in signals:
            continue
        tracks[name] = SignalTrack(
            name=name,
            raw=np.asarray(signals[name], dtype=float),
            dq_range=cfg.get("dq_range"),
            settings=settings,
        ).process()
    return tracks
