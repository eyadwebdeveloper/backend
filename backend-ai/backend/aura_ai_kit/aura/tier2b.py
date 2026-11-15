"""Tier 2b: forecaster (ADVISORY ONLY).

Answers "where is this heading?" for a few daily values (resting heart
rate, HRV, sleep): the last 30 days in, the next 14 days out, with an
uncertainty band.

Two steps, per the AI plan, so we are not fooled:
  1. TrendBaseline: a straight-line trend plus smoothing (the honest
     baseline every learned model must beat)
  2. a learned sequence model: a small GRU when PyTorch is available,
     otherwise a ridge regression on the flattened window (stated plainly
     in the report whenever the fallback runs)

The AI plan's honesty rule is enforced in code: if the learned model does
not beat the simple baseline on held-out astronauts, the report says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ewma_nan, robust_sd

try:  # PyTorch is optional; the kit falls back to ridge without it
    import torch
    from torch import nn
    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False

FORECAST_SIGNALS = ("resting_hr", "hrv_rmssd", "sleep_hours")


def _preprocess(hist: np.ndarray) -> np.ndarray:
    """Forward-fill a (T, S) window. Inputs are the DQ-gated, already-smoothed
    tracker series from aura.features.SignalTrack, so no extra smoothing is
    applied here. Leading NaNs (window starts before the tracker exists) are
    filled with the training median after normalization (see fit)."""
    return pd.DataFrame(np.asarray(hist, dtype=float)).ffill().to_numpy()


class TrendBaseline:
    """Straight-line trend (OLS over the last W forward-filled values) with a
    band from fit residuals. Step 1 of the AI plan. The regression runs on
    the filled raw values, not on an EWMA, because an EWMA pre-filter lags a
    ramp systematically and would bias the projected level."""

    name = "trend_baseline"

    def __init__(self, input_window: int = 30, horizon: int = 14):
        self.W = input_window
        self.H = horizon

    def forecast(self, hist: np.ndarray):
        """hist: (T, S) native units, NaN allowed. Returns (mu (H, S), band (H, S))."""
        vals = pd.DataFrame(np.asarray(hist, dtype=float)).ffill().to_numpy()
        T, S = vals.shape
        W = min(self.W, T)
        seg = vals[-W:]
        x = np.arange(W, dtype=float)
        mu = np.full((self.H, S), np.nan)
        band = np.full((self.H, S), np.nan)
        for s in range(S):
            y = seg[:, s]
            m = np.isfinite(y)
            if m.sum() < W * 0.6:
                continue
            xm, ym = x[m].mean(), y[m].mean()
            denom = ((x[m] - xm) ** 2).sum()
            b = ((x[m] - xm) * (y[m] - ym)).sum() / max(denom, 1e-12)
            a = ym - b * xm
            fut = np.arange(W + 1, W + 1 + self.H, dtype=float)
            mu[:, s] = a + b * fut
            band[:, s] = 1.96 * float(np.std(y[m] - (a + b * x[m])))
        return mu, band


class _GRUNet(nn.Module if TORCH_OK else object):
    def __init__(self, n_in: int, hidden: int, horizon: int, n_out: int):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * n_out)
        self.H, self.S = horizon, n_out

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).view(-1, self.H, self.S)


class GRUForecaster:
    """Small GRU over normalized 30-day windows. Requires PyTorch.

    Trains only on windows whose 14-day future is fully realized and clean
    (mask all True): a NaN target would poison the MSE loss and corrupt the
    whole network. Evaluation may use partial futures via the mask."""

    name = "gru"

    def __init__(self, input_window: int = 30, horizon: int = 14,
                 hidden: int = 32, seed: int = 0):
        self.W, self.H, self.hidden, self.seed = input_window, horizon, hidden, seed

    def fit(self, windows: np.ndarray, targets: np.ndarray,
            masks: np.ndarray) -> "GRUForecaster":
        torch.manual_seed(self.seed)
        windows = np.asarray(windows, dtype=float)
        targets = np.asarray(targets, dtype=float)
        masks = np.asarray(masks, dtype=bool)
        complete = masks.all(axis=(1, 2))
        windows, targets = windows[complete], targets[complete]
        N, W, S = windows.shape

        flat = windows.reshape(-1, S)
        self.center = np.nanmedian(flat, axis=0)
        self.scale = np.array([max(robust_sd(flat[:, s]), 1e-6) for s in range(S)])

        Xall = np.nan_to_num(self._norm(self._preprocess_windows(windows)), nan=0.0)
        Yall = (targets - self.center) / self.scale

        n_val = max(int(N * 0.15), 1)
        idx = np.random.default_rng(self.seed).permutation(N)
        vi, ti = idx[:n_val], idx[n_val:]
        X, Y = Xall[ti], Yall[ti]
        Xv, Yv = Xall[vi], Yall[vi]

        net = _GRUNet(S, self.hidden, self.H, S)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        Xvt = torch.tensor(Xv, dtype=torch.float32)
        Yvt = torch.tensor(Yv, dtype=torch.float32)

        best_val, best_state, patience = np.inf, None, 0
        bs = 256
        rng = np.random.default_rng(self.seed)
        for epoch in range(300):
            net.train()
            for start in range(0, len(Xt), bs):
                sel = rng.permutation(len(Xt))[:bs] if len(Xt) > bs else np.arange(len(Xt))
                opt.zero_grad()
                loss = lossf(net(Xt[torch.tensor(sel)]), Yt[torch.tensor(sel)])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(lossf(net(Xvt), Yvt))
            if val < best_val - 1e-5:
                best_val, best_state, patience = val, net.state_dict(), 0
            else:
                patience += 1
                if patience >= 25:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net

        # uncertainty band from validation residuals, per horizon and signal
        net.eval()
        with torch.no_grad():
            pv = self._unnorm(net(Xvt).numpy())
        resid = np.abs(pv - targets[vi])          # native units
        self.resid_sd = np.nanmedian(resid, axis=0)
        fallback = np.nanmedian(resid) if np.isfinite(np.nanmedian(resid)) else 1.0
        self.resid_sd = np.where(np.isfinite(self.resid_sd), self.resid_sd, fallback)
        return self

    def _norm(self, w: np.ndarray) -> np.ndarray:
        return (w - self.center) / self.scale

    def _unnorm(self, w: np.ndarray) -> np.ndarray:
        return w * self.scale + self.center

    def _preprocess_windows(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([_preprocess(w) for w in windows])

    def forecast(self, hist: np.ndarray):
        x = _preprocess(hist)[-self.W:]
        xn = torch.tensor(np.nan_to_num(((x - self.center) / self.scale), nan=0.0)[None, ...],
                          dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            mu = self._unnorm(self.net(xn).numpy())[0]
        band = 1.96 * self.resid_sd
        return mu, np.broadcast_to(band, mu.shape)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


class RidgeForecaster:
    """Fallback learned model when PyTorch is unavailable (stated in the report).
    Same NaN discipline as the GRU: train on fully-realized windows only."""

    name = "ridge_fallback"

    def __init__(self, input_window: int = 30, horizon: int = 14, seed: int = 0):
        self.W, self.H, self.seed = input_window, horizon, seed

    def fit(self, windows, targets, masks=None):
        from sklearn.linear_model import Ridge
        windows = np.asarray(windows, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if masks is not None:
            complete = np.asarray(masks, dtype=bool).all(axis=(1, 2))
            windows, targets = windows[complete], targets[complete]
        N, W, S = windows.shape
        flat = windows.reshape(-1, S)
        self.center = np.nanmedian(flat, axis=0)
        self.scale = np.array([max(robust_sd(flat[:, s]), 1e-6) for s in range(S)])
        X = np.nan_to_num(self._norm(self._preprocess_windows(windows)).reshape(N, -1), nan=0.0)
        Y = ((targets - self.center) / self.scale).reshape(N, -1)
        n_val = max(int(N * 0.15), 1)
        idx = np.random.default_rng(self.seed).permutation(N)
        vi, ti = idx[:n_val], idx[n_val:]
        self.model = Ridge(alpha=3.0).fit(X[ti], Y[ti])
        pv = self._unnorm(self.model.predict(X[vi]).reshape(-1, self.H, S))
        resid = np.abs(pv - targets[vi])
        self.resid_sd = np.nanmedian(resid, axis=0)
        fallback = np.nanmedian(resid) if np.isfinite(np.nanmedian(resid)) else 1.0
        self.resid_sd = np.where(np.isfinite(self.resid_sd), self.resid_sd, fallback)
        return self

    def _norm(self, w):
        return (w - self.center) / self.scale

    def _unnorm(self, w):
        return w * self.scale + self.center

    def _preprocess_windows(self, windows):
        return np.stack([_preprocess(w) for w in windows])

    def forecast(self, hist):
        x = _preprocess(hist)[-self.W:]
        X = np.nan_to_num(self._norm(x)[None, ...].reshape(1, -1), nan=0.0)
        mu = self._unnorm(self.model.predict(X).reshape(self.H, -1))
        band = 1.96 * self.resid_sd
        return mu, np.broadcast_to(band, mu.shape)

    def n_parameters(self) -> int:
        return int(self.model.coef_.size + self.model.intercept_.size)


def make_learned_forecaster(backend: str = "auto", **kw):
    if backend == "auto":
        return GRUForecaster(**kw) if TORCH_OK else RidgeForecaster(**kw)
    if backend == "gru":
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available; use backend='ridge' or 'auto'")
        return GRUForecaster(**kw)
    return RidgeForecaster(**kw)


# ---------------- windows and evaluation helpers ----------------

def build_windows(df: pd.DataFrame, signals=FORECAST_SIGNALS, W: int = 30, H: int = 14,
                  stride: int = 1, min_input_frac: float = 0.8, min_target: int = 10):
    """Sliding windows from a long-format df (kept for ad-hoc use; the pipeline
    uses build_windows_from_tracks, which respects the data-quality gate)."""
    windows, targets, masks, meta = [], [], [], []
    for astro_id, g in df.groupby("astro_id", sort=True):
        arr = g[list(signals)].to_numpy(dtype=float)
        T = len(arr)
        for start in range(0, T - W - H + 1, stride):
            inp, tgt = arr[start:start + W], arr[start + W:start + W + H]
            if np.isfinite(inp).mean() < min_input_frac:
                continue
            msk = np.isfinite(tgt)
            if msk.sum() < min_target:
                continue
            windows.append(inp)
            targets.append(tgt)
            masks.append(msk)
            meta.append((astro_id, start))
    return (np.array(windows), np.array(targets), np.array(masks), meta)


def build_windows_from_tracks(tracks_by_astro: dict, signals=FORECAST_SIGNALS,
                              W: int = 30, H: int = 14, stride: int = 1,
                              min_input_frac: float = 0.8, min_target: int = 10):
    """Sliding windows built from the DQ-gated tracker series.

    Input  = SignalTrack.ewma (range-checked, median-filtered, EWMA-tracked):
             corrupt spikes and frozen values never reach the forecaster.
    Target = the raw value on days the signal passed the quality gate,
             NaN otherwise; MAE is computed only on usable days (mask).
    """
    windows, targets, masks, meta = [], [], [], []
    for astro_id, tracks in tracks_by_astro.items():
        inp = np.column_stack([tracks[s].ewma for s in signals])
        raw = np.column_stack([tracks[s].raw for s in signals])
        okm = np.column_stack([tracks[s].ok for s in signals])
        T = inp.shape[0]
        for start in range(0, T - W - H + 1, stride):
            w = inp[start:start + W]
            if np.isfinite(w).mean() < min_input_frac:
                continue
            tgt = raw[start + W:start + W + H]
            msk = okm[start + W:start + W + H]
            if msk.sum() < min_target:
                continue
            windows.append(w)
            targets.append(np.where(msk, tgt, np.nan))
            masks.append(msk)
            meta.append((astro_id, start))
    return (np.array(windows), np.array(targets), np.array(masks), meta)


def mae_by_horizon(mu: np.ndarray, target: np.ndarray, mask: np.ndarray,
                   horizons=(7, 14)) -> dict:
    """Mean absolute error per signal at 1-indexed horizons, native units."""
    out = {}
    for h in horizons:
        errs = np.abs(mu[:, h - 1, :] - target[:, h - 1, :])
        m = mask[:, h - 1, :]
        per_sig = {}
        for j, sig in enumerate(FORECAST_SIGNALS):
            vals = errs[:, j][m[:, j]]
            per_sig[sig] = float(np.mean(vals)) if vals.size else float("nan")
        out[h] = per_sig
    return out


def crossing_day(z_series: np.ndarray, threshold: float, direction: str,
                 start: int = 0, limit: int | None = None) -> int | None:
    """First day at/after `start` where the z series crosses the threshold."""
    end = len(z_series) if limit is None else min(len(z_series), start + limit)
    for t in range(start, end):
        v = z_series[t]
        if not np.isfinite(v):
            continue
        if direction == "upper" and v >= threshold:
            return t
        if direction == "lower" and v <= -threshold:
            return t
    return None
