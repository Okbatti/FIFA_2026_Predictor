from __future__ import annotations
import numpy as np

CLASSES = ["home", "draw", "away"]


def log_loss(preds: list[dict], actuals: list[str], eps: float = 1e-12) -> float:
    tot = 0.0
    for p, y in zip(preds, actuals):
        tot += -np.log(min(max(p[y], eps), 1.0))
    return tot / len(preds)


def brier(preds: list[dict], actuals: list[str]) -> float:
    tot = 0.0
    for p, y in zip(preds, actuals):
        for c in CLASSES:
            tot += (p[c] - (1.0 if c == y else 0.0)) ** 2
    return tot / len(preds)


def calibration_bins(preds: list[dict], actuals: list[str], n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        pred_p, obs = [], []
        for p, y in zip(preds, actuals):
            ph = p["home"]
            if (lo <= ph < hi) or (b == n_bins - 1 and ph == hi):
                pred_p.append(ph)
                obs.append(1.0 if y == "home" else 0.0)
        out.append({
            "lo": lo,
            "hi": hi,
            "mean_pred": float(np.mean(pred_p)) if pred_p else None,
            "obs_freq": float(np.mean(obs)) if obs else None,
            "n": len(pred_p),
        })
    return out


def pick_best_weight(predict_fn, actuals: list[str], grid: list[float]) -> float:
    """predict_fn(w) -> list[outcome-prob dicts] aligned with actuals. Minimize log-loss."""
    best_w, best_score = grid[0], float("inf")
    for w in grid:
        score = log_loss(predict_fn(w), actuals)
        if score < best_score:
            best_score, best_w = score, w
    return best_w
