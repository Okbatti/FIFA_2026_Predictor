from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from predictor.models.dixon_coles import tau
from predictor import config

def blend_lambdas(dc: tuple[float,float], ml: tuple[float,float], w: float) -> tuple[float,float]:
    return (w*dc[0] + (1-w)*ml[0], w*dc[1] + (1-w)*ml[1])

def score_grid(lh: float, la: float, rho: float = -0.05,
               max_goals: int = config.MAX_GOALS_GRID) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals+1), lh)
    a = poisson.pmf(np.arange(max_goals+1), la)
    grid = np.outer(h, a)
    # Dixon-Coles low-score correction on the 2x2 corner
    for i in (0,1):
        for j in (0,1):
            grid[i,j] *= tau(i,j,lh,la,rho)
    return grid / grid.sum()

def outcome_probs(grid: np.ndarray) -> dict[str,float]:
    home = float(np.tril(grid,-1).sum())
    away = float(np.triu(grid,1).sum())
    draw = float(np.trace(grid))
    return {"home":home,"draw":draw,"away":away}

def top_scorelines(grid: np.ndarray, k: int = 3) -> list[tuple[tuple[int,int],float]]:
    flat = [((i,j), float(grid[i,j])) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat[:k]
