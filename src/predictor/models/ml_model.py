from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

class GoalsML:
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self._home = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  objective="count:poisson", subsample=0.9)
        self._away = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  objective="count:poisson", subsample=0.9)

    def fit(self, feats: pd.DataFrame) -> "GoalsML":
        X = feats[self.feature_cols]
        self._home.fit(X, feats["home_goals"])
        self._away.fit(X, feats["away_goals"])
        return self

    def predict_lambdas(self, feat_row: dict) -> tuple[float, float]:
        X = pd.DataFrame([{c: feat_row[c] for c in self.feature_cols}])
        lh = float(np.clip(self._home.predict(X)[0], 0.05, None))
        la = float(np.clip(self._away.predict(X)[0], 0.05, None))
        return lh, la
