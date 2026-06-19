from __future__ import annotations
import json
import numpy as np
import pandas as pd
from predictor import config
from predictor.features import build_features
from predictor.models.dixon_coles import DixonColes
from predictor.models.ml_model import GoalsML
from predictor.models.ensemble import blend_lambdas, score_grid, outcome_probs, top_scorelines
from predictor.simulate import simulate_bracket, make_sampler
from predictor import evaluate

FEATURES = ["elo_diff", "home_form", "away_form", "rest_diff"]


def _outcome(actual_home, actual_away):
    if actual_home > actual_away:
        return "home"
    if actual_home < actual_away:
        return "away"
    return "draw"


def run_pipeline(matches: pd.DataFrame, bracket: dict | None = None) -> dict:
    # Guarantee datetime dtype: concatenating live data with an empty results.csv
    # frame can coerce the date column to object, which breaks vectorized .dt access.
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], utc=True)
    finished = matches[matches.status == "FINISHED"].dropna(subset=["home_goals", "away_goals"])
    feats, elo = build_features(matches)

    dc = DixonColes().fit(finished)
    ml = GoalsML(FEATURES).fit(feats)

    def predict_blended(home, away, w, neutral=True):
        dc_l = dc.predict_lambdas(home, away, neutral=neutral)
        adv = 0.0 if neutral else elo.home_adv
        feat_row = {
            "elo_diff": (elo.rating(home) + adv) - elo.rating(away),
            "home_form": 1.3,
            "away_form": 1.3,
            "rest_diff": 0,
        }
        ml_l = ml.predict_lambdas(feat_row)
        return blend_lambdas(dc_l, ml_l, w)

    # tune blend weight on finished games using pre-match features from feats
    # (feats rows carry the actual pre-match elo_diff/form/rest, avoiding the
    #  train/serve mismatch that would arise from using post-match elo.rating())
    actuals = [_outcome(r.home_goals, r.away_goals) for r in feats.itertuples()]

    def preds_for_w(w):
        out = []
        for row in feats.itertuples():
            ml_l = ml.predict_lambdas({
                "elo_diff": row.elo_diff,
                "home_form": row.home_form,
                "away_form": row.away_form,
                "rest_diff": row.rest_diff,
            })
            dc_l = dc.predict_lambdas(row.home, row.away)
            lh, la = blend_lambdas(dc_l, ml_l, w)
            out.append(outcome_probs(score_grid(lh, la, rho=dc.rho)))
        return out

    best_w = evaluate.pick_best_weight(preds_for_w, actuals, grid=[0.3, 0.5, 0.7, 0.9, 1.0])

    # Cache best predictions once; reuse for all metrics (avoid triple recomputation)
    best_preds = preds_for_w(best_w)
    metrics = {
        "log_loss": evaluate.log_loss(best_preds, actuals),
        "brier": evaluate.brier(best_preds, actuals),
        "blend_weight": best_w,
        "n_train": len(finished),
    }
    calib = evaluate.calibration_bins(best_preds, actuals)

    # next games (scheduled)
    # NOTE: predict_blended uses post-match elo.rating() and hardcoded form/rest (1.3/1.3/0)
    # for future fixtures — form and rest are genuinely unknown for upcoming opponents, so
    # this is an accepted limitation of the MVP for the scheduling path only.
    upcoming = matches[matches.status == "SCHEDULED"]
    next_rows = []
    for r in upcoming.itertuples():
        lh, la = predict_blended(r.home, r.away, best_w)
        g = score_grid(lh, la, rho=dc.rho)
        p = outcome_probs(g)
        tops = top_scorelines(g, 3)
        next_rows.append({
            "date": r.date,
            "home": r.home,
            "away": r.away,
            "p_home": p["home"],
            "p_draw": p["draw"],
            "p_away": p["away"],
            "top_scores": ";".join(f"{i}-{j}:{v:.2f}" for (i, j), v in tops),
        })
    next_df = pd.DataFrame(next_rows)

    # cup odds via simulation
    cup_df = pd.DataFrame()
    if bracket is not None:
        # Strip empty rounds (can appear when not all knockout fixtures are known yet)
        clean_bracket = {"rounds": [r for r in bracket["rounds"] if r]}
        rng = np.random.default_rng(0)
        sampler = make_sampler(lambda h, a: predict_blended(h, a, best_w), rng)
        probs = simulate_bracket(clean_bracket, sampler, n=config.SIM_N)
        cup_df = pd.DataFrame(
            [{"team": t, **v} for t, v in probs.items()]
        ).sort_values("win", ascending=False)

    rankings = pd.DataFrame(
        [{"team": t, "strength": s, "elo": elo.rating(t)} for t, s in dc.team_strength().items()]
    ).sort_values("elo", ascending=False)

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    next_df.to_parquet(config.ARTIFACTS / "next_games.parquet")
    cup_df.to_parquet(config.ARTIFACTS / "cup_odds.parquet")
    rankings.to_parquet(config.ARTIFACTS / "rankings.parquet")
    (config.ARTIFACTS / "metrics.json").write_text(
        json.dumps({**metrics, "calibration": calib}, indent=2)
    )
    (config.ARTIFACTS / "meta.json").write_text(
        json.dumps(
            {
                "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "blend_weight": best_w,
            },
            indent=2,
        )
    )

    return {"metrics": metrics, "next_games": next_df, "cup_odds": cup_df}


def main():
    from predictor.ingest import fetch_wc_matches, load_results, merge_results, save_results

    live = fetch_wc_matches()
    merged = merge_results(load_results(), live)
    save_results(merged)
    # bracket construction from current knockout fixtures is added when WC2026 reaches knockouts;
    # until then run without simulation.
    run_pipeline(merged, bracket=None)


if __name__ == "__main__":
    main()
