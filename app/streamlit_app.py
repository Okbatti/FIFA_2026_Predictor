import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

st.set_page_config(page_title="WC2026 Predictor", layout="wide")
st.title("FIFA World Cup 2026 — Match & Bracket Predictor")

def _load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}

def _load_parquet(name):
    p = ART / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

meta = _load_json("meta.json")
if meta:
    st.caption(f"Last updated: {meta.get('updated_utc','?')} · blend weight w={meta.get('blend_weight','?')}")

tab1, tab2, tab3 = st.tabs(["Next Games", "Cup Odds", "Model Report"])

with tab1:
    ng = _load_parquet("next_games.parquet")
    if ng.empty:
        st.info("No upcoming fixtures available yet.")
    else:
        for r in ng.itertuples():
            st.subheader(f"{r.home} vs {r.away}")
            cols = st.columns(3)
            cols[0].metric(r.home, f"{r.p_home:.0%}")
            cols[1].metric("Draw", f"{r.p_draw:.0%}")
            cols[2].metric(r.away, f"{r.p_away:.0%}")
            st.caption(f"Likely scores: {r.top_scores}")

with tab2:
    cup = _load_parquet("cup_odds.parquet")
    if cup.empty:
        st.info("Bracket simulation runs once the knockout stage begins.")
    else:
        st.dataframe(cup, use_container_width=True)
        fig = px.bar(cup.head(12), x="team", y="win", title="Title probability (top 12)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    metrics = _load_json("metrics.json")
    if metrics:
        c = st.columns(3)
        c[0].metric("Log-loss", f"{metrics.get('log_loss',0):.3f}")
        c[1].metric("Brier", f"{metrics.get('brier',0):.3f}")
        c[2].metric("Train games", metrics.get("n_train","?"))
        calib = pd.DataFrame(metrics.get("calibration", []))
        if not calib.empty:
            calib = calib.dropna(subset=["mean_pred","obs_freq"])
            fig = px.line(calib, x="mean_pred", y="obs_freq", markers=True,
                          title="Calibration (home-win)")
            fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1, line=dict(dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
    rankings = _load_parquet("rankings.parquet")
    if not rankings.empty:
        st.subheader("Team strength rankings")
        st.dataframe(rankings, use_container_width=True)
