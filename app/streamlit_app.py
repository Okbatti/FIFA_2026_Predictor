"""FIFA World Cup 2026 Predictor — broadcast-style Streamlit dashboard.

Pure presentation layer: reads the artifacts written by scripts/update.py and
renders them with a pitch-green + gold scoreboard aesthetic. No model logic.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

# ---------------------------------------------------------------- flag lookup
# Country flag emoji keyed by football-data.org team spelling. Unknown -> ⚽.
_FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "Spain": "🇪🇸",
    "England": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Belgium": "🇧🇪",
    "Italy": "🇮🇹", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Mexico": "🇲🇽", "United States": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Morocco": "🇲🇦",
    "Senegal": "🇸🇳", "Ivory Coast": "🇨🇮", "Cote d'Ivoire": "🇨🇮",
    "Nigeria": "🇳🇬", "Ghana": "🇬🇭", "Cameroon": "🇨🇲", "Egypt": "🇪🇬",
    "Tunisia": "🇹🇳", "Algeria": "🇩🇿", "South Africa": "🇿🇦", "Cape Verde Islands": "🇨🇻",
    "Congo DR": "🇨🇩", "Australia": "🇦🇺", "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷",
    "Qatar": "🇶🇦", "Jordan": "🇯🇴", "Uzbekistan": "🇺🇿", "Switzerland": "🇨🇭",
    "Denmark": "🇩🇰", "Sweden": "🇸🇪", "Norway": "🇳🇴", "Poland": "🇵🇱",
    "Austria": "🇦🇹", "Scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "Turkey": "🇹🇷", "Türkiye": "🇹🇷", "Ukraine": "🇺🇦", "Serbia": "🇷🇸",
    "Czechia": "🇨🇿", "Bosnia-Herzegovina": "🇧🇦", "Wales": "🏴\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f",
    "Ecuador": "🇪🇨", "Paraguay": "🇵🇾", "Peru": "🇵🇪", "Chile": "🇨🇱",
    "Panama": "🇵🇦", "Costa Rica": "🇨🇷", "Honduras": "🇭🇳", "Jamaica": "🇯🇲",
    "Haiti": "🇭🇹", "Curaçao": "🇨🇼", "New Zealand": "🇳🇿", "Greece": "🇬🇷",
}


def flag(team: str | None) -> str:
    if not isinstance(team, str):
        return "⚽"
    return _FLAGS.get(team, "⚽")


# ---------------------------------------------------------------- page + style
st.set_page_config(page_title="WC2026 Predictor", page_icon="🏆", layout="wide")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;500;600;700&display=swap' rel='stylesheet'>"
    "<style>"
    ":root{--pitch-0:#02160c;--pitch-1:#063b1f;--gold:#FFC400;--gold-soft:#ffe08a;--ink:#EAF4ED;--muted:#8fb6a0;--win:#26d07c;--draw:#5c7868;--loss:#ff6b6b;}"
    ".stApp{background:radial-gradient(1200px 600px at 80% -10%,rgba(255,196,0,.10),transparent 60%),repeating-linear-gradient(90deg,rgba(255,255,255,.020) 0 70px,rgba(0,0,0,0) 70px 140px),linear-gradient(160deg,var(--pitch-0),var(--pitch-1) 55%,#02130a);color:var(--ink);font-family:'Barlow Condensed',sans-serif;}"
    ".block-container{padding-top:1.4rem;max-width:1180px;}"
    "h1,h2,h3,h4{font-family:'Bebas Neue',sans-serif !important;letter-spacing:.04em;color:var(--ink);}"
    ".hero{border:1px solid rgba(255,196,0,.35);border-radius:18px;padding:22px 28px;background:linear-gradient(135deg,rgba(10,90,48,.55),rgba(2,22,12,.55));box-shadow:0 18px 50px rgba(0,0,0,.45);position:relative;overflow:hidden;}"
    ".hero:before{content:'';position:absolute;right:-40px;top:-40px;width:180px;height:180px;background:radial-gradient(circle,rgba(255,196,0,.30),transparent 70%);filter:blur(6px);}"
    ".hero h1{font-size:3.1rem;line-height:.92;margin:0;}"
    ".hero .accent{color:var(--gold);}"
    ".hero p{font-family:'Barlow Condensed';color:var(--muted);margin:.3rem 0 0;font-size:1.05rem;letter-spacing:.06em;text-transform:uppercase;}"
    ".pill{display:inline-block;margin-top:12px;padding:5px 14px;border-radius:999px;background:rgba(255,196,0,.12);border:1px solid rgba(255,196,0,.4);color:var(--gold-soft);font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;}"
    ".stTabs [data-baseweb='tab-list']{gap:6px;border-bottom:1px solid rgba(255,255,255,.08);}"
    ".stTabs [data-baseweb='tab']{font-family:'Bebas Neue';font-size:1.25rem;letter-spacing:.06em;background:transparent;color:var(--muted);padding:8px 18px;}"
    ".stTabs [aria-selected='true']{color:var(--gold) !important;border-bottom:3px solid var(--gold);}"
    ".match{border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px 20px;margin-bottom:14px;background:linear-gradient(180deg,rgba(10,90,48,.30),rgba(2,22,12,.45));box-shadow:0 10px 26px rgba(0,0,0,.35);}"
    ".match .row{display:flex;align-items:center;justify-content:space-between;gap:10px;}"
    ".match .team{display:flex;align-items:center;gap:10px;flex:1;font-size:1.45rem;font-family:'Barlow Condensed';font-weight:600;}"
    ".match .team.away{justify-content:flex-end;text-align:right;}"
    ".match .flag{font-size:1.9rem;line-height:1;}"
    ".score{font-family:'Bebas Neue';font-size:2.5rem;color:var(--gold);min-width:120px;text-align:center;background:rgba(0,0,0,.30);border:1px solid rgba(255,196,0,.35);border-radius:12px;padding:2px 10px;text-shadow:0 0 18px rgba(255,196,0,.35);}"
    ".score small{display:block;font-family:'Barlow Condensed';font-size:.72rem;color:var(--muted);letter-spacing:.1em;margin-top:-4px;}"
    ".wdl{display:flex;height:16px;border-radius:8px;overflow:hidden;margin-top:14px;border:1px solid rgba(255,255,255,.08);}"
    ".wdl span{display:flex;align-items:center;justify-content:center;font-size:.72rem;color:#04140b;font-weight:700;}"
    ".wdl .w{background:var(--win);}.wdl .d{background:var(--draw);color:var(--ink);}.wdl .l{background:var(--loss);}"
    ".legend{display:flex;justify-content:space-between;margin-top:6px;font-size:.8rem;color:var(--muted);letter-spacing:.04em;}"
    ".chips{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;}"
    ".chip{font-size:.8rem;padding:3px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.10);color:var(--gold-soft);letter-spacing:.03em;}"
    ".podium{display:flex;gap:14px;margin:6px 0 18px;}"
    ".pod{flex:1;border-radius:16px;padding:18px 14px;text-align:center;background:linear-gradient(180deg,rgba(10,90,48,.40),rgba(2,22,12,.55));border:1px solid rgba(255,255,255,.08);}"
    ".pod.first{border-color:var(--gold);box-shadow:0 0 30px rgba(255,196,0,.25);transform:translateY(-6px);}"
    ".pod .rank{font-family:'Bebas Neue';font-size:1.1rem;color:var(--muted);letter-spacing:.1em;}"
    ".pod .pflag{font-size:2.8rem;}"
    ".pod .pname{font-family:'Barlow Condensed';font-weight:700;font-size:1.35rem;margin-top:4px;}"
    ".pod .podds{font-family:'Bebas Neue';font-size:2.6rem;color:var(--gold);text-shadow:0 0 18px rgba(255,196,0,.35);}"
    ".pod.first .podds{font-size:3.2rem;}"
    ".stat{border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px 18px;background:linear-gradient(180deg,rgba(10,90,48,.30),rgba(2,22,12,.45));}"
    ".stat .label{font-family:'Barlow Condensed';text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-size:.82rem;}"
    ".stat .val{font-family:'Bebas Neue';font-size:2.6rem;color:var(--gold);line-height:1;}"
    ".stat .sub{font-size:.8rem;color:var(--muted);}"
    "hr{border-color:rgba(255,255,255,.08);}"
    "</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
def _load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}


def _load_parquet(name):
    p = ART / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


meta = _load_json("meta.json")
updated = meta.get("updated_utc", "—")
if isinstance(updated, str) and "T" in updated:
    updated = updated.split(".")[0].replace("T", " ") + " UTC"

st.markdown(
    f"""
<div class="hero">
  <h1>WORLD CUP <span class="accent">2026</span> · PREDICTOR</h1>
  <p>Elo × Dixon-Coles × XGBoost ensemble · Monte-Carlo bracket</p>
  <span class="pill">⟳ Updated {updated}</span>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

tab1, tab2, tab3 = st.tabs(["⚽  Next Games", "🏆  Cup Odds", "📊  Model Report"])


def _parse_top(top_scores: str):
    out = []
    for part in str(top_scores).split(";"):
        if ":" in part:
            sc, p = part.split(":")
            out.append((sc.strip(), float(p)))
    return out


# ---------------------------------------------------------------- Next Games
with tab1:
    ng = _load_parquet("next_games.parquet")
    if ng.empty:
        st.info("No upcoming fixtures available yet.")
    else:
        st.caption(f"{len(ng)} upcoming fixtures · win / draw / loss probability and most-likely scorelines")
        for r in ng.itertuples():
            tops = _parse_top(r.top_scores)
            best = tops[0][0] if tops else "—"
            ph, pd_, pa = r.p_home, r.p_draw, r.p_away
            chips = "".join(f'<span class="chip">{s} · {p:.0%}</span>' for s, p in tops[:3])
            st.markdown(
                f"""
<div class="match">
  <div class="row">
    <div class="team home"><span class="flag">{flag(r.home)}</span><span>{r.home}</span></div>
    <div class="score">{best}<small>most likely</small></div>
    <div class="team away"><span>{r.away}</span><span class="flag">{flag(r.away)}</span></div>
  </div>
  <div class="wdl">
    <span class="w" style="width:{max(ph*100,6):.0f}%">{ph:.0%}</span>
    <span class="d" style="width:{max(pd_*100,6):.0f}%">{pd_:.0%}</span>
    <span class="l" style="width:{max(pa*100,6):.0f}%">{pa:.0%}</span>
  </div>
  <div class="legend"><span>{flag(r.home)} {r.home} win</span><span>draw</span><span>{r.away} win {flag(r.away)}</span></div>
  <div class="chips">{chips}</div>
</div>
""",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- Cup Odds
with tab2:
    cup = _load_parquet("cup_odds.parquet")
    if cup.empty:
        st.info("Cup odds appear once group-stage fixtures are available to simulate.")
    else:
        cup = cup.sort_values("win", ascending=False).reset_index(drop=True)
        st.caption("Title & stage probabilities — full-tournament Monte-Carlo from the current standings")

        top3 = cup.head(3)
        medals = ["① CHAMPION", "② RUNNER-UP", "③ THIRD"]
        order = [1, 0, 2] if len(top3) >= 3 else list(range(len(top3)))  # center the leader
        cells = []
        for slot, idx in enumerate(order):
            row = top3.iloc[idx]
            first = "first" if idx == 0 else ""
            cells.append(
                f'<div class="pod {first}"><div class="rank">{medals[idx]}</div>'
                f'<div class="pflag">{flag(row.team)}</div><div class="pname">{row.team}</div>'
                f'<div class="podds">{row.win:.1%}</div></div>'
            )
        st.markdown(f'<div class="podium">{"".join(cells)}</div>', unsafe_allow_html=True)

        show = cup.copy()
        show["team"] = show["team"].map(lambda t: f"{flag(t)}  {t}")
        stage_cols = [c for c in ["R32", "R16", "QF", "SF", "FINAL", "win"] if c in show.columns]
        colcfg = {
            c: st.column_config.ProgressColumn(
                c.upper() if c != "win" else "WIN", format="%.0f%%", min_value=0.0, max_value=1.0
            )
            for c in stage_cols
        }
        colcfg["team"] = st.column_config.TextColumn("TEAM")
        st.dataframe(
            show[["team"] + stage_cols], column_config=colcfg,
            hide_index=True, use_container_width=True, height=460,
        )

        fig = px.bar(cup.head(12), x="win", y="team", orientation="h",
                     title="TITLE PROBABILITY · TOP 12")
        fig.update_traces(marker_color="#FFC400", marker_line_width=0)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Barlow Condensed", font_color="#EAF4ED",
            yaxis=dict(autorange="reversed", title=""), xaxis=dict(tickformat=".0%", title=""),
            margin=dict(l=10, r=10, t=50, b=10), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Model Report
with tab3:
    metrics = _load_json("metrics.json")
    if metrics:
        ll = metrics.get("log_loss", 0.0)
        br = metrics.get("brier", 0.0)
        ntr = metrics.get("n_train", "—")
        nwc = metrics.get("n_wc_finished", "—")
        bw = metrics.get("blend_weight", "—")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat"><div class="label">Log-loss</div><div class="val">{ll:.3f}</div><div class="sub">vs random 1.10 · lower better</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat"><div class="label">Brier</div><div class="val">{br:.3f}</div><div class="sub">vs random 0.67 · lower better</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat"><div class="label">Training games</div><div class="val">{ntr}</div><div class="sub">{nwc} WC2026 finished</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat"><div class="label">Blend weight</div><div class="val">{bw}</div><div class="sub">Dixon-Coles share</div></div>', unsafe_allow_html=True)
        st.caption("Metrics are in-sample (training) estimates — a time-series holdout is a planned upgrade.")

        calib = pd.DataFrame(metrics.get("calibration", []))
        if not calib.empty:
            calib = calib.dropna(subset=["mean_pred", "obs_freq"])
        if not calib.empty:
            fig = px.line(calib, x="mean_pred", y="obs_freq", markers=True,
                          title="CALIBRATION · HOME-WIN")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="#8fb6a0"))
            fig.update_traces(line_color="#FFC400", marker_color="#26d07c")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Barlow Condensed", font_color="#EAF4ED",
                xaxis=dict(title="predicted", tickformat=".0%"),
                yaxis=dict(title="observed", tickformat=".0%"),
                margin=dict(l=10, r=10, t=50, b=10), height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

    rankings = _load_parquet("rankings.parquet")
    if not rankings.empty:
        st.markdown("### TEAM STRENGTH RANKINGS")
        rk = rankings.copy()
        rk.insert(0, "rank", range(1, len(rk) + 1))
        rk["team"] = rk["team"].map(lambda t: f"{flag(t)}  {t}")
        st.dataframe(
            rk, hide_index=True, use_container_width=True, height=420,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "team": st.column_config.TextColumn("TEAM"),
                "elo": st.column_config.NumberColumn("ELO", format="%.0f"),
                "strength": st.column_config.NumberColumn("DC STRENGTH", format="%.2f"),
            },
        )
