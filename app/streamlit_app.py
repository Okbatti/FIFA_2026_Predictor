"""FIFA World Cup 2026 Predictor — editorial-analytics Streamlit dashboard.

Pure presentation layer: reads the artifacts written by scripts/update.py and
renders them in a clean editorial style (off-white paper, Fraunces serif
headlines, electric-blue accent). No model logic.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ART = Path(__file__).resolve().parents[1] / "artifacts"

# ---------------------------------------------------------------- flag lookup
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
st.set_page_config(page_title="WC2026 Predictor", page_icon="🏆", layout="centered")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Archivo:wght@400;500;600;700&display=swap' rel='stylesheet'>"
    "<style>"
    ":root{--paper:#f7f5f0;--panel:#ffffff;--ink:#17171b;--muted:#6f6c63;--line:#e4e0d7;--accent:#1f5eff;--accent-soft:#eaf0ff;--draw:#cfccc2;--away:#e5484d;}"
    ".stApp{background:var(--paper);color:var(--ink);font-family:'Archivo',sans-serif;}"
    ".block-container{padding-top:2rem;max-width:840px;}"
    "h1,h2,h3,h4{font-family:'Fraunces',serif !important;color:var(--ink);letter-spacing:-.01em;}"
    "#MainMenu,header,footer{visibility:hidden;}"
    ".eyebrow{font-family:'Archivo';text-transform:uppercase;letter-spacing:.22em;font-size:.74rem;font-weight:600;color:var(--accent);}"
    ".display{font-family:'Fraunces',serif;font-weight:600;font-size:3.05rem;line-height:1.0;margin:.18rem 0 .35rem;letter-spacing:-.02em;}"
    ".display .ital{font-style:italic;font-weight:500;}"
    ".dek{font-family:'Archivo';color:var(--muted);font-size:1.02rem;max-width:620px;}"
    ".rule{height:2px;background:var(--ink);margin:1.1rem 0 .5rem;}"
    ".meta{font-family:'Archivo';font-size:.8rem;color:var(--muted);letter-spacing:.02em;display:flex;gap:14px;flex-wrap:wrap;}"
    ".meta b{color:var(--ink);font-weight:600;}"
    ".sec{font-family:'Archivo';text-transform:uppercase;letter-spacing:.16em;font-size:.78rem;font-weight:600;color:var(--muted);margin:1.4rem 0 .7rem;border-bottom:1px solid var(--line);padding-bottom:.4rem;}"
    ".stTabs [data-baseweb='tab-list']{gap:26px;border-bottom:1px solid var(--line);}"
    ".stTabs [data-baseweb='tab']{font-family:'Archivo';font-weight:600;font-size:.92rem;letter-spacing:.03em;color:var(--muted);background:transparent;padding:6px 2px;}"
    ".stTabs [aria-selected='true']{color:var(--ink) !important;border-bottom:2px solid var(--accent);}"
    ".match{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin-bottom:12px;box-shadow:0 1px 2px rgba(20,20,20,.04),0 8px 22px rgba(20,20,20,.05);}"
    ".match .row{display:flex;align-items:center;justify-content:space-between;gap:12px;}"
    ".match .team{display:flex;align-items:center;gap:11px;flex:1;font-family:'Archivo';font-weight:600;font-size:1.16rem;}"
    ".match .team.away{justify-content:flex-end;text-align:right;}"
    ".match .flag{font-size:1.7rem;line-height:1;}"
    ".score{font-family:'Fraunces',serif;font-weight:600;font-size:2.05rem;min-width:96px;text-align:center;color:var(--ink);font-variant-numeric:tabular-nums;border-bottom:3px solid var(--accent);padding-bottom:2px;line-height:1;}"
    ".score small{display:block;font-family:'Archivo';font-weight:600;font-size:.62rem;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);border:0;margin-top:5px;}"
    ".wdl{display:flex;height:9px;border-radius:6px;overflow:hidden;margin-top:15px;background:var(--paper);}"
    ".wdl span{display:block;}.wdl .w{background:var(--accent);}.wdl .d{background:var(--draw);}.wdl .l{background:var(--away);}"
    ".legend{display:flex;justify-content:space-between;margin-top:7px;font-family:'Archivo';font-size:.78rem;color:var(--muted);}"
    ".legend b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums;}"
    ".legend .hp{color:var(--accent);}.legend .ap{color:var(--away);}"
    ".chips{margin-top:11px;display:flex;gap:7px;flex-wrap:wrap;}"
    ".chip{font-family:'Archivo';font-size:.76rem;padding:3px 9px;border-radius:6px;background:var(--paper);border:1px solid var(--line);color:var(--muted);font-variant-numeric:tabular-nums;}"
    ".chip b{color:var(--ink);font-weight:600;}"
    ".lb{display:flex;align-items:center;gap:14px;padding:11px 4px;border-bottom:1px solid var(--line);}"
    ".lb.top{padding-top:14px;padding-bottom:14px;}"
    ".lb .rk{font-family:'Fraunces',serif;font-weight:600;font-size:1.05rem;color:var(--muted);width:26px;text-align:right;font-variant-numeric:tabular-nums;}"
    ".lb.top .rk{color:var(--accent);}"
    ".lb .lf{font-size:1.55rem;}"
    ".lb .nm{flex:1;font-family:'Archivo';font-weight:600;font-size:1.05rem;}"
    ".lb .track{flex:1.3;height:8px;background:var(--paper);border-radius:5px;overflow:hidden;}"
    ".lb .fill{height:100%;background:var(--accent);border-radius:5px;}"
    ".lb .pc{font-family:'Fraunces',serif;font-weight:600;font-size:1.18rem;width:62px;text-align:right;font-variant-numeric:tabular-nums;}"
    ".stat{border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:var(--panel);}"
    ".stat .label{font-family:'Archivo';text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-size:.72rem;font-weight:600;}"
    ".stat .val{font-family:'Fraunces',serif;font-weight:600;font-size:2.2rem;color:var(--ink);line-height:1.05;font-variant-numeric:tabular-nums;}"
    ".stat .sub{font-family:'Archivo';font-size:.76rem;color:var(--muted);}"
    "</style>",
    unsafe_allow_html=True,
)


def _load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}


def _load_parquet(name):
    p = ART / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _plotly(fig, height=380):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Archivo", font_color="#17171b", margin=dict(l=10, r=10, t=46, b=10),
        height=height, title_font_family="Fraunces", title_font_size=16,
    )
    fig.update_xaxes(gridcolor="#e4e0d7", zeroline=False)
    fig.update_yaxes(gridcolor="#e4e0d7", zeroline=False)
    return fig


meta = _load_json("meta.json")
updated = meta.get("updated_utc", "—")
if isinstance(updated, str) and "T" in updated:
    updated = updated.split(".")[0].replace("T", " ") + " UTC"
bw = meta.get("blend_weight", "—")

st.markdown(
    f"<div class='eyebrow'>Forecast · updated daily</div>"
    f"<div class='display'>World Cup 2026 <span class='ital'>Predictor</span></div>"
    f"<div class='dek'>An Elo · Dixon-Coles · XGBoost ensemble trained on international results, "
    f"with a Monte-Carlo simulation of the remaining tournament.</div>"
    f"<div class='rule'></div>"
    f"<div class='meta'><span>Last updated <b>{updated}</b></span><span>Blend weight <b>{bw}</b></span></div>",
    unsafe_allow_html=True,
)
st.write("")

tab1, tab2, tab3 = st.tabs(["Next Games", "Cup Odds", "Model Report"])


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
        st.markdown(f"<div class='sec'>Upcoming fixtures · {len(ng)}</div>", unsafe_allow_html=True)
        for r in ng.itertuples():
            tops = _parse_top(r.top_scores)
            best = tops[0][0] if tops else "—"
            ph, pd_, pa = r.p_home, r.p_draw, r.p_away
            chips = "".join(f"<span class='chip'>{s} <b>{p:.0%}</b></span>" for s, p in tops[:3])
            st.markdown(
                f"<div class='match'>"
                f"<div class='row'>"
                f"<div class='team'><span class='flag'>{flag(r.home)}</span><span>{r.home}</span></div>"
                f"<div class='score'>{best}<small>most likely</small></div>"
                f"<div class='team away'><span>{r.away}</span><span class='flag'>{flag(r.away)}</span></div>"
                f"</div>"
                f"<div class='wdl'>"
                f"<span class='w' style='width:{ph*100:.2f}%'></span>"
                f"<span class='d' style='width:{pd_*100:.2f}%'></span>"
                f"<span class='l' style='width:{pa*100:.2f}%'></span>"
                f"</div>"
                f"<div class='legend'>"
                f"<span class='hp'>{r.home} win <b>{ph:.0%}</b></span>"
                f"<span>draw <b>{pd_:.0%}</b></span>"
                f"<span class='ap'><b>{pa:.0%}</b> {r.away} win</span>"
                f"</div>"
                f"<div class='chips'>{chips}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------- Cup Odds
with tab2:
    cup = _load_parquet("cup_odds.parquet")
    if cup.empty:
        st.info("Cup odds appear once group-stage fixtures are available to simulate.")
    else:
        cup = cup.sort_values("win", ascending=False).reset_index(drop=True)
        st.markdown("<div class='sec'>Title race · probability to win</div>", unsafe_allow_html=True)
        top = cup.head(12)
        mx = float(top["win"].max()) or 1.0
        rows = []
        for i, row in enumerate(top.itertuples(), start=1):
            cls = "lb top" if i <= 3 else "lb"
            rows.append(
                f"<div class='{cls}'>"
                f"<div class='rk'>{i}</div>"
                f"<div class='lf'>{flag(row.team)}</div>"
                f"<div class='nm'>{row.team}</div>"
                f"<div class='track'><div class='fill' style='width:{row.win / mx * 100:.1f}%'></div></div>"
                f"<div class='pc'>{row.win:.1%}</div>"
                f"</div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

        st.markdown("<div class='sec'>Stage-by-stage probabilities</div>", unsafe_allow_html=True)
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
            hide_index=True, use_container_width=True, height=440,
        )

# ---------------------------------------------------------------- Model Report
with tab3:
    metrics = _load_json("metrics.json")
    if metrics:
        ll = metrics.get("log_loss", 0.0)
        br = metrics.get("brier", 0.0)
        ntr = metrics.get("n_train", "—")
        nwc = metrics.get("n_wc_finished", "—")
        st.markdown("<div class='sec'>Backtest accuracy</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat'><div class='label'>Log-loss</div><div class='val'>{ll:.3f}</div><div class='sub'>random ≈ 1.10 · lower is better</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><div class='label'>Brier</div><div class='val'>{br:.3f}</div><div class='sub'>random ≈ 0.67 · lower is better</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat'><div class='label'>Training games</div><div class='val'>{ntr}</div><div class='sub'>{nwc} from WC2026</div></div>", unsafe_allow_html=True)
        st.caption("Metrics are in-sample (training) estimates — a time-series holdout is a planned upgrade.")

        calib = pd.DataFrame(metrics.get("calibration", []))
        if not calib.empty:
            calib = calib.dropna(subset=["mean_pred", "obs_freq"])
        if not calib.empty:
            st.markdown("<div class='sec'>Calibration · home-win</div>", unsafe_allow_html=True)
            fig = px.line(calib, x="mean_pred", y="obs_freq", markers=True)
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#b8b5ab"))
            fig.update_traces(line_color="#1f5eff", marker_color="#1f5eff")
            fig.update_layout(xaxis_title="predicted", yaxis_title="observed")
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(_plotly(fig, 340), use_container_width=True)

    rankings = _load_parquet("rankings.parquet")
    if not rankings.empty:
        st.markdown("<div class='sec'>Team strength rankings</div>", unsafe_allow_html=True)
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
