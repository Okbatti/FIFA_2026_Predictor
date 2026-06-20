"""FIFA World Cup 2026 Predictor — atmospheric Streamlit dashboard.

Pure presentation layer: reads the artifacts written by scripts/update.py and
renders them over a living gradient-mesh "night pitch" background with glass
cards, Syne display type and a volt-green accent. No model logic.
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
st.set_page_config(page_title="WC2026 Predictor", page_icon="🏆", layout="wide")

_GRAIN = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E"
          "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E"
          "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap' rel='stylesheet'>"
    "<style>"
    ":root{--bg0:#04100b;--volt:#c6ff3a;--volt-d:#9bd400;--cyan:#2ee6b0;--blue:#6c8cff;--ink:#eafff4;--muted:#7fa896;--draw:#536b60;--away:#ff5d7a;--glass:rgba(255,255,255,.045);--glassb:rgba(255,255,255,.10);}"
    ".stApp{background:radial-gradient(120% 120% at 50% -10%,#0a2018 0%,#05130d 45%,#030b07 100%);color:var(--ink);font-family:'Outfit',sans-serif;}"
    ".stApp:before{content:'';position:fixed;inset:-25%;z-index:0;pointer-events:none;background:radial-gradient(38% 46% at 16% 18%,rgba(46,230,176,.22),transparent 60%),radial-gradient(42% 50% at 86% 12%,rgba(108,140,255,.20),transparent 62%),radial-gradient(46% 54% at 74% 88%,rgba(198,255,58,.14),transparent 60%),radial-gradient(40% 48% at 22% 92%,rgba(46,230,176,.14),transparent 60%);filter:blur(10px);animation:drift 26s ease-in-out infinite alternate;}"
    "@keyframes drift{0%{transform:translate3d(0,0,0) scale(1)}100%{transform:translate3d(-3%,-2.5%,0) scale(1.12)}}"
    ".stApp:after{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.05;mix-blend-mode:overlay;background-image:" + _GRAIN + ";}"
    ".bgpitch{position:fixed;inset:0;z-index:0;pointer-events:none;display:flex;align-items:center;justify-content:center;opacity:.05;}"
    ".bgpitch .circle{width:560px;height:560px;border:2px solid var(--ink);border-radius:50%;position:relative;}"
    ".bgpitch .circle:before{content:'';position:absolute;left:50%;top:-100vh;width:2px;height:300vh;background:var(--ink);transform:translateX(-50%);}"
    ".bgpitch .spot{width:14px;height:14px;background:var(--ink);border-radius:50%;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);}"
    "[data-testid='stMain'] .block-container{position:relative;z-index:2;max-width:900px;margin:0 auto;padding-top:2.2rem;}"
    "h1,h2,h3,h4{font-family:'Syne',sans-serif !important;color:var(--ink);letter-spacing:-.01em;}"
    "#MainMenu,header,footer{visibility:hidden;}"
    ".hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 30px 26px;background:linear-gradient(135deg,rgba(46,230,176,.10),rgba(108,140,255,.06));border:1px solid var(--glassb);backdrop-filter:blur(8px);}"
    ".hero .ghost{position:absolute;right:-12px;top:-46px;font-family:'Syne';font-weight:800;font-size:13rem;line-height:1;color:rgba(198,255,58,.07);pointer-events:none;}"
    ".eyebrow{font-family:'Outfit';text-transform:uppercase;letter-spacing:.32em;font-size:.72rem;font-weight:600;color:var(--volt);}"
    ".display{font-family:'Syne',sans-serif;font-weight:800;font-size:3.1rem;line-height:.98;margin:.3rem 0 .5rem;letter-spacing:-.02em;}"
    ".display .vt{color:var(--volt);text-shadow:0 0 30px rgba(198,255,58,.45);}"
    ".dek{font-family:'Outfit';color:var(--muted);font-size:1rem;max-width:600px;line-height:1.5;}"
    ".meta{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;}"
    ".tag{font-family:'Outfit';font-size:.76rem;color:var(--ink);background:rgba(255,255,255,.06);border:1px solid var(--glassb);padding:5px 12px;border-radius:999px;}"
    ".tag b{color:var(--volt);font-weight:600;}"
    ".sec{font-family:'Syne';text-transform:uppercase;letter-spacing:.14em;font-size:.86rem;font-weight:700;color:var(--muted);margin:1.7rem 0 .8rem;}"
    ".stTabs [data-baseweb='tab-list']{gap:8px;border-bottom:1px solid var(--glassb);}"
    ".stTabs [data-baseweb='tab']{font-family:'Syne';font-weight:700;font-size:.92rem;letter-spacing:.02em;color:var(--muted);background:transparent;padding:7px 4px;}"
    ".stTabs [aria-selected='true']{color:var(--volt) !important;border-bottom:2px solid var(--volt);}"
    ".match{background:var(--glass);border:1px solid var(--glassb);border-radius:18px;padding:18px 22px;margin-bottom:14px;backdrop-filter:blur(14px) saturate(1.15);box-shadow:0 14px 40px rgba(0,0,0,.45);transition:transform .15s ease,border-color .15s ease;}"
    ".match:hover{transform:translateY(-2px);border-color:rgba(198,255,58,.35);}"
    ".match .row{display:flex;align-items:center;justify-content:space-between;gap:12px;}"
    ".match .team{display:flex;align-items:center;gap:12px;flex:1;font-family:'Outfit';font-weight:600;font-size:1.2rem;}"
    ".match .team.away{justify-content:flex-end;text-align:right;}"
    ".match .flag{font-size:1.85rem;line-height:1;filter:drop-shadow(0 2px 6px rgba(0,0,0,.5));}"
    ".score{font-family:'Syne',sans-serif;font-weight:800;font-size:2.3rem;min-width:108px;text-align:center;color:var(--volt);font-variant-numeric:tabular-nums;text-shadow:0 0 26px rgba(198,255,58,.45);line-height:1;}"
    ".score small{display:block;font-family:'Outfit';font-weight:600;font-size:.6rem;text-transform:uppercase;letter-spacing:.18em;color:var(--muted);margin-top:6px;text-shadow:none;}"
    ".wdl{display:flex;height:8px;border-radius:6px;overflow:hidden;margin-top:16px;background:rgba(0,0,0,.35);}"
    ".wdl span{display:block;}.wdl .w{background:linear-gradient(90deg,var(--volt-d),var(--volt));box-shadow:0 0 14px rgba(198,255,58,.5);}.wdl .d{background:var(--draw);}.wdl .l{background:var(--away);}"
    ".legend{display:flex;justify-content:space-between;margin-top:9px;font-family:'Outfit';font-size:.8rem;color:var(--muted);}"
    ".legend b{font-weight:700;font-variant-numeric:tabular-nums;}"
    ".legend .hp b{color:var(--volt);}.legend .ap b{color:var(--away);}.legend .dp b{color:var(--ink);}"
    ".chips{margin-top:13px;display:flex;gap:8px;flex-wrap:wrap;}"
    ".chip{font-family:'Outfit';font-size:.76rem;padding:4px 11px;border-radius:8px;background:rgba(255,255,255,.05);border:1px solid var(--glassb);color:var(--muted);font-variant-numeric:tabular-nums;}"
    ".chip b{color:var(--ink);font-weight:600;}"
    ".lb{display:flex;align-items:center;gap:15px;padding:13px 16px;margin-bottom:9px;background:var(--glass);border:1px solid var(--glassb);border-radius:14px;backdrop-filter:blur(12px);}"
    ".lb.top{border-color:rgba(198,255,58,.4);box-shadow:0 0 26px rgba(198,255,58,.12);}"
    ".lb .rk{font-family:'Syne';font-weight:800;font-size:1.15rem;color:var(--muted);width:30px;text-align:center;font-variant-numeric:tabular-nums;}"
    ".lb.top .rk{color:var(--volt);text-shadow:0 0 16px rgba(198,255,58,.5);}"
    ".lb .lf{font-size:1.7rem;filter:drop-shadow(0 2px 6px rgba(0,0,0,.5));}"
    ".lb .nm{flex:1;font-family:'Outfit';font-weight:600;font-size:1.08rem;}"
    ".lb .track{flex:1.25;height:9px;background:rgba(0,0,0,.35);border-radius:6px;overflow:hidden;}"
    ".lb .fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--volt));border-radius:6px;box-shadow:0 0 14px rgba(198,255,58,.45);}"
    ".lb .pc{font-family:'Syne';font-weight:800;font-size:1.25rem;color:var(--volt);width:66px;text-align:right;font-variant-numeric:tabular-nums;text-shadow:0 0 18px rgba(198,255,58,.4);}"
    ".stat{background:var(--glass);border:1px solid var(--glassb);border-radius:16px;padding:16px 18px;backdrop-filter:blur(12px);}"
    ".stat .label{font-family:'Outfit';text-transform:uppercase;letter-spacing:.14em;color:var(--muted);font-size:.72rem;font-weight:600;}"
    ".stat .val{font-family:'Syne';font-weight:800;font-size:2.3rem;color:var(--volt);line-height:1.05;font-variant-numeric:tabular-nums;text-shadow:0 0 22px rgba(198,255,58,.4);}"
    ".stat .sub{font-family:'Outfit';font-size:.76rem;color:var(--muted);}"
    ".rail{position:fixed;top:0;bottom:0;width:clamp(0px,calc((100vw - 960px)/2),330px);z-index:1;pointer-events:none;display:flex;flex-direction:column;align-items:center;justify-content:space-between;padding:46px 0;overflow:hidden;}"
    ".rail.l{left:0;}.rail.r{right:0;}"
    ".rail .glyph{font-size:3.4rem;filter:drop-shadow(0 0 22px rgba(198,255,58,.35));opacity:.8;}"
    ".rail .vtext{writing-mode:vertical-rl;font-family:'Syne';font-weight:800;font-size:2.5rem;letter-spacing:.12em;color:rgba(198,255,58,.12);text-transform:uppercase;white-space:nowrap;}"
    ".rail.r .vtext{transform:rotate(180deg);}"
    ".rail .foot{display:flex;flex-direction:column;align-items:center;gap:8px;}"
    ".rail .hosts{font-size:1.5rem;letter-spacing:4px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.5));}"
    ".rail .lab{font-family:'Outfit';font-weight:600;text-transform:uppercase;letter-spacing:.2em;font-size:.64rem;color:var(--muted);}"
    ".rail .stat2{font-family:'Syne';font-weight:800;font-size:1.5rem;color:rgba(234,255,244,.55);text-align:center;line-height:1;}"
    "@media(max-width:1100px){.rail{display:none;}}"
    "</style>"
    "<div class='bgpitch'><div class='circle'><div class='spot'></div></div></div>"
    "<div class='rail l'><div class='glyph'>🏆</div>"
    "<div class='vtext'>FIFA World Cup</div>"
    "<div class='foot'><div class='hosts'>🇨🇦 🇺🇸 🇲🇽</div><div class='lab'>Hosts · 2026</div></div></div>"
    "<div class='rail r'><div class='glyph'>⚽</div>"
    "<div class='vtext'>Monte Carlo · Forecast</div>"
    "<div class='foot'><div class='stat2'>48<br>teams</div><div class='lab'>104 matches</div></div></div>",
    unsafe_allow_html=True,
)


def _load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else {}


def _load_parquet(name):
    p = ART / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _plotly(fig, height=360):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit", font_color="#eafff4", margin=dict(l=10, r=10, t=40, b=10),
        height=height, title_font_family="Syne", title_font_size=16,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zeroline=False)
    return fig


meta = _load_json("meta.json")
updated = meta.get("updated_utc", "—")
if isinstance(updated, str) and "T" in updated:
    updated = updated.split(".")[0].replace("T", " ") + " UTC"
bw = meta.get("blend_weight", "—")

st.markdown(
    f"<div class='hero'><div class='ghost'>26</div>"
    f"<div class='eyebrow'>Live forecast · updated daily</div>"
    f"<div class='display'>WORLD CUP <span class='vt'>2026</span></div>"
    f"<div class='dek'>An Elo · Dixon-Coles · XGBoost ensemble trained on international results, "
    f"with a Monte-Carlo simulation of the whole remaining tournament.</div>"
    f"<div class='meta'><span class='tag'>Updated <b>{updated}</b></span>"
    f"<span class='tag'>Blend weight <b>{bw}</b></span></div></div>",
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
                f"<span class='dp'>draw <b>{pd_:.0%}</b></span>"
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
        c1.markdown(f"<div class='stat'><div class='label'>Log-loss</div><div class='val'>{ll:.3f}</div><div class='sub'>random ≈ 1.10 · lower better</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><div class='label'>Brier</div><div class='val'>{br:.3f}</div><div class='sub'>random ≈ 0.67 · lower better</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat'><div class='label'>Training games</div><div class='val'>{ntr}</div><div class='sub'>{nwc} from WC2026</div></div>", unsafe_allow_html=True)
        st.caption("Metrics are in-sample (training) estimates — a time-series holdout is a planned upgrade.")

        calib = pd.DataFrame(metrics.get("calibration", []))
        if not calib.empty:
            calib = calib.dropna(subset=["mean_pred", "obs_freq"])
        if not calib.empty:
            st.markdown("<div class='sec'>Calibration · home-win</div>", unsafe_allow_html=True)
            fig = px.line(calib, x="mean_pred", y="obs_freq", markers=True)
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#7fa896"))
            fig.update_traces(line_color="#c6ff3a", marker_color="#2ee6b0")
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
