# streamlit_app_v3.py
# ─────────────────────────────────────────────────────────────
# NEXUS  3rd GEN  –  Enhanced Signal Strategy
# Light theme · Improved ML (HistGBT + Ensemble + Calibration)
# Extended Features · Portfolio MC · Walk-Forward OOS
# ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed.*")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── ML Imports ──────────────────────────────────────────────
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# Light Design System
# ─────────────────────────────────────────────────────────────
THEME = {
    "bg":          "#F7F8FC",
    "bg_card":     "#FFFFFF",
    "bg_panel":    "#EEF1F8",
    "bg_sidebar":  "#1E2435",
    "accent1":     "#2563EB",   # Royal blue – primary
    "accent2":     "#F59E0B",   # Amber – warning
    "accent3":     "#0EA5E9",   # Sky – info
    "accent4":     "#7C3AED",   # Violet – prob
    "green":       "#16A34A",
    "red":         "#DC2626",
    "green_soft":  "#DCFCE7",
    "red_soft":    "#FEE2E2",
    "blue_soft":   "#DBEAFE",
    "text":        "#0F172A",
    "text_sec":    "#475569",
    "muted":       "#94A3B8",
    "border":      "#E2E8F0",
    "border_med":  "#CBD5E1",
    "shadow":      "rgba(15,23,42,0.07)",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFE",
    font=dict(family="'DM Mono', 'Fira Code', monospace", color=THEME["text_sec"], size=10),
    xaxis=dict(
        gridcolor="#EEF1F8", gridwidth=1,
        showline=True, linecolor=THEME["border_med"],
        zeroline=False, tickfont=dict(size=10),
        tickcolor=THEME["muted"],
    ),
    yaxis=dict(
        gridcolor="#EEF1F8", gridwidth=1,
        showline=True, linecolor=THEME["border_med"],
        zeroline=False, tickfont=dict(size=10),
        tickcolor=THEME["muted"],
    ),
    hoverlabel=dict(
        bgcolor=THEME["bg_card"],
        font_color=THEME["text"],
        font_family="'DM Mono', monospace",
        bordercolor=THEME["border_med"],
    ),
    margin=dict(t=52, b=44, l=58, r=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(size=10),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bordercolor=THEME["border"], borderwidth=1,
    ),
    modebar=dict(
        bgcolor="rgba(255,255,255,0.6)",
        color=THEME["muted"],
        activecolor=THEME["accent1"],
    ),
)

st.set_page_config(
    page_title="NEXUS — Signal Strategy v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCAL_TZ    = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 6
pd.options.display.float_format = "{:,.4f}".format

# ─────────────────────────────────────────────────────────────
# Global CSS  –  Light, Friendly, Modern
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@300;400;500&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&display=swap');

:root {
  --bg:         #F7F8FC;
  --bg-card:    #FFFFFF;
  --bg-panel:   #EEF1F8;
  --sidebar-bg: #1E2435;
  --blue:       #2563EB;
  --blue-soft:  #DBEAFE;
  --amber:      #F59E0B;
  --sky:        #0EA5E9;
  --violet:     #7C3AED;
  --green:      #16A34A;
  --green-soft: #DCFCE7;
  --red:        #DC2626;
  --red-soft:   #FEE2E2;
  --text:       #0F172A;
  --text-sec:   #475569;
  --muted:      #94A3B8;
  --border:     #E2E8F0;
  --border-med: #CBD5E1;
  --shadow:     rgba(15,23,42,0.07);
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ─ Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: none !important;
  box-shadow: 4px 0 24px rgba(0,0,0,0.12) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  color: #64748B !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.12) !important;
  color: #E2E8F0 !important;
  border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
  color: #93C5FD !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong { color: #E2E8F0 !important; }

/* ─ Metrics ───────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px 18px !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
  transition: all 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--blue) !important;
  box-shadow: 0 4px 16px rgba(37,99,235,0.10) !important;
  transform: translateY(-1px) !important;
}
[data-testid="metric-container"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ─ Expander ─────────────────────────────────────────── */
details {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  margin-bottom: 10px !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
  overflow: hidden !important;
}
details summary {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  padding: 14px 18px !important;
  color: var(--text-sec) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  cursor: pointer !important;
  background: var(--bg-card) !important;
}
details[open] summary {
  color: var(--blue) !important;
  background: var(--blue-soft) !important;
}
details > div { padding: 0 4px 4px 4px !important; }

/* ─ Tables ────────────────────────────────────────────── */
.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
}
.stDataFrame thead th {
  background: #F1F5F9 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  color: var(--text-sec) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  border-bottom: 1px solid var(--border-med) !important;
}
.stDataFrame tbody td {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  color: var(--text) !important;
  border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr:hover { background: var(--blue-soft) !important; }

/* ─ Buttons ───────────────────────────────────────────── */
.stButton > button {
  background: transparent !important;
  border: 1.5px solid var(--blue) !important;
  color: var(--blue) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  transition: all 0.15s ease !important;
  padding: 6px 16px !important;
}
.stButton > button:hover {
  background: var(--blue) !important;
  color: white !important;
  box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
}
.stButton > button[data-testid*="primary"] {
  background: var(--blue) !important;
  color: white !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 10px rgba(37,99,235,0.30) !important;
}
.stButton > button[data-testid*="primary"]:hover {
  background: #1D4ED8 !important;
  box-shadow: 0 4px 18px rgba(37,99,235,0.40) !important;
}

/* ─ Download buttons ──────────────────────────────────── */
.stDownloadButton > button {
  background: transparent !important;
  border: 1.5px solid var(--border-med) !important;
  color: var(--text-sec) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber) !important;
  background: #FFFBEB !important;
}

/* ─ Progress ─────────────────────────────────────────── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--blue), var(--sky)) !important;
  border-radius: 4px !important;
}

/* ─ Alerts ────────────────────────────────────────────── */
.stAlert {
  border-radius: 8px !important;
  font-size: 13px !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ─ Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 2px solid var(--border) !important;
  background: transparent !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  border-radius: 6px 6px 0 0 !important;
  padding: 8px 16px !important;
  transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--bg-panel) !important; color: var(--text-sec) !important; }
.stTabs [aria-selected="true"] {
  color: var(--blue) !important;
  background: var(--blue-soft) !important;
  border-bottom: 2px solid var(--blue) !important;
}

/* ─ Custom components ─────────────────────────────────── */
.nexus-header {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 36px;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.nexus-sub {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 6px;
}
.section-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0 16px 0;
  border-bottom: 2px solid var(--border);
  margin: 28px 0 20px 0;
}
.section-bar-label {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--text-sec);
}
.section-bar-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--blue);
  box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
}
.section-bar-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border-med), transparent);
}

/* KPI Grid */
.kpi-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
.kpi-box {
  flex: 1;
  min-width: 110px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: 0 1px 4px var(--shadow);
  transition: all 0.2s ease;
}
.kpi-box:hover {
  border-color: var(--blue);
  box-shadow: 0 4px 16px rgba(37,99,235,0.08);
  transform: translateY(-1px);
}
.kpi-label {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
}
.kpi-value {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
}
.kpi-pos .kpi-value { color: var(--green); }
.kpi-neg .kpi-value { color: var(--red); }
.kpi-info .kpi-value { color: var(--blue); }
.kpi-warn .kpi-value { color: var(--amber); }
.kpi-vio  .kpi-value { color: var(--violet); }

/* Signal Badges */
.signal-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.sig-enter {
  background: var(--green-soft);
  color: var(--green);
  border: 1.5px solid rgba(22,163,74,0.3);
}
.sig-exit {
  background: var(--red-soft);
  color: var(--red);
  border: 1.5px solid rgba(220,38,38,0.3);
}
.sig-hold {
  background: var(--bg-panel);
  color: var(--muted);
  border: 1.5px solid var(--border);
}

/* Ticker header card */
.ticker-card-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 20px;
  background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
  border: 1px solid var(--border);
  border-radius: 10px 10px 0 0;
  margin-bottom: -1px;
}
.ticker-symbol {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: var(--blue);
  letter-spacing: -0.01em;
}
.ticker-name {
  font-family: 'DM Sans', sans-serif;
  font-size: 13px;
  color: var(--text-sec);
}

/* Model badge */
.model-badge {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  background: #F0F4FF;
  border: 1px solid #BFDBFE;
  color: var(--blue);
  border-radius: 20px;
  padding: 2px 10px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 500;
}

/* Sidebar logo area */
.sidebar-logo {
  padding: 20px 16px 18px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 16px;
  background: linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(124,58,237,0.10) 100%);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(index=False, sep=";", decimal=",", date_format="%d.%m.%Y",
                     float_format=float_format).encode("utf-8-sig")

def _normalize_tickers(items: List[str]) -> List[str]:
    cleaned = []
    for x in items or []:
        if not isinstance(x, str): continue
        s = x.strip().upper()
        if s: cleaned.append(s)
    return list(dict.fromkeys(cleaned))

def parse_ticker_csv(path_or_buffer) -> List[str]:
    try: df = pd.read_csv(path_or_buffer)
    except Exception: df = pd.read_csv(path_or_buffer, sep=";")
    if df.empty: return []
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("ticker","symbol","symbols","isin","code"):
        if key in cols_lower:
            return _normalize_tickers(df[cols_lower[key]].astype(str).tolist())
    return _normalize_tickers(df[df.columns[0]].astype(str).tolist())

def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0

@st.cache_data(show_spinner=False, ttl=86400)
def get_ticker_name(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        try: info = tk.get_info()
        except Exception: info = getattr(tk, "info", {}) or {}
        for k in ("shortName","longName","displayName","companyName","name"):
            if k in info and info[k]: return str(info[k])
    except Exception: pass
    return ticker

def section(label: str):
    st.markdown(f"""
    <div class="section-bar">
        <div class="section-bar-dot"></div>
        <div class="section-bar-label">{label}</div>
        <div class="section-bar-line"></div>
    </div>
    """, unsafe_allow_html=True)

def kpi_row(items: list):
    parts = "".join(
        f'<div class="kpi-box {cls}">'
        f'<div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'</div>'
        for lbl, val, cls in items
    )
    st.markdown(f'<div class="kpi-grid">{parts}</div>', unsafe_allow_html=True)

def _pct(v, decimals=2):
    if not np.isfinite(v): return "–"
    s = "+" if v > 0 else ""
    return f"{s}{v:.{decimals}f}%"

def _eur(v):
    if not np.isfinite(v): return "–"
    s = "+" if v > 0 else ""
    return f"{s}{v:,.2f}€"

def _apply_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**PLOTLY_BASE, height=height)
    fig.update_xaxes(
        showspikes=True, spikecolor=THEME["muted"],
        spikethickness=1, spikedash="dot", spikemode="across",
    )
    fig.update_yaxes(
        showspikes=True, spikecolor=THEME["muted"],
        spikethickness=1, spikedash="dot",
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div style="font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:22px;
                  color:#F1F5F9;letter-spacing:-0.02em;">📈 NEXUS</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#93C5FD;
                  letter-spacing:0.14em;margin-top:3px;text-transform:uppercase;">
        Signal Strategy v3 · AI Enhanced
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**UNIVERSE**")
    ticker_source = st.selectbox("Quelle", ["Manuell", "CSV-Upload"], label_visibility="collapsed")
    tickers_final: List[str] = []

    if ticker_source == "Manuell":
        tickers_input = st.text_input("Tickers (Komma)", value="REGN, LULU, VOW3.DE, REI, DDL")
        tickers_final = _normalize_tickers([t for t in tickers_input.split(",") if t.strip()])
    else:
        st.caption("CSV mit Spalte **ticker**")
        uploads = st.file_uploader("CSV", type=["csv"], accept_multiple_files=True)
        collected = []
        if uploads:
            for up in uploads:
                try: collected += parse_ticker_csv(up)
                except Exception as e: st.error(f"{up.name}: {e}")
        base  = _normalize_tickers(collected)
        extra = st.text_input("Weitere Ticker", value="", key="extra_csv")
        extras = _normalize_tickers([t for t in extra.split(",") if t.strip()]) if extra else []
        tickers_final = _normalize_tickers(base + extras)
        if tickers_final:
            st.caption(f"**{len(tickers_final)}** Ticker")
            max_n = st.number_input("Max (0=alle)", 0, len(tickers_final), 0, step=10)
            if max_n: tickers_final = tickers_final[:int(max_n)]
            tickers_final = st.multiselect("Auswahl", tickers_final, default=tickers_final)

    if not tickers_final:
        tickers_final = _normalize_tickers(["REGN","VOW3.DE","LULU","REI","DDL"])

    st.download_button("⬇ Ticker-CSV", to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
                       file_name="tickers.csv", mime="text/csv", use_container_width=True)
    TICKERS = tickers_final

    st.divider()
    st.markdown("**ZEITRAUM**")
    col_d1, col_d2 = st.columns(2)
    START_DATE = col_d1.date_input("Von", pd.to_datetime("2025-01-01"))
    END_DATE   = col_d2.date_input("Bis", pd.to_datetime(datetime.now(LOCAL_TZ).date()))

    st.divider()
    st.markdown("**MODELL**")
    c1s, c2s = st.columns(2)
    LOOKBACK = c1s.number_input("Lookback", 10, 252, 35, step=5)
    HORIZON  = c2s.number_input("Horizon",  1, 10, 5)
    THRESH   = st.number_input("Target Threshold", 0.0, 0.1, 0.046, step=0.005, format="%.3f")
    ENTRY_PROB = st.slider("Entry Prob", 0.0, 1.0, 0.62, step=0.01)
    EXIT_PROB  = st.slider("Exit Prob",  0.0, 1.0, 0.48, step=0.01)
    if EXIT_PROB >= ENTRY_PROB:
        st.error("⚠ Exit muss kleiner als Entry sein"); st.stop()
    MIN_HOLD_DAYS = st.number_input("Min. Hold (Tage)", 0, 252, 5, step=1)
    COOLDOWN_DAYS = st.number_input("Cooldown (Tage)", 0, 252, 0, step=1)

    st.divider()
    st.markdown("**ML ALGORITHMUS**")
    model_type = st.selectbox(
        "Algorithmus",
        ["Ensemble (HistGBT+RF+ET)", "HistGradientBoosting", "GradientBoosting",
         "RandomForest", "ExtraTrees"],
        index=0,
    )
    use_calibration  = st.checkbox("Platt-Kalibrierung", value=True, help="Wahrscheinlichkeiten kalibrieren")
    use_feat_select  = st.checkbox("Feature-Selektion", value=False, help="Irrelevante Features entfernen")
    c5s, c6s = st.columns(2)
    n_estimators  = c5s.number_input("n_estimators", 10, 500, 120, step=10)
    learning_rate = c6s.number_input("lr", 0.01, 1.0, 0.08, step=0.01, format="%.2f")
    max_depth     = st.number_input("max_depth", 1, 10, 4, step=1)
    MODEL_PARAMS  = dict(n_estimators=int(n_estimators), learning_rate=float(learning_rate),
                         max_depth=int(max_depth), random_state=42)

    st.divider()
    st.markdown("**EXECUTION**")
    c3s, c4s = st.columns(2)
    COMMISSION   = c3s.number_input("Commission", 0.0, 0.02, 0.004, step=0.0001, format="%.4f")
    SLIPPAGE_BPS = c4s.number_input("Slippage bp", 0, 50, 5, step=1)
    POS_FRAC     = st.slider("Position Size", 0.1, 1.0, 1.0, step=0.1)
    INIT_CAP_PER_TICKER = st.number_input("Capital/Ticker (€)", 1000.0, 1_000_000.0, 10_000.0, step=1000.0, format="%.0f")

    st.divider()
    st.markdown("**INTRADAY**")
    use_live          = st.checkbox("Intraday Tail", value=True)
    intraday_interval = st.selectbox("Intervall", ["1m","2m","5m","15m"], index=2)
    fallback_last_session = st.checkbox("Session Fallback", value=False)
    exec_mode         = st.selectbox("Execution", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
    moc_cutoff_min    = st.number_input("MOC Cutoff (min)", 5, 60, 15, step=5)
    intraday_chart_type = st.selectbox("Intraday Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

    st.divider()
    st.markdown("**WALK-FORWARD / OOS**")
    use_walk_forward = st.checkbox("Walk-Forward OOS", value=False)
    wf_min_train     = st.number_input("WF min_train Bars", 40, 500, 120, step=10)

    st.divider()
    st.markdown("**OPTIONS DATA**")
    use_chain_live = st.checkbox("Live Optionskette (PCR/VOI)", value=True)
    atm_band_pct   = st.slider("ATM-Band ±%", 1, 15, 5) / 100.0
    max_days_to_exp= st.slider("Max. Laufzeit (Tage)", 7, 45, 21)
    n_expiries     = st.slider("Nächste n Verfälle", 1, 4, 2)

    st.divider()
    st.markdown("**PORTFOLIO FORECAST**")
    FORECAST_DAYS = st.number_input("Forecast Horizon (Tage)", 1, 30, 7, step=1)
    MC_SIMS       = st.number_input("MC Simulationen", 200, 5000, 1500, step=100)

    st.divider()
    col_r1, col_r2 = st.columns(2)
    if col_r1.button("🗑 Cache", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    if col_r2.button("↺ Rerun", use_container_width=True):
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
def get_price_data_tail_intraday(ticker, years=3, use_tail=True, interval="5m",
                                  fallback_last_session=False, exec_mode_key="Next Open (backtest+live)",
                                  moc_cutoff_min_val=15):
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)
    if df.empty: raise ValueError(f"Keine Daten: {ticker}")
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)
    df = df.sort_index().drop_duplicates()
    meta = {"tail_is_intraday": False, "tail_ts": None}

    if not use_tail:
        df.dropna(subset=["High","Low","Close","Open"], inplace=True)
        return df, meta

    try:
        intraday = tk.history(period="1d", interval=interval, auto_adjust=True, actions=False, prepost=False)
        if not intraday.empty:
            if intraday.index.tz is None: intraday.index = intraday.index.tz_localize("UTC")
            intraday.index = intraday.index.tz_convert(LOCAL_TZ)
            intraday = intraday.sort_index()
        else: intraday = pd.DataFrame()
    except Exception: intraday = pd.DataFrame()

    if exec_mode_key.startswith("Market-On-Close") and not intraday.empty:
        intraday = intraday.loc[:datetime.now(LOCAL_TZ) - timedelta(minutes=int(moc_cutoff_min_val))]

    if intraday.empty and fallback_last_session:
        try:
            intr5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
            if not intr5.empty:
                if intr5.index.tz is None: intr5.index = intr5.index.tz_localize("UTC")
                intr5.index = intr5.index.tz_convert(LOCAL_TZ)
                intraday = intr5.loc[str(intr5.index[-1].date())]
        except Exception: pass

    if not intraday.empty:
        lb = intraday.iloc[-1]
        dk = pd.Timestamp(lb.name.date(), tz=LOCAL_TZ)
        row = {"Open": float(intraday["Open"].iloc[0]), "High": float(intraday["High"].max()),
               "Low": float(intraday["Low"].min()), "Close": float(lb["Close"]),
               "Volume": float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan}
        if dk in df.index:
            for k, v in row.items(): df.loc[dk, k] = v
        else: df.loc[dk] = row
        df = df.sort_index()
        meta["tail_is_intraday"] = True; meta["tail_ts"] = lb.name

    df.dropna(subset=["High","Low","Close","Open"], inplace=True)
    return df, meta

@st.cache_data(show_spinner=False, ttl=180)
def get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval="5m"):
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty: return intr
    if intr.index.tz is None: intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr = intr.sort_index()
    keep = set(pd.Index(intr.index.normalize().unique())[-sessions:])
    return intr.loc[intr.index.normalize().isin(keep)].copy()

def load_all_prices(tickers, start, end, use_tail, interval, fallback_last, exec_key, moc_cutoff):
    price_map, meta_map = {}, {}
    if not tickers: return price_map, meta_map
    with st.spinner(f"📡 Kursdaten laden · {len(tickers)} Ticker"):
        prog = st.progress(0.0)
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
            fmap = {ex.submit(get_price_data_tail_intraday, tk, 3, use_tail, interval,
                              fallback_last, exec_key, int(moc_cutoff)): tk for tk in tickers}
            done = 0
            for fut in as_completed(fmap):
                tk = fmap[fut]
                try:
                    df_full, meta = fut.result()
                    df_use = df_full.loc[str(start):str(end)].copy()
                    if not df_use.empty:
                        price_map[tk] = df_use; meta_map[tk] = meta
                except Exception as e: st.error(f"❌ {tk}: {e}")
                finally:
                    done += 1; prog.progress(done / len(tickers))
    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────
def _atm_strike(ref_px, strikes):
    if not np.isfinite(ref_px) or strikes.size == 0: return np.nan
    return float(strikes[np.argmin(np.abs(strikes - ref_px))])

def _band_mask(strikes, atm, band):
    if not np.isfinite(atm): return pd.Series([False]*len(strikes), index=strikes.index)
    return strikes.between(atm*(1-band), atm*(1+band))

@st.cache_data(show_spinner=False, ttl=180)
def get_equity_chain_aggregates_for_today(ticker, ref_price, atm_band, n_exps, max_days):
    tk = yf.Ticker(ticker)
    try: exps = tk.options or []
    except Exception: exps = []
    if not exps: return pd.DataFrame()

    today = pd.Timestamp.today(tz=LOCAL_TZ).normalize()
    exps_filt = sorted(
        [(pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize(), e)
         for e in exps if (pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize() - today).days <= max_days],
        key=lambda x: x[0]
    )
    exps_use = [e for _, e in exps_filt[:max(1, n_exps)]]
    if not exps_use: return pd.DataFrame()

    rows = []
    for e in exps_use:
        try:
            ch = tk.option_chain(e); calls, puts = ch.calls.copy(), ch.puts.copy()
        except Exception: continue
        for df in (calls, puts):
            for c in ["volume","openInterest","impliedVolatility","strike"]:
                if c not in df.columns: df[c] = np.nan
        strikes = np.sort(pd.concat([calls["strike"],puts["strike"]]).dropna().unique())
        atm = _atm_strike(ref_price, strikes)
        mC = calls[_band_mask(calls["strike"], atm, atm_band)]
        mP = puts [_band_mask(puts ["strike"], atm, atm_band)]
        rows.append({"exp":e,
            "vol_c":float(np.nansum(mC["volume"])),    "vol_p":float(np.nansum(mP["volume"])),
            "oi_c": float(np.nansum(mC["openInterest"])),"oi_p": float(np.nansum(mP["openInterest"])),
            "voi_c":float(np.nansum(mC["volume"]))/max(float(np.nansum(mC["openInterest"])),1.0),
            "voi_p":float(np.nansum(mP["volume"]))/max(float(np.nansum(mP["openInterest"])),1.0),
            "iv_c": float(np.nanmean(mC["impliedVolatility"])) if len(mC) else np.nan,
            "iv_p": float(np.nanmean(mP["impliedVolatility"])) if len(mP) else np.nan})
    if not rows: return pd.DataFrame()
    agg = pd.DataFrame(rows).agg({"vol_c":"sum","vol_p":"sum","oi_c":"sum","oi_p":"sum",
                                  "voi_c":"mean","voi_p":"mean","iv_c":"mean","iv_p":"mean"})
    out = pd.DataFrame([{
        "PCR_vol": float(agg["vol_p"]/max(agg["vol_c"],1.0)),
        "PCR_oi":  float(agg["oi_p"] /max(agg["oi_c"],1.0)),
        "VOI_call": float(agg["voi_c"]), "VOI_put": float(agg["voi_p"]),
        "IV_skew_p_minus_c": float(agg["iv_p"] - agg["iv_c"]),
        "VOL_tot": float(agg["vol_c"]+agg["vol_p"]),
        "OI_tot":  float(agg["oi_c"]+agg["oi_p"]),
    }])
    out.index = [pd.Timestamp.today(tz=LOCAL_TZ).normalize()]
    return out


# ─────────────────────────────────────────────────────────────
# ★  ENHANCED Feature Engineering
#    +RSI, +MACD, +Bollinger, +ATR, +Momentum, +Volume features
#    +Trend strength, +Mean reversion, +Volatility regime
# ─────────────────────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = _ema(close, fast); ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    sig_line  = _ema(macd_line, signal)
    hist_line = macd_line - sig_line
    return macd_line, sig_line, hist_line

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma   = close.rolling(period).mean()
    std  = close.rolling(period).std(ddof=0)
    upper = ma + n_std * std
    lower = ma - n_std * std
    pct_b = (close - lower) / (upper - lower + 1e-12)
    width = (upper - lower) / (ma + 1e-12)
    return pct_b, width

def _stochastic(high, low, close, k=14, d=3):
    lo  = low.rolling(k).min()
    hi  = high.rolling(k).max()
    k_  = 100 * (close - lo) / (hi - lo + 1e-12)
    d_  = k_.rolling(d).mean()
    return k_, d_

def make_features(df, lookback, horizon, exog=None):
    if len(df) < (lookback + horizon + 30):
        raise ValueError("Zu wenige Bars.")
    f = df.copy()
    c = f["Close"]; h = f["High"]; l = f["Low"]; o = f["Open"]
    v = f["Volume"] if "Volume" in f.columns else pd.Series(np.nan, index=f.index)

    # ── Original features ──
    f["Range"]     = h.rolling(lookback).max() - l.rolling(lookback).min()
    f["SlopeHigh"] = h.rolling(lookback).apply(slope, raw=True)
    f["SlopeLow"]  = l.rolling(lookback).apply(slope, raw=True)

    # ── Momentum / Returns ──
    for w in [3, 5, 10, 21]:
        f[f"Ret_{w}d"]   = c.pct_change(w)
        f[f"RetLog_{w}d"] = np.log(c / c.shift(w))
    f["Ret_vs_ATH"]  = c / c.rolling(min(lookback*3, 252)).max() - 1

    # ── RSI ──
    for p in [7, 14, 21]:
        f[f"RSI_{p}"] = _rsi(c, p)
    f["RSI_slope"] = f["RSI_14"].diff(5)

    # ── MACD ──
    macd_l, macd_s, macd_h = _macd(c)
    f["MACD_line"] = macd_l; f["MACD_signal"] = macd_s; f["MACD_hist"] = macd_h
    f["MACD_hist_slope"] = macd_h.diff(3)

    # ── Bollinger Bands ──
    for p in [10, 20]:
        pb, bw = _bollinger(c, p)
        f[f"BollPctB_{p}"] = pb; f[f"BollWidth_{p}"] = bw
    f["BollWidth_trend"] = f["BollWidth_20"].diff(5)

    # ── ATR / Volatility ──
    f["ATR_14"] = _atr(h, l, c, 14)
    f["ATR_rel"] = f["ATR_14"] / (c + 1e-12)          # normalized
    f["Vol_5d"]  = c.pct_change().rolling(5).std()
    f["Vol_20d"] = c.pct_change().rolling(20).std()
    f["Vol_regime"] = f["Vol_5d"] / (f["Vol_20d"] + 1e-12)  # short vs long vol ratio

    # ── EMA trend features ──
    for sp in [9, 21, 50]:
        f[f"EMA_{sp}"] = _ema(c, sp)
        f[f"Price_vs_EMA_{sp}"] = c / f[f"EMA_{sp}"] - 1
    f["EMA_9_21_cross"]  = f["EMA_9"]  / (f["EMA_21"] + 1e-12) - 1
    f["EMA_21_50_cross"] = f["EMA_21"] / (f["EMA_50"] + 1e-12) - 1

    # ── Stochastic ──
    f["Stoch_K"], f["Stoch_D"] = _stochastic(h, l, c)
    f["Stoch_K_slope"] = f["Stoch_K"].diff(3)

    # ── Volume features ──
    if not v.isna().all():
        f["Vol_20ma"]   = v.rolling(20).mean()
        f["Vol_ratio"]  = v / (f["Vol_20ma"] + 1e-12)
        f["OBV"]        = (np.sign(c.diff()) * v).fillna(0).cumsum()
        f["OBV_slope"]  = f["OBV"].diff(5) / (f["OBV"].abs() + 1e-12)

    # ── Price pattern features ──
    f["Gap"]          = o / c.shift(1) - 1    # overnight gap
    f["IntraBar_Ret"] = c / o - 1              # intraday move
    f["HL_ratio"]     = (h - l) / (c + 1e-12) # bar range normalized
    f["ClosePos"]     = (c - l) / (h - l + 1e-12)  # close position in range

    # ── Mean reversion ──
    f["Zscore_10"]  = (c - c.rolling(10).mean()) / (c.rolling(10).std(ddof=0) + 1e-12)
    f["Zscore_20"]  = (c - c.rolling(20).mean()) / (c.rolling(20).std(ddof=0) + 1e-12)
    f["Zscore_slope"] = f["Zscore_20"].diff(3)

    # ── Trend strength (ADX-like) ──
    f["SlopeClose"]   = c.rolling(lookback).apply(slope, raw=True)
    f["Slope_norm"]   = f["SlopeClose"] / (c + 1e-12)

    # Drop EMA columns (high-cardinality raw levels, keep the derived ratios)
    f.drop(columns=[f"EMA_{sp}" for sp in [9, 21, 50]], errors="ignore", inplace=True)

    f = f.iloc[max(lookback-1, 50):].copy()

    if exog is not None and not exog.empty:
        f = f.join(exog, how="left").ffill()

    f["FutureRetExec"] = f["Open"].shift(-int(horizon)) / f["Open"].shift(-1) - 1
    return f


# ─────────────────────────────────────────────────────────────
# ★  ENHANCED Model Builder
# ─────────────────────────────────────────────────────────────
def _build_base_clf(model_type: str, params: dict, n_features: int):
    """Construct base classifier based on user selection."""
    n_est = params.get("n_estimators", 120)
    lr    = params.get("learning_rate", 0.08)
    depth = params.get("max_depth", 4)
    seed  = params.get("random_state", 42)

    if model_type == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=n_est, learning_rate=lr,
            max_depth=depth, random_state=seed,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, min_samples_leaf=5,
        )
    elif model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            random_state=seed, n_jobs=-1,
            max_features="sqrt", min_samples_leaf=3,
        )
    elif model_type == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=n_est, max_depth=depth,
            random_state=seed, n_jobs=-1,
            max_features="sqrt", min_samples_leaf=3,
        )
    elif model_type.startswith("Ensemble"):
        # Voting ensemble: HistGBT + RF + ExtraTrees
        hgbt = HistGradientBoostingClassifier(
            max_iter=n_est, learning_rate=lr, max_depth=depth,
            random_state=seed, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, min_samples_leaf=5,
        )
        rf = RandomForestClassifier(
            n_estimators=max(50, n_est//2), max_depth=depth,
            random_state=seed, n_jobs=-1, max_features="sqrt",
        )
        et = ExtraTreesClassifier(
            n_estimators=max(50, n_est//2), max_depth=depth,
            random_state=seed+1, n_jobs=-1, max_features="sqrt",
        )
        return VotingClassifier(
            estimators=[("hgbt", hgbt), ("rf", rf), ("et", et)],
            voting="soft", weights=[2, 1, 1],
        )
    else:  # GradientBoosting (classic)
        return GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=lr,
            max_depth=depth, random_state=seed,
            subsample=0.8, min_samples_leaf=5,
            max_features="sqrt",
        )


def _build_pipeline(model_type: str, params: dict, calibrate: bool,
                    use_feat_select: bool, n_features: int) -> Pipeline:
    base = _build_base_clf(model_type, params, n_features)

    if calibrate and not model_type.startswith("Ensemble"):
        # CalibratedClassifierCV needs enough samples; fallback gracefully
        clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    elif calibrate and model_type.startswith("Ensemble"):
        clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    else:
        clf = base

    steps = [("imp", SimpleImputer(strategy="median"))]
    if use_feat_select:
        # Use a quick RF for feature importance selection
        sel_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=99, n_jobs=-1)
        steps.append(("sel", SelectFromModel(sel_rf, threshold="median")))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ─────────────────────────────────────────────────────────────
# Feature column registry
# ─────────────────────────────────────────────────────────────
BASE_FEATURE_COLS = [
    "Range","SlopeHigh","SlopeLow",
    "Ret_3d","Ret_5d","Ret_10d","Ret_21d",
    "RetLog_3d","RetLog_5d","RetLog_10d","Ret_vs_ATH",
    "RSI_7","RSI_14","RSI_21","RSI_slope",
    "MACD_line","MACD_signal","MACD_hist","MACD_hist_slope",
    "BollPctB_10","BollWidth_10","BollPctB_20","BollWidth_20","BollWidth_trend",
    "ATR_rel","Vol_5d","Vol_20d","Vol_regime",
    "Price_vs_EMA_9","Price_vs_EMA_21","Price_vs_EMA_50",
    "EMA_9_21_cross","EMA_21_50_cross",
    "Stoch_K","Stoch_D","Stoch_K_slope",
    "Vol_ratio","OBV_slope",
    "Gap","IntraBar_Ret","HL_ratio","ClosePos",
    "Zscore_10","Zscore_20","Zscore_slope",
    "Slope_norm",
]
OPT_FEATURE_COLS = ["PCR_vol","PCR_oi","VOI_call","VOI_put",
                     "IV_skew_p_minus_c","VOL_tot","OI_tot"]


@st.cache_data(show_spinner=False, ttl=3600)
def build_feature_cache(df, lookback, horizon, threshold):
    feat = make_features(df, lookback, horizon)
    hist = feat.iloc[:-1].dropna(subset=["FutureRetExec"]).copy()
    if hist.empty: return None, None
    hist["Target"] = (hist["FutureRetExec"] > threshold).astype(int)
    return feat, hist


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────
def backtest_next_open(df, entry_thr, exit_thr, commission, slippage_bps,
                       init_cap, pos_frac, min_hold_days=0, cooldown_days=0):
    df = df.copy(); n = len(df)
    if n < 2: raise ValueError("Zu wenige Datenpunkte.")
    cash_g=init_cap; cash_n=init_cap; shares=0.0; in_pos=False
    cb_g=0.0; cb_n=0.0; last_ei=None; last_xi=None
    eq_g, eq_n, trades, cum = [], [], [], 0.0

    for i in range(n):
        if i > 0:
            ot  = float(df["Open"].iloc[i])
            sb  = ot*(1+slippage_bps/10000.0); ss=ot*(1-slippage_bps/10000.0)
            pp  = float(df["SignalProb"].iloc[i-1]); de=df.index[i]
            cool= True
            if (not in_pos) and cooldown_days>0 and last_xi is not None:
                cool = (i-last_xi)>=int(cooldown_days)
            if (not in_pos) and (pp>entry_thr) and cool:
                inv=cash_n*float(pos_frac); fee=inv*float(commission)
                sh=max((inv-fee)/sb,0.0)
                if sh>0 and (sh*sb+fee)<=cash_n+1e-6:
                    shares=sh; cb_g=sh*sb; cb_n=sh*sb+fee
                    cash_g-=cb_g; cash_n-=cb_n; in_pos=True; last_ei=i
                    trades.append({"Date":de,"Typ":"Entry","Price":round(sb,4),"Shares":round(sh,4),
                                   "Gross P&L":0.0,"Fees":round(fee,2),"Net P&L":0.0,
                                   "kum P&L":round(cum,2),"Prob":round(pp,4),"HoldDays":np.nan})
            elif in_pos and pp<exit_thr:
                held=(i-last_ei) if last_ei is not None else 0
                if not (int(min_hold_days)>0 and held<int(min_hold_days)):
                    gv=shares*ss; fe=gv*float(commission)
                    pnl_g=gv-cb_g; pnl_n=(gv-fe)-cb_n
                    cash_g+=gv; cash_n+=(gv-fe)
                    in_pos=False; shares=0.0; cb_g=0.0; cb_n=0.0; cum+=pnl_n
                    trades.append({"Date":de,"Typ":"Exit","Price":round(ss,4),"Shares":0.0,
                                   "Gross P&L":round(pnl_g,2),"Fees":round(fe,2),"Net P&L":round(pnl_n,2),
                                   "kum P&L":round(cum,2),"Prob":round(pp,4),"HoldDays":int(held)})
                    last_xi=i; last_ei=None
        ct=float(df["Close"].iloc[i])
        eq_g.append(cash_g+(shares*ct if in_pos else 0.0))
        eq_n.append(cash_n+(shares*ct if in_pos else 0.0))

    df_bt=df.copy(); df_bt["Equity_Gross"]=eq_g; df_bt["Equity_Net"]=eq_n
    return df_bt, trades


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────
def _cagr(v):
    if len(v)<2: return np.nan
    dt0,dt1=pd.to_datetime(v.index[0]),pd.to_datetime(v.index[-1])
    yrs=max((dt1-dt0).days/365.25,1e-9)
    return (v.iloc[-1]/v.iloc[0])**(1/yrs)-1

def _sortino(rets):
    if rets.empty: return np.nan
    mean=rets.mean()*252; down=rets[rets<0]; dd=down.std()*np.sqrt(252) if len(down) else np.nan
    return mean/dd if dd and np.isfinite(dd) and dd>0 else np.nan

def _winrate(trades):
    if not trades: return np.nan
    pnl=[]; e=None
    for ev in trades:
        if ev["Typ"]=="Entry": e=ev
        elif ev["Typ"]=="Exit" and e is not None: pnl.append(float(ev.get("Net P&L",0.0))); e=None
    return float((np.array(pnl,float)>0).mean()) if pnl else np.nan

def compute_performance(df_bt, trades, init_cap):
    net_ret=(df_bt["Equity_Net"].iloc[-1]/init_cap-1)*100
    rets=df_bt["Equity_Net"].pct_change().dropna()
    vol=rets.std()*sqrt(252)*100
    sharpe=(rets.mean()*sqrt(252))/(rets.std()+1e-12)
    dd=(df_bt["Equity_Net"]-df_bt["Equity_Net"].cummax())/df_bt["Equity_Net"].cummax()
    max_dd=dd.min()*100
    calmar=(net_ret/100)/abs(max_dd/100) if max_dd<0 else np.nan
    return {
        "Strategy Net (%)":round(float(net_ret),2),
        "Strategy Gross (%)":round(float((df_bt["Equity_Gross"].iloc[-1]/init_cap-1)*100),2),
        "Buy & Hold Net (%)":round(float((df_bt["Close"].iloc[-1]/df_bt["Close"].iloc[0]-1)*100),2),
        "Volatility (%)":round(float(vol),2),"Sharpe-Ratio":round(float(sharpe),2),
        "Sortino-Ratio":round(float(_sortino(rets)),2) if np.isfinite(_sortino(rets)) else np.nan,
        "Max Drawdown (%)":round(float(max_dd),2),
        "Calmar-Ratio":round(float(calmar),2) if np.isfinite(calmar) else np.nan,
        "Fees (€)":round(float(sum(t.get("Fees",0.) for t in trades)),2),
        "Phase":"Open" if trades and trades[-1]["Typ"]=="Entry" else "Flat",
        "Number of Trades":int(sum(1 for t in trades if t["Typ"]=="Exit")),
        "Net P&L (€)":round(float(df_bt["Equity_Net"].iloc[-1]-init_cap),2),
        "CAGR (%)":round(100*float(_cagr(df_bt["Equity_Net"])) if np.isfinite(_cagr(df_bt["Equity_Net"])) else np.nan,2),
        "Winrate (%)":round(100*float(_winrate(trades)) if np.isfinite(_winrate(trades)) else np.nan,2),
        "InitCap (€)":float(init_cap),
    }

def compute_round_trips(all_trades):
    rows=[]
    for tk,tr in all_trades.items():
        name=get_ticker_name(tk); ce=None
        for ev in tr:
            if ev["Typ"]=="Entry": ce=ev
            elif ev["Typ"]=="Exit" and ce is not None:
                ed=pd.to_datetime(ce["Date"]); xd=pd.to_datetime(ev["Date"])
                sh=float(ce.get("Shares",0.)); ep=float(ce.get("Price",np.nan))
                xp=float(ev.get("Price",np.nan)); fe=float(ce.get("Fees",0.)); fx=float(ev.get("Fees",0.))
                pnl=float(ev.get("Net P&L",0.)); cost=sh*ep+fe
                rows.append({"Ticker":tk,"Name":name,"Entry Date":ed,"Exit Date":xd,
                             "Hold (days)":(xd-ed).days,"Entry Prob":ce.get("Prob",np.nan),
                             "Exit Prob":ev.get("Prob",np.nan),"Shares":round(sh,4),
                             "Entry Price":round(ep,4),"Exit Price":round(xp,4),
                             "PnL Net (€)":round(pnl,2),"Fees (€)":round(fe+fx,2),
                             "Return (%)":round(pnl/cost*100,2) if cost else np.nan}); ce=None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Forecast helpers
# ─────────────────────────────────────────────────────────────
def estimate_expected_return(feat, forecast_days, threshold):
    if feat is None or feat.empty or "Open" not in feat.columns or "SignalProb" not in feat.columns: return {}
    fr=feat["Open"].shift(-int(forecast_days))/feat["Open"].shift(-1)-1
    tmp=pd.DataFrame({"FutureRet":fr}).dropna()
    if tmp.empty: return {}
    tmp["T"]=(tmp["FutureRet"]>float(threshold)).astype(int)
    mu1=float(tmp.loc[tmp["T"]==1,"FutureRet"].mean()) if tmp["T"].sum()>0 else 0.0
    mu0=float(tmp.loc[tmp["T"]==0,"FutureRet"].mean()) if (1-tmp["T"]).sum()>0 else 0.0
    p=float(pd.to_numeric(feat["SignalProb"].iloc[-1],errors="coerce"))
    if not np.isfinite(p): p=0.5
    return {"mu1":mu1,"mu0":mu0,"p":p,"exp_ret":p*mu1+(1-p)*mu0}

def _ensure_psd(cov, eps=1e-12):
    cov=(cov+cov.T)/2
    try:
        w,v=np.linalg.eigh(cov); return (v*np.maximum(w,eps))@v.T
    except Exception: return np.diag(np.maximum(np.diag(cov),eps))

def portfolio_mc(exp_rets, cov, nav0, sims=1500, seed=42):
    tickers=exp_rets.index.tolist(); cov=cov.reindex(index=tickers,columns=tickers).fillna(0.)
    w=np.ones(len(tickers))/max(len(tickers),1)
    rng=np.random.default_rng(int(seed))
    draws=rng.multivariate_normal(exp_rets.values,_ensure_psd(cov.values),size=int(sims))
    pr=draws@w; nv=nav0*(1+pr)
    q=np.quantile(pr,[.05,.5,.95]); qn=np.quantile(nv,[.05,.5,.95])
    return {"q05":float(q[0]),"q50":float(q[1]),"q95":float(q[2]),
            "nq05":float(qn[0]),"nq50":float(qn[1]),"nq95":float(qn[2]),
            "port_rets":pr,"nav_paths":nv}


# ─────────────────────────────────────────────────────────────
# ★  ENHANCED Model Training
# ─────────────────────────────────────────────────────────────
def make_features_and_train(df, lookback, horizon, threshold, model_params,
                             entry_prob, exit_prob, init_capital, pos_frac,
                             min_hold_days=0, cooldown_days=0, exog_df=None,
                             walk_forward=False, wf_min_train=120,
                             model_type="Ensemble (HistGBT+RF+ET)",
                             calibrate=True, use_feat_sel=False):
    feat = make_features(df, lookback, horizon, exog=exog_df)
    hist = feat.iloc[:-1].dropna(subset=["FutureRetExec"]).copy()
    if len(hist) < 30: raise ValueError("Zu wenige Datenpunkte.")

    x_cols = [c for c in BASE_FEATURE_COLS if c in feat.columns]
    x_cols += [c for c in OPT_FEATURE_COLS if c in feat.columns]
    hist["Target"] = (hist["FutureRetExec"] > threshold).astype(int)

    def make_pipe():
        return _build_pipeline(model_type, model_params, calibrate, use_feat_sel, len(x_cols))

    # ── Feature importance fallback for HistGBT / GBT ──
    def get_feat_imp(pipe, x_cols):
        try:
            clf = pipe.named_steps.get("clf")
            if clf is None: return None
            # unwrap calibrated
            if hasattr(clf, "base_estimator"): clf = clf.base_estimator
            if hasattr(clf, "estimators_") or hasattr(clf, "feature_importances_"):
                return dict(zip(x_cols, clf.feature_importances_))
        except Exception: pass
        return None

    if hist["Target"].nunique() < 2:
        feat["SignalProb"] = 0.5; feat_imp = None
    elif not walk_forward:
        X_tr = hist[x_cols].values; y_tr = hist["Target"].values
        pipe = make_pipe()
        try:
            pipe.fit(X_tr, y_tr)
        except Exception:
            # fallback to simple GBT
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("clf", GradientBoostingClassifier(**model_params))])
            pipe.fit(X_tr, y_tr)
        feat["SignalProb"] = pipe.predict_proba(feat[x_cols].values)[:,1]
        feat_imp = get_feat_imp(pipe, x_cols)
    else:
        probs = np.full(len(feat), np.nan)
        mt = max(int(wf_min_train), lookback+horizon+10)
        for t in range(mt, len(feat)):
            tr = feat.iloc[:t].dropna(subset=["FutureRetExec"]).copy()
            if len(tr) < mt: continue
            tr["Target"] = (tr["FutureRetExec"] > threshold).astype(int)
            if tr["Target"].nunique() < 2: continue
            p = make_pipe()
            try: p.fit(tr[x_cols].values, tr["Target"].values)
            except Exception:
                p = Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("clf", GradientBoostingClassifier(**model_params))])
                p.fit(tr[x_cols].values, tr["Target"].values)
            probs[t] = p.predict_proba(feat[x_cols].iloc[[t]].values)[0,1]
        feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)
        feat_imp = None

    df_bt, trades = backtest_next_open(
        feat.iloc[:-1], entry_prob, exit_prob, COMMISSION, SLIPPAGE_BPS,
        init_capital, pos_frac, min_hold_days=int(min_hold_days), cooldown_days=int(cooldown_days)
    )
    perf = compute_performance(df_bt, trades, init_capital)
    return feat, df_bt, trades, perf, feat_imp


# ─────────────────────────────────────────────────────────────
# Chart builders  (light theme)
# ─────────────────────────────────────────────────────────────
def chart_price_signal(feat, trades, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.70, 0.30], vertical_spacing=0.03)

    if all(c in feat.columns for c in ["Open","High","Low","Close"]):
        fig.add_trace(go.Candlestick(
            x=feat.index, open=feat["Open"], high=feat["High"],
            low=feat["Low"], close=feat["Close"], name="OHLC", showlegend=False,
            increasing=dict(line=dict(color=THEME["green"],width=1),
                           fillcolor="rgba(22,163,74,0.20)"),
            decreasing=dict(line=dict(color=THEME["red"],width=1),
                           fillcolor="rgba(220,38,38,0.20)"),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=feat.index, y=feat["Close"], mode="lines",
                                  line=dict(color=THEME["accent1"],width=1.5), showlegend=False), row=1, col=1)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["Date"] = pd.to_datetime(tdf["Date"])
        for typ, sym, col in [("Entry","triangle-up", THEME["green"]),
                               ("Exit", "triangle-down", THEME["red"])]:
            sub = tdf[tdf["Typ"]==typ]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Date"], y=sub["Price"], mode="markers", name=typ,
                    marker=dict(symbol=sym, size=12, color=col,
                               line=dict(color="white",width=1.5)),
                    hovertemplate=f"<b>{typ}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ), row=1, col=1)

    prob = feat["SignalProb"]
    fig.add_trace(go.Scatter(
        x=feat.index, y=prob, mode="lines", name="Signal Prob",
        line=dict(color=THEME["accent4"], width=1.5),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
        hovertemplate="%{x|%Y-%m-%d}<br>P=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=ENTRY_PROB, row=2, col=1, line_color=THEME["green"],
                  line_dash="dash", line_width=1.5, opacity=0.7)
    fig.add_hline(y=EXIT_PROB,  row=2, col=1, line_color=THEME["red"],
                  line_dash="dash", line_width=1.5, opacity=0.7)

    _apply_theme(fig, 500)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Preis & Signal Probability",
                   font=dict(size=12, color=THEME["text_sec"])),
        xaxis_rangeslider_visible=False,
        yaxis2=dict(range=[0,1], title="P", tickformat=".2f"),
    )
    return fig


def chart_equity(df_bt, ticker, init_cap):
    eq = df_bt["Equity_Net"]
    bh = init_cap * df_bt["Close"] / df_bt["Close"].iloc[0]
    dd = (eq - eq.cummax()) / eq.cummax() * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, mode="lines", name="Strategie",
        line=dict(color=THEME["accent1"], width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        hovertemplate="%{x|%Y-%m-%d}  %{y:,.2f}€<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh, mode="lines", name="Buy & Hold",
        line=dict(color=THEME["muted"], width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}  B&H: %{y:,.2f}€<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=float(init_cap), row=1, col=1,
                  line_color=THEME["muted"], line_dash="dot",
                  line_width=1, opacity=0.4)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, mode="lines", name="Drawdown",
        line=dict(color=THEME["red"], width=1),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}  DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    _apply_theme(fig, 450)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Equity Kurve & Drawdown",
                   font=dict(size=12, color=THEME["text_sec"])),
        yaxis=dict(title="NAV (€)", tickformat=",.0f"),
        yaxis2=dict(title="DD %", tickformat=".1f"),
    )
    return fig


def chart_feature_importance(feat_imp: dict, ticker: str, top_n: int = 20):
    if not feat_imp: return None
    items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [i[0] for i in items][::-1]
    vals  = [i[1] for i in items][::-1]
    colors = [THEME["accent1"] if v >= np.median(vals) else THEME["accent3"] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    _apply_theme(fig, max(350, top_n*22))
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Feature Importance (Top {top_n})",
                   font=dict(size=12, color=THEME["text_sec"])),
        xaxis_title="Importance", yaxis_title="",
        showlegend=False,
    )
    return fig


def chart_intraday(intra, ticker, tdf, chart_type, interval):
    fig = go.Figure()
    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(
            x=intra.index, open=intra["Open"], high=intra["High"],
            low=intra["Low"], close=intra["Close"], showlegend=False,
            increasing=dict(line=dict(color=THEME["green"],width=1),
                           fillcolor="rgba(22,163,74,0.18)"),
            decreasing=dict(line=dict(color=THEME["red"],width=1),
                           fillcolor="rgba(220,38,38,0.18)"),
        ))
    else:
        fig.add_trace(go.Scatter(x=intra.index, y=intra["Close"], mode="lines",
                                  line=dict(color=THEME["accent1"],width=1.5), showlegend=False))

    for _, ds in intra.groupby(intra.index.normalize()):
        fig.add_vline(x=ds.index.min(), line_width=1, line_dash="dot",
                      line_color=THEME["border_med"], opacity=0.6)

    if not tdf.empty:
        tdf2 = tdf.copy(); tdf2["Date"] = pd.to_datetime(tdf2["Date"])
        last_days = set(intra.index.normalize())
        ev = tdf2[tdf2["Date"].dt.normalize().isin(last_days)]
        for typ, col, sym in [("Entry",THEME["green"],"triangle-up"),
                               ("Exit", THEME["red"],  "triangle-down")]:
            xs,ys=[],[]
            for d,ds in intra.groupby(intra.index.normalize()):
                h_ = ev[(ev["Typ"]==typ)&(ev["Date"].dt.normalize()==d)]
                if h_.empty: continue
                xs.append(ds.index.min()); ys.append(float(h_["Price"].iloc[-1]))
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=typ,
                    marker=dict(symbol=sym,size=12,color=col,line=dict(color="white",width=1.5))))

    _apply_theme(fig, 400)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Intraday 5d ({interval})",
                   font=dict(size=12, color=THEME["text_sec"])),
        xaxis_rangeslider_visible=False,
    )
    return fig


def chart_corr(corr):
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        text=corr.round(2).astype(str).values, texttemplate="%{text}",
        colorscale=[
            [0.0, "#EF4444"], [0.4, "#FEF3C7"],
            [0.5, "#F8FAFC"],
            [0.6, "#DBEAFE"], [1.0, "#2563EB"],
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="ρ", thickness=12,
                      tickfont=dict(family="'DM Mono', monospace", size=9, color=THEME["text_sec"])),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>",
        textfont=dict(size=10 if corr.shape[0]<=8 else 8, color=THEME["text"]),
    ))
    n = corr.shape[0]
    _apply_theme(fig, max(380, n*46))
    fig.update_layout(
        title=dict(text="Korrelationsmatrix  ·  Close Returns",
                   font=dict(size=12, color=THEME["text_sec"])),
        xaxis=dict(tickangle=-30, showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        plot_bgcolor=THEME["bg_card"],
    )
    return fig


def chart_portfolio_nav(nav):
    fig = go.Figure(go.Scatter(
        x=nav.index, y=nav.values, mode="lines",
        line=dict(color=THEME["accent1"], width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="%{x|%Y-%m-%d}  %{y:,.0f}€<extra></extra>",
        name="NAV",
    ))
    _apply_theme(fig, 360)
    fig.update_layout(
        title=dict(text="Portfolio NAV  ·  Equal-Weight",
                   font=dict(size=12, color=THEME["text_sec"])),
        yaxis=dict(title="NAV (€)", tickformat=",.0f"),
        showlegend=False,
    )
    return fig


def chart_mc_histogram(port_rets, q05, q50, q95, forecast_days, sims):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_rets*100, nbinsx=50,
        marker=dict(
            color=["rgba(22,163,74,0.55)" if v>=0 else "rgba(220,38,38,0.55)"
                   for v in port_rets],
            line=dict(color="rgba(255,255,255,0.4)", width=0.5),
        ),
        showlegend=False,
        hovertemplate="Return: %{x:.2f}%<br>n: %{y}<extra></extra>",
    ))
    for val, lbl, col in [(q05*100,"5%",THEME["red"]),
                           (q50*100,"50%",THEME["accent3"]),
                           (q95*100,"95%",THEME["green"])]:
        fig.add_vline(x=val, line_dash="dash", line_color=col, line_width=2, opacity=0.9,
                      annotation_text=f"  {lbl}: {val:.2f}%",
                      annotation_font=dict(color=col, size=10,
                                           family="'DM Mono', monospace"),
                      annotation_position="top right")
    _apply_theme(fig, 360)
    fig.update_layout(
        title=dict(text=f"MC Portfolio Returns  ·  {forecast_days}d  ·  {sims} Sim.",
                   font=dict(size=12, color=THEME["text_sec"])),
        xaxis_title="Return (%)", yaxis_title="Häufigkeit", bargap=0.04,
    )
    return fig


def chart_histogram(data, xlabel, title, bins):
    mean_v = float(data.mean()) if len(data) else np.nan
    fig = go.Figure(go.Histogram(
        x=data, nbinsx=bins,
        marker=dict(
            color=["rgba(22,163,74,0.55)" if v>=0 else "rgba(220,38,38,0.55)"
                   for v in data],
            line=dict(color="rgba(255,255,255,0.4)", width=0.5),
        ),
        showlegend=False,
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>n: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=THEME["muted"], line_width=1.5, opacity=0.5)
    if np.isfinite(mean_v):
        fig.add_vline(x=mean_v, line_dash="dot", line_color=THEME["accent2"], line_width=2,
                      annotation_text=f"  Ø {mean_v:.2f}",
                      annotation_font=dict(color=THEME["accent2"],size=10,
                                           family="'DM Mono', monospace"),
                      annotation_position="top right")
    _apply_theme(fig, 340)
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=THEME["text_sec"])),
        xaxis_title=xlabel, yaxis_title="n", bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────
section("PARAMETER-OPTIMIERUNG")
with st.expander("⚙  Random-Search Optimizer  ·  Walk-Forward Light", expanded=False):
    oc1, oc2 = st.columns(2)
    with oc1:
        n_trials       = st.number_input("Trials", 10, 1000, 80, step=10)
        seed_opt       = st.number_input("Seed", 0, 10000, 42)
        lambda_trades  = st.number_input("Penalty λ/Trade", 0.0, 1.0, 0.02, step=0.005)
        min_trades_req = st.number_input("Min. Trades", 0, 10000, 5, step=1)
    with oc2:
        lb_lo, lb_hi   = st.slider("Lookback", 10, 252, (30,120), step=5)
        hz_lo, hz_hi   = st.slider("Horizon", 1, 10, (3,8))
        thr_lo, thr_hi = st.slider("Threshold", 0.0, 0.10, (0.035,0.10), step=0.005, format="%.3f")
        en_lo,  en_hi  = st.slider("Entry Prob", 0.0, 1.0, (0.55,0.85), step=0.01)
        ex_lo,  ex_hi  = st.slider("Exit Prob",  0.0, 1.0, (0.30,0.60), step=0.01)

    @st.cache_data(show_spinner=False)
    def _opt_prices(tickers, start, end, use_tail, interval, fb, ek, moc):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fb, ek, moc)[0]

    if st.button("🚀 Suche starten", type="primary", use_container_width=True):
        import random
        from collections import Counter
        rng_opt = random.Random(int(seed_opt))
        pm = _opt_prices(tuple(TICKERS), str(START_DATE), str(END_DATE),
                         use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min))

        min_len = max(120, int(wf_min_train)+40)
        feasible = {tk: df.copy() for tk,df in (pm or {}).items()
                    if isinstance(df, pd.DataFrame) and len(df)>=min_len}
        if len(feasible)<2:
            st.error("Zu wenige Ticker nach Prefilter."); st.stop()

        rows_o, best_o = [], None
        prog_o = st.progress(0.0); status_o = st.empty(); err_c = Counter()

        for trial in range(int(n_trials)):
            p = dict(lookback=rng_opt.randrange(lb_lo,lb_hi+1,5),
                     horizon=rng_opt.randrange(hz_lo,hz_hi+1),
                     thresh=rng_opt.uniform(thr_lo,thr_hi),
                     entry=rng_opt.uniform(en_lo,en_hi),
                     exit=rng_opt.uniform(ex_lo,ex_hi))
            if p["exit"]>=p["entry"]: prog_o.progress((trial+1)/int(n_trials)); continue
            sharps_o, trad_o, ok_t = [], 0, 0
            for tk, df0 in feasible.items():
                if len(df0)<max(80,p["lookback"]+p["horizon"]+40): continue
                try:
                    feat_o, hist_o = build_feature_cache(df0, int(p["lookback"]),
                                                          int(p["horizon"]), float(p["thresh"]))
                    if feat_o is None or hist_o is None or hist_o["Target"].nunique()<2:
                        raise ValueError("degenerate")
                    x_c = [c for c in BASE_FEATURE_COLS if c in feat_o.columns]
                    n_h=len(hist_o); split=max(int(wf_min_train),int(0.6*n_h))
                    if n_h-split<30: raise ValueError("OOS too short")
                    pp_opt = Pipeline([("i",SimpleImputer(strategy="median")),
                                       ("m",HistGradientBoostingClassifier(
                                           max_iter=80, max_depth=4, random_state=42,
                                           early_stopping=False))])
                    pp_opt.fit(hist_o.iloc[:split][x_c].values, hist_o.iloc[:split]["Target"].values)
                    feat_o["SignalProb"] = pp_opt.predict_proba(feat_o[x_c].values)[:,1]
                    df_bt_o, tr_o = backtest_next_open(feat_o.iloc[:-1],
                                                        float(p["entry"]),float(p["exit"]),
                                                        COMMISSION,SLIPPAGE_BPS,
                                                        float(INIT_CAP_PER_TICKER),float(POS_FRAC),
                                                        int(MIN_HOLD_DAYS),int(COOLDOWN_DAYS))
                    oos_idx  = hist_o.index[split:]
                    oos_eq   = df_bt_o.loc[df_bt_o.index.intersection(oos_idx)]["Equity_Net"]
                    if len(oos_eq)<30: raise ValueError("OOS too short")
                    r=oos_eq.pct_change().dropna()
                    if r.empty: raise ValueError("empty rets")
                    sharps_o.append(float((r.mean()/(r.std(ddof=0)+1e-12))*np.sqrt(252)))
                    trad_o+=int(sum(1 for t in tr_o if t["Typ"]=="Exit")); ok_t+=1
                except Exception as e: err_c[str(e)[:80]]+=1
            if ok_t<2 or not sharps_o: prog_o.progress((trial+1)/int(n_trials)); continue
            sh_med=float(np.nanmedian(sharps_o))
            if not np.isfinite(sh_med) or trad_o<int(min_trades_req):
                prog_o.progress((trial+1)/int(n_trials)); continue
            score=sh_med-float(lambda_trades)*(trad_o/max(1,ok_t))
            rec=dict(trial=trial,score=score,sharpe_med=sh_med,trades=trad_o,ok_tickers=ok_t,**p)
            rows_o.append(rec)
            if best_o is None or score>best_o["score"]: best_o=rec
            if (trial+1)%10==0: status_o.caption(f"Trial {trial+1}/{n_trials} · Best: {best_o['score']:.3f}")
            prog_o.progress((trial+1)/int(n_trials))

        if not rows_o:
            st.error("Keine gültigen Ergebnisse.")
            if err_c: st.dataframe(pd.DataFrame(err_c.most_common(10),columns=["Error","Count"]))
        else:
            df_res_o=pd.DataFrame(rows_o).sort_values("score",ascending=False)
            st.success(f"✅ Best Score: **{best_o['score']:.3f}** · Sharpe: **{best_o['sharpe_med']:.2f}** · Trades: **{best_o['trades']}**")
            bc=st.columns(5)
            bc[0].metric("Lookback",   int(best_o["lookback"]))
            bc[1].metric("Horizon",    int(best_o["horizon"]))
            bc[2].metric("Threshold",  f"{best_o['thresh']:.3f}")
            bc[3].metric("Entry Prob", f"{best_o['entry']:.2f}")
            bc[4].metric("Exit Prob",  f"{best_o['exit']:.2f}")
            st.dataframe(df_res_o.head(25), use_container_width=True)
            st.download_button("⬇ Optimizer CSV", to_csv_eu(df_res_o),
                               file_name="optimizer_results.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
  padding: 32px 0 12px;
  border-bottom: 2px solid {THEME['border']};
  margin-bottom: 32px;
  display: flex;
  align-items: flex-end;
  gap: 20px;
">
  <div>
    <div class="nexus-header">📈 NEXUS</div>
    <div class="nexus-sub">Signal Strategy v3 · Ensemble AI · Walk-Forward OOS · Portfolio MC</div>
  </div>
  <div style="margin-left:auto; display:flex; gap:8px; align-items:center; padding-bottom:4px;">
    <span class="model-badge">🧠 {model_type}</span>
    {'<span class="model-badge">📐 Kalibriert</span>' if use_calibration else ''}
    {'<span class="model-badge">🔍 FeatSel</span>' if use_feat_select else ''}
  </div>
</div>
""", unsafe_allow_html=True)

results     = []
all_trades: Dict[str, List[dict]] = {}
all_dfs:    Dict[str, pd.DataFrame] = {}
all_feat:   Dict[str, pd.DataFrame] = {}
all_feat_imp: Dict[str, dict] = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
)

options_live: Dict[str, pd.DataFrame] = {}
if use_chain_live:
    with st.spinner("📊 Optionsketten einlesen …"):
        po=st.progress(0.0); tks=list(price_map.keys())
        for i,tk in enumerate(tks):
            try:
                df_o=price_map[tk]
                if df_o is None or df_o.empty: continue
                ch=get_equity_chain_aggregates_for_today(tk,float(df_o["Close"].iloc[-1]),
                                                         atm_band_pct,int(n_expiries),int(max_days_to_exp))
                if not ch.empty: options_live[tk]=ch
            except Exception: pass
            finally: po.progress((i+1)/max(1,len(tks)))

live_forecasts_run: List[dict] = []
_decide = lambda p,en,ex: "Enter / Add" if p>en else ("Exit / Reduce" if p<ex else "Hold / No Trade")

# ─────────────────────────────────────────────────────────────
# Per-Ticker Loop
# ─────────────────────────────────────────────────────────────
for ticker in TICKERS:
    if ticker not in price_map: continue
    df   = price_map[ticker]
    meta = meta_map.get(ticker, {})
    name = get_ticker_name(ticker)

    with st.expander(f"📊  {ticker}  ·  {name}", expanded=False):
        try:
            ts  = df.index[-1]
            sfx = f" · intraday {meta['tail_ts'].strftime('%H:%M')}" if meta.get("tail_is_intraday") else ""
            col_info1, col_info2 = st.columns([3,1])
            with col_info1:
                st.caption(
                    f"🕐 **{ts.strftime('%Y-%m-%d %H:%M %Z')}**{sfx}  ·  "
                    f"{'🔄 Walk-Forward OOS' if use_walk_forward else '📊 In-Sample'}  ·  "
                    f"Target: ret > {THRESH:.3f} in {HORIZON}d"
                )
            with col_info2:
                st.caption(f"Bars: **{len(df)}** · Model: **{model_type.split('(')[0].strip()}**")

            exog_tk = None
            if use_chain_live and ticker in options_live and not options_live[ticker].empty:
                ch_e = options_live[ticker].copy()
                ch_e.index = [df.index[-1].normalize()]
                exog_tk = ch_e

            feat, df_bt, trades, metrics, feat_imp = make_features_and_train(
                df, int(LOOKBACK), int(HORIZON), float(THRESH), MODEL_PARAMS,
                float(ENTRY_PROB), float(EXIT_PROB),
                init_capital=float(INIT_CAP_PER_TICKER), pos_frac=float(POS_FRAC),
                min_hold_days=int(MIN_HOLD_DAYS), cooldown_days=int(COOLDOWN_DAYS),
                exog_df=exog_tk, walk_forward=bool(use_walk_forward),
                wf_min_train=int(wf_min_train),
                model_type=model_type, calibrate=bool(use_calibration),
                use_feat_sel=bool(use_feat_select),
            )
            metrics["Ticker"] = ticker
            results.append(metrics); all_trades[ticker]=trades
            all_dfs[ticker]=df_bt; all_feat[ticker]=feat
            if feat_imp: all_feat_imp[ticker]=feat_imp

            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            live_act   = _decide(live_prob, float(ENTRY_PROB), float(EXIT_PROB))
            row = {"AsOf": pd.Timestamp(feat.index[-1]).strftime("%Y-%m-%d %H:%M"),
                   "Ticker": ticker, "Name": name,
                   f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob,4),
                   "Action": live_act, "Close": round(live_close,4),
                   "Bar": "intraday" if meta.get("tail_is_intraday") else "daily"}
            if use_chain_live and exog_tk is not None:
                for col_n in ["PCR_vol","PCR_oi","VOI_call","VOI_put","IV_skew_p_minus_c","VOL_tot","OI_tot"]:
                    vv=exog_tk.iloc[-1].get(col_n,np.nan)
                    if pd.notna(vv): row[col_n]=round(float(vv),4)
            live_forecasts_run.append(row)

            # ── Signal badge ──
            if live_act=="Enter / Add":
                badge_cls, badge_lbl, badge_icon = "sig-enter","ENTER / ADD","▲"
            elif live_act=="Exit / Reduce":
                badge_cls, badge_lbl, badge_icon = "sig-exit","EXIT / REDUCE","▼"
            else:
                badge_cls, badge_lbl, badge_icon = "sig-hold","HOLD","◆"

            st.markdown(
                f'<span class="signal-badge {badge_cls}">'
                f'{badge_icon} {badge_lbl}'
                f'  <span style="opacity:0.7">P={live_prob:.3f}</span>'
                f'</span>',
                unsafe_allow_html=True
            )

            # ── KPIs ──
            mn = metrics
            phase_cls = "kpi-info" if mn["Phase"]=="Open" else ""
            kpi_row([
                ("Netto Return",  _pct(mn["Strategy Net (%)"]),
                 "kpi-pos" if mn["Strategy Net (%)"]>0 else "kpi-neg"),
                ("Buy & Hold",   _pct(mn["Buy & Hold Net (%)"]),
                 "kpi-pos" if mn["Buy & Hold Net (%)"]>0 else "kpi-neg"),
                ("Sharpe",       f"{mn['Sharpe-Ratio']:.2f}",               "kpi-info"),
                ("Sortino",      f"{mn['Sortino-Ratio']:.2f}" if np.isfinite(mn["Sortino-Ratio"]) else "–", "kpi-info"),
                ("Max Drawdown", _pct(mn["Max Drawdown (%)"]),               "kpi-neg"),
                ("CAGR",         f"{mn['CAGR (%)']:.1f}%" if np.isfinite(mn["CAGR (%)"]) else "–", "kpi-pos" if mn.get("CAGR (%)",0)>0 else ""),
                ("Winrate",      f"{mn['Winrate (%)']:.1f}%" if np.isfinite(mn["Winrate (%)"]) else "–", ""),
                ("Trades",       f"{mn['Number of Trades']}",                ""),
                ("Phase",        mn["Phase"],                                 phase_cls),
            ])

            # ── Charts row 1 ──
            cc1, cc2 = st.columns(2)
            tdf_loc = pd.DataFrame(trades)
            with cc1:
                st.plotly_chart(chart_price_signal(feat, trades, ticker),
                                use_container_width=True, config={"displayModeBar": False})
            intra = get_intraday_last_n_sessions(ticker, 5, 10, intraday_interval)
            with cc2:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar.")
                else:
                    st.plotly_chart(chart_intraday(intra, ticker, tdf_loc, intraday_chart_type, intraday_interval),
                                    use_container_width=True, config={"displayModeBar": False})

            # ── Equity chart ──
            st.plotly_chart(chart_equity(df_bt, ticker, float(INIT_CAP_PER_TICKER)),
                            use_container_width=True, config={"displayModeBar": False})

            # ── Feature Importance ──
            if feat_imp:
                with st.expander(f"🔍 Feature Importance  ·  {ticker}", expanded=False):
                    top_n_fi = st.slider("Top Features", 5, min(40, len(feat_imp)), 20,
                                         key=f"fi_n_{ticker}")
                    fig_fi = chart_feature_importance(feat_imp, ticker, top_n_fi)
                    if fig_fi:
                        st.plotly_chart(fig_fi, use_container_width=True,
                                        config={"displayModeBar": False})

            # ── Trade log ──
            with st.expander(f"🗒  Trade-Log  ·  {ticker}", expanded=False):
                if tdf_loc.empty:
                    st.info("Keine Trades im gewählten Zeitraum.")
                else:
                    td = tdf_loc.copy(); td["Ticker"]=ticker; td["Name"]=name
                    if "Date" in td.columns:
                        td["Date"] = pd.to_datetime(td["Date"]).dt.strftime("%d.%m.%Y")
                    td = td.rename(columns={"Prob":"Signal Prob","HoldDays":"Hold (days)",
                                            "Net P&L":"PnL","kum P&L":"CumPnL"})
                    desired = ["Ticker","Name","Date","Typ","Price","Shares",
                               "Signal Prob","Hold (days)","PnL","CumPnL","Fees"]
                    sc = [c for c in desired if c in td.columns]

                    def _rc(row):
                        t=str(row.get("Typ","")).lower()
                        if "entry" in t: return [f"color:{THEME['green']};font-weight:600"]*len(row)
                        if "exit"  in t: return [f"color:{THEME['red']};font-weight:600"]*len(row)
                        return [""]*len(row)

                    st.dataframe(td[sc].style.format(
                        {"Price":"{:.2f}","Shares":"{:.4f}","Signal Prob":"{:.4f}",
                         "PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}"}
                    ).apply(_rc,axis=1), use_container_width=True)
                    st.download_button(f"⬇ Trades {ticker}", to_csv_eu(td[sc]),
                                       file_name=f"trades_{ticker}.csv", mime="text/csv",
                                       key=f"dl_tr_{ticker}")

        except Exception as e:
            st.error(f"❌ {ticker}: {e}")
            import traceback; st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────
# Live Forecast Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (pd.DataFrame(live_forecasts_run)
               .drop_duplicates(subset=["Ticker"], keep="last")
               .sort_values(["AsOf","Ticker"]).reset_index(drop=True))
    live_df["Target_5d"] = (pd.to_numeric(live_df["Close"],errors="coerce")*(1+float(THRESH))).round(2)
    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"
    if prob_col not in live_df.columns:
        cand = [c for c in live_df.columns if c.startswith("P(") and c.endswith("d)")]
        if cand: prob_col = cand[0]

    section(f"LIVE FORECAST BOARD  ·  {HORIZON}d HORIZON")

    n_enter=(live_df["Action"]=="Enter / Add").sum()
    n_exit =(live_df["Action"]=="Exit / Reduce").sum()
    n_hold =len(live_df)-n_enter-n_exit
    kpi_row([
        ("Signale gesamt",f"{len(live_df)}", ""),
        ("▲ Enter",       f"{n_enter}",      "kpi-pos"),
        ("▼ Exit",        f"{n_exit}",       "kpi-neg"),
        ("◆ Hold",        f"{n_hold}",       ""),
    ])

    desired_lf = ["AsOf","Ticker","Name",prob_col,"Action","Close","Target_5d","Bar"]
    if use_chain_live:
        desired_lf = ["AsOf","Ticker","Name",prob_col,"Action",
                      "PCR_oi","PCR_vol","VOI_call","VOI_put","Close","Target_5d","Bar"]
    show_lf = [c for c in desired_lf if c in live_df.columns]

    def _board_color(row):
        a=str(row.get("Action","")).lower()
        if "enter" in a: return [f"background:rgba(22,163,74,0.07)"]*len(row)
        if "exit"  in a: return [f"background:rgba(220,38,38,0.07)"]*len(row)
        return [""]*len(row)

    fmt_lf={prob_col:"{:.4f}","Close":"{:.2f}","Target_5d":"{:.2f}"}
    for c in ["PCR_oi","PCR_vol","VOI_call","VOI_put"]:
        if c in live_df.columns: fmt_lf[c]="{:.3f}"

    st.dataframe(
        live_df[show_lf].style.format(fmt_lf).apply(_board_color,axis=1),
        use_container_width=True,
        height=min(600,40+35*len(live_df)),
    )
    st.download_button("⬇ Forecasts CSV", to_csv_eu(live_df),
                       file_name=f"live_forecasts_{HORIZON}d.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────
# Summary & Portfolio
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"]/float(INIT_CAP_PER_TICKER))*100
    if "Phase" in summary_df.columns:
        summary_df["Phase"] = (summary_df["Phase"].astype(str).str.strip().str.lower()
                               .map(lambda x: "Open" if x=="open" else ("Flat" if x=="flat" else x.capitalize())))

    total_net   = float(summary_df["Net P&L (€)"].sum())
    total_fees  = float(summary_df["Fees (€)"].sum())
    total_gross = total_net + total_fees
    total_cap   = float(INIT_CAP_PER_TICKER)*len(summary_df)

    section("PORTFOLIO SUMMARY")
    kpi_row([
        ("Netto P&L",     _eur(total_net),                                            "kpi-pos" if total_net>0 else "kpi-neg"),
        ("Brutto P&L",    _eur(total_gross),                                          "kpi-pos" if total_gross>0 else "kpi-neg"),
        ("Trading Costs", f"–{total_fees:,.2f}€",                                     "kpi-neg"),
        ("Trades total",  f"{int(summary_df['Number of Trades'].sum())}",              ""),
        ("Ø CAGR",        f"{summary_df['CAGR (%)'].dropna().mean():.2f}%"
                          if "CAGR (%)" in summary_df else "–",                        "kpi-info"),
        ("Ø Winrate",     f"{summary_df['Winrate (%)'].dropna().mean():.1f}%"
                          if "Winrate (%)" in summary_df else "–",                     ""),
        ("Ø Sharpe",      f"{summary_df['Sharpe-Ratio'].dropna().mean():.2f}"
                          if "Sharpe-Ratio" in summary_df else "–",                    "kpi-vio"),
    ])

    def _num_col(v):
        try:
            f_ = float(v)
            if f_ > 0: return f"color:{THEME['green']};font-weight:500"
            if f_ < 0: return f"color:{THEME['red']};font-weight:500"
        except: pass
        return ""

    def _phase_col(v):
        if str(v)=="Open":
            return f"background:rgba(37,99,235,0.08);color:{THEME['accent1']};border-radius:4px;padding:2px 8px;"
        if str(v)=="Flat":
            return f"background:rgba(148,163,184,0.10);color:{THEME['muted']};border-radius:4px;padding:2px 8px;"
        return ""

    pct_c = ["Strategy Net (%)","Strategy Gross (%)","Buy & Hold Net (%)","Net P&L (%)","CAGR (%)"]
    styled_s = (
        summary_df.style.format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}","Buy & Hold Net (%)":"{:.2f}",
            "Volatility (%)":"{:.2f}","Sharpe-Ratio":"{:.2f}","Sortino-Ratio":"{:.2f}",
            "Max Drawdown (%)":"{:.2f}","Calmar-Ratio":"{:.2f}","Fees (€)":"{:.2f}",
            "Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}","CAGR (%)":"{:.2f}",
            "Winrate (%)":"{:.2f}","InitCap (€)":"{:.0f}",
        })
        .applymap(_num_col, subset=[c for c in pct_c if c in summary_df.columns])
    )
    if "Phase" in summary_df.columns:
        styled_s = styled_s.applymap(_phase_col, subset=["Phase"])
    st.dataframe(styled_s, use_container_width=True)
    st.download_button("⬇ Summary CSV", to_csv_eu(summary_df.reset_index()),
                       file_name="strategy_summary.csv", mime="text/csv")

    # ── Open Positions ──────────────────────────────────────
    section("OFFENE POSITIONEN")
    open_positions = []
    for ticker, tr in all_trades.items():
        if tr and tr[-1]["Typ"]=="Entry":
            le=next(t for t in reversed(tr) if t["Typ"]=="Entry")
            lc=float(all_dfs[ticker]["Close"].iloc[-1])
            upnl=(lc-float(le["Price"]))*float(le["Shares"])
            open_positions.append({
                "Ticker":ticker,"Name":get_ticker_name(ticker),
                "Entry Date":pd.to_datetime(le["Date"]).strftime("%Y-%m-%d"),
                "Entry Price":round(float(le["Price"]),2),"Close":round(lc,2),
                "Signal Prob":round(float(all_feat[ticker]["SignalProb"].iloc[-1]),4),
                "uPnL (€)":round(upnl,2),
            })
    if open_positions:
        op_df=pd.DataFrame(open_positions).sort_values("Entry Date",ascending=False)
        def _upnl_c(v):
            try: return f"color:{THEME['green']};font-weight:500" if float(v)>=0 else f"color:{THEME['red']};font-weight:500"
            except: return ""
        st.dataframe(op_df.style.format(
            {"Entry Price":"{:.2f}","Close":"{:.2f}","Signal Prob":"{:.4f}","uPnL (€)":"{:.2f}"}
        ).applymap(_upnl_c,subset=["uPnL (€)"]),use_container_width=True)
        st.download_button("⬇ Open Positions",to_csv_eu(op_df),
                           file_name="open_positions.csv",mime="text/csv")
    else:
        st.info("✅ Keine offenen Positionen.")

    # ── Round Trips ─────────────────────────────────────────
    rt_df=compute_round_trips(all_trades)
    if not rt_df.empty:
        section("ROUND-TRIPS")
        rt_df["Entry Date"]=pd.to_datetime(rt_df["Entry Date"])
        rt_df["Exit Date"] =pd.to_datetime(rt_df["Exit Date"])

        ret=pd.to_numeric(rt_df.get("Return (%)"),errors="coerce").dropna()
        pnl=pd.to_numeric(rt_df.get("PnL Net (€)"),errors="coerce").dropna()
        kpi_row([
            ("Trades",   f"{len(ret)}",                                                ""),
            ("Winrate",  f"{100*(ret>0).mean():.1f}%" if len(ret) else "–",           "kpi-pos"),
            ("Ø Return", f"{ret.mean():.2f}%" if len(ret) else "–",                   "kpi-info"),
            ("Median",   f"{ret.median():.2f}%" if len(ret) else "–",                 ""),
            ("Std",      f"{ret.std():.2f}%" if len(ret) else "–",                    "kpi-warn"),
            ("Best",     f"{ret.max():.2f}%" if len(ret) else "–",                    "kpi-pos"),
            ("Worst",    f"{ret.min():.2f}%" if len(ret) else "–",                    "kpi-neg"),
        ])

        rt_disp=rt_df.copy()
        rt_disp["Entry Date"]=rt_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        rt_disp["Exit Date"] =rt_disp["Exit Date"].dt.strftime("%Y-%m-%d")
        if "Hold (days)" in rt_disp.columns:
            rt_disp["Hold (days)"]=rt_disp["Hold (days)"].round().astype("Int64")

        def _ret_c(v):
            try: return f"color:{THEME['green']};font-weight:600" if float(v)>=0 else f"color:{THEME['red']};font-weight:600"
            except: return ""
        st.dataframe(rt_disp.sort_values("Exit Date",ascending=False).style.format(
            {"Shares":"{:.4f}","Entry Price":"{:.2f}","Exit Price":"{:.2f}",
             "PnL Net (€)":"{:.2f}","Fees (€)":"{:.2f}","Return (%)":"{:.2f}",
             "Entry Prob":"{:.4f}","Exit Prob":"{:.4f}"}
        ).applymap(_ret_c,subset=["Return (%)","PnL Net (€)"]),use_container_width=True)
        st.download_button("⬇ Round-Trips CSV",to_csv_eu(rt_disp),
                           file_name="round_trips.csv",mime="text/csv")

        # Histograms
        section("RETURN-VERTEILUNG")
        bins_rt=st.slider("Bins",10,120,30,step=5,key="rt_bins")
        hc1,hc2=st.columns(2)
        with hc1:
            if not ret.empty:
                st.plotly_chart(chart_histogram(ret,"Return (%)","Trade Returns (%)",bins_rt),
                                use_container_width=True,config={"displayModeBar":False})
        with hc2:
            if not pnl.empty:
                st.plotly_chart(chart_histogram(pnl,"PnL Net (€)","Trade P&L (€)",bins_rt),
                                use_container_width=True,config={"displayModeBar":False})

    # ── Portfolio Equal-Weight ──────────────────────────────
    section("PORTFOLIO  ·  EQUAL-WEIGHT")
    price_series=[]
    for tk,dfbt in all_dfs.items():
        if isinstance(dfbt,pd.DataFrame) and "Close" in dfbt.columns and len(dfbt)>=2:
            s=dfbt["Close"].copy(); s.name=tk
            try:
                if getattr(s.index,"tz",None) is not None: s.index=s.index.tz_localize(None)
            except Exception: pass
            s.index=pd.to_datetime(s.index).normalize(); price_series.append(s)

    prices_port=pd.DataFrame(); port_ret=pd.Series(dtype=float)

    if len(price_series)<2:
        st.info("Mindestens 2 Ticker nötig für Portfolio-Analyse.")
    else:
        prices_port=pd.concat(price_series,axis=1,join="outer").sort_index()
        rets_ew=prices_port.pct_change()
        valid=rets_ew.notna().sum(axis=1)>=2
        rets2=rets_ew.loc[valid].copy()
        if not rets2.empty:
            w_row=rets2.notna().astype(float); w_row=w_row.div(w_row.sum(axis=1),axis=0)
            port_ret=(rets2.fillna(0.0)*w_row).sum(axis=1).dropna()

        if port_ret.empty:
            st.info("Portfolio-Returns leer.")
        else:
            ann_r=(1+port_ret).prod()**(252/len(port_ret))-1
            ann_v=port_ret.std(ddof=0)*np.sqrt(252)
            sh_p =(port_ret.mean()/(port_ret.std(ddof=0)+1e-12))*np.sqrt(252)
            nav0 =float(INIT_CAP_PER_TICKER)*len(summary_df)
            nav  =nav0*(1+port_ret).cumprod()
            max_dd_p=float(((nav/nav.cummax())-1).min())

            kpi_row([
                ("Return p.a.", f"{ann_r*100:.2f}%",   "kpi-pos" if ann_r>0 else "kpi-neg"),
                ("Vol p.a.",    f"{ann_v*100:.2f}%",   "kpi-warn"),
                ("Sharpe",      f"{sh_p:.2f}",          "kpi-info"),
                ("Max DD",      f"{max_dd_p*100:.2f}%", "kpi-neg"),
                ("NAV",         f"{nav.iloc[-1]:,.0f}€", "kpi-pos" if nav.iloc[-1]>=nav0 else "kpi-neg"),
            ])
            st.plotly_chart(chart_portfolio_nav(nav),
                            use_container_width=True,config={"displayModeBar":False})
            st.download_button("⬇ Portfolio Returns CSV",
                               to_csv_eu(pd.DataFrame({"Date":port_ret.index,"PortRet":port_ret.values})),
                               file_name="portfolio_returns.csv",mime="text/csv")

    # ── Korrelation ─────────────────────────────────────────
    if not prices_port.empty and prices_port.shape[1]>=2:
        section("KORRELATION  ·  CLOSE RETURNS")
        ka,kb,kc,kd=st.columns(4)
        freq_lbl  =ka.selectbox("Frequenz",  ["täglich","wöchentlich","monatlich"],index=0)
        corr_meth =kb.selectbox("Methode",   ["Pearson","Spearman","Kendall"],     index=0)
        min_obs_c =kc.slider("Min. Punkte",10,300,20,step=5)
        use_ffill_c=kd.checkbox("FFill Lücken",value=True)

        pc=prices_port.copy()
        if use_ffill_c: pc=pc.ffill()
        if freq_lbl=="wöchentlich": pc=pc.resample("W").last()
        elif freq_lbl=="monatlich": pc=pc.resample("M").last()
        rc=pc.pct_change().dropna(how="all")
        rc=rc[rc.notna().sum()[rc.notna().sum()>=int(min_obs_c)].index]

        if rc.shape[1]>=2:
            corr=rc.corr(method=corr_meth.lower())
            st.plotly_chart(chart_corr(corr),use_container_width=True,config={"displayModeBar":False})
            m=corr.values.copy(); np.fill_diagonal(m,np.nan); off=m[np.isfinite(m)]
            cov_c=rc.cov(); vols_c=np.sqrt(np.diag(cov_c.values)); n_c=len(vols_c)
            w_c=np.ones(n_c)/n_c; pv=float(w_c@cov_c.values@w_c)
            denom=float(np.sum((w_c[:,None]*w_c[None,:]*vols_c[:,None]*vols_c[None,:])))+1e-12
            diag_p=float(np.sum((w_c**2)*(vols_c**2))); ipc=(pv-diag_p)/denom
            kpi_row([
                ("Ø Paar-ρ",    f"{np.mean(off):.2f}"   if off.size else "–", ""),
                ("Median-ρ",    f"{np.median(off):.2f}" if off.size else "–", ""),
                ("σ(ρ)",        f"{np.std(off):.2f}"    if off.size else "–", ""),
                ("IPC (norm.)", f"{ipc:.2f}"            if np.isfinite(ipc) else "–", "kpi-info"),
            ])
            st.caption(f"Basis: {len(rc)} Zeitpunkte · {freq_lbl} · {corr_meth}")
            st.download_button("⬇ Korrelationsmatrix CSV",
                               to_csv_eu(corr.reset_index().rename(columns={"index":"Ticker"})),
                               file_name="correlation_matrix.csv",mime="text/csv")
        else:
            st.info("Zu wenig Overlap für Korrelationsmatrix.")

    # ── Portfolio Forecast (MC) ──────────────────────────────
    section(f"PORTFOLIO FORECAST  ·  {int(FORECAST_DAYS)}d  ·  MC={int(MC_SIMS)}")

    rows_fc=[]
    for tk,feat in all_feat.items():
        est=estimate_expected_return(feat,int(FORECAST_DAYS),float(THRESH))
        if not est: continue
        rows_fc.append({"Ticker":tk,"Name":get_ticker_name(tk),
                        "p (Prob)":est["p"],"μ1":est["mu1"],"μ0":est["mu0"],
                        f"E[r {int(FORECAST_DAYS)}d]":est["exp_ret"]})

    if not rows_fc:
        st.info("Forecast: Nicht genug Daten.")
    else:
        fc_df=pd.DataFrame(rows_fc).set_index("Ticker")
        ercol=f"E[r {int(FORECAST_DAYS)}d]"
        st.dataframe(fc_df.sort_values(ercol,ascending=False).style.format(
            {"p (Prob)":"{:.4f}","μ1":"{:.4f}","μ0":"{:.4f}",ercol:"{:.4f}"}
        ).applymap(_num_col,subset=[ercol]),use_container_width=True)

        exp_rets=fc_df[ercol].astype(float).dropna()
        exp_rets=(exp_rets.reindex(prices_port.columns.intersection(exp_rets.index)).dropna()
                  if not prices_port.empty else exp_rets)

        if len(exp_rets)<2:
            st.info("Für Portfolio-MC mind. 2 Ticker nötig.")
        else:
            if not prices_port.empty:
                dr=prices_port[exp_rets.index].pct_change().dropna(how="all")
                cov_h=dr.cov(min_periods=60)*float(FORECAST_DAYS)
            else:
                cov_h=pd.DataFrame(np.diag(np.full(len(exp_rets),(0.02)**2)),
                                   index=exp_rets.index,columns=exp_rets.index)

            nav0_fc=float(INIT_CAP_PER_TICKER)*len(summary_df)
            out_mc =portfolio_mc(exp_rets,cov_h,nav0_fc,sims=int(MC_SIMS),seed=42)

            kpi_row([
                ("E[Return EW]", f"{exp_rets.mean()*100:.2f}%",                                  "kpi-info"),
                ("5%-Quantil",   f"{out_mc['q05']*100:.2f}%  ·  {out_mc['nq05']:,.0f}€",         "kpi-neg"),
                ("Median",       f"{out_mc['q50']*100:.2f}%  ·  {out_mc['nq50']:,.0f}€",         ""),
                ("95%-Quantil",  f"{out_mc['q95']*100:.2f}%  ·  {out_mc['nq95']:,.0f}€",         "kpi-pos"),
            ])
            st.plotly_chart(
                chart_mc_histogram(out_mc["port_rets"],out_mc["q05"],out_mc["q50"],
                                   out_mc["q95"],int(FORECAST_DAYS),int(MC_SIMS)),
                use_container_width=True,config={"displayModeBar":False}
            )

else:
    st.warning("⚠ Keine Ergebnisse — Ticker & Datenabdeckung prüfen.")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
  margin-top:60px;
  padding:20px 0;
  border-top:2px solid {THEME['border']};
  text-align:center;
  color:{THEME['muted']};
  font-family:'DM Mono',monospace;
  font-size:10px;
  letter-spacing:0.12em;
  text-transform:uppercase;
">
  NEXUS SIGNAL STRATEGY v3.0
  · Ensemble AI (HistGBT + RF + ExtraTrees)
  · Platt Calibration
  · 35+ Features
  · Walk-Forward OOS
  · MC Portfolio Forecast
</div>
""", unsafe_allow_html=True)
