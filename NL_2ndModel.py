# streamlit_app_v2_light.py
# ─────────────────────────────────────────────────────────────
# NEXT LEVEL 2ND MODELL – Signal-basierte Strategie v2 (Light Theme)
# pro Ticker separates Konto · Portfolio Forecast (MC) · Walk-Forward
# Weisser Hintergrund · Verbesserte Lesbarkeit · Grössere Schriften
# ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight.*")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# Design System — Hell / Weiss
# ─────────────────────────────────────────────────────────────
THEME = {
    # Hintergründe
    "bg":          "#FFFFFF",
    "bg_page":     "#F8F9FB",
    "bg_card":     "#FFFFFF",
    "bg_header":   "#F0F4FF",
    "bg_sidebar":  "#FAFBFF",
    # Akzentfarben
    "blue":        "#1D4ED8",    # Primär — Strategie / Action
    "blue_light":  "#EFF6FF",
    "green":       "#15803D",    # Positiv / Enter
    "green_light": "#F0FDF4",
    "red":         "#B91C1C",    # Negativ / Exit
    "red_light":   "#FEF2F2",
    "amber":       "#B45309",    # Warnung / Kosten
    "amber_light": "#FFFBEB",
    "purple":      "#6D28D9",    # Wahrscheinlichkeit
    "purple_light":"#F5F3FF",
    "teal":        "#0F766E",    # Info / Hold
    "teal_light":  "#F0FDFA",
    # Text & Rahmen
    "text":        "#0F172A",    # Fast Schwarz — sehr lesbar
    "text_sub":    "#334155",    # Dunkelgrau
    "text_muted":  "#64748B",    # Mittelgrau
    "border":      "#E2E8F0",
    "border_dark": "#CBD5E1",
    "grid":        "#F1F5F9",
}

# Plotly: sauberes helles Layout mit grossen, lesbaren Beschriftungen
PLOTLY_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFBFF",
    font=dict(
        family="'Inter', 'Segoe UI', system-ui, sans-serif",
        color=THEME["text_sub"],
        size=12,
    ),
    title=dict(
        font=dict(size=14, color=THEME["text"], family="'Inter', sans-serif"),
        x=0.0, xanchor="left",
    ),
    xaxis=dict(
        gridcolor=THEME["grid"], gridwidth=1,
        showline=True, linecolor=THEME["border_dark"], linewidth=1,
        zeroline=False,
        tickfont=dict(size=11, color=THEME["text_muted"]),
        title_font=dict(size=12, color=THEME["text_sub"]),
    ),
    yaxis=dict(
        gridcolor=THEME["grid"], gridwidth=1,
        showline=True, linecolor=THEME["border_dark"], linewidth=1,
        zeroline=False,
        tickfont=dict(size=11, color=THEME["text_muted"]),
        title_font=dict(size=12, color=THEME["text_sub"]),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        font_color=THEME["text"],
        font_family="'Inter', sans-serif",
        font_size=12,
        bordercolor=THEME["border_dark"],
    ),
    margin=dict(t=55, b=45, l=65, r=25),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=THEME["border"],
        borderwidth=1,
        font=dict(size=11, color=THEME["text_sub"]),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    modebar=dict(
        bgcolor="rgba(255,255,255,0)",
        color=THEME["text_muted"],
        activecolor=THEME["blue"],
    ),
)

st.set_page_config(
    page_title="NEXUS — 2nd AI Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCAL_TZ    = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 6
pd.options.display.float_format = "{:,.4f}".format

# ─────────────────────────────────────────────────────────────
# Global CSS — Helles Theme, maximale Lesbarkeit
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:           #FFFFFF;
  --bg-page:      #F8F9FB;
  --bg-card:      #FFFFFF;
  --bg-sidebar:   #FAFBFF;
  --blue:         #1D4ED8;
  --blue-lt:      #EFF6FF;
  --green:        #15803D;
  --green-lt:     #F0FDF4;
  --red:          #B91C1C;
  --red-lt:       #FEF2F2;
  --amber:        #B45309;
  --amber-lt:     #FFFBEB;
  --purple:       #6D28D9;
  --purple-lt:    #F5F3FF;
  --teal:         #0F766E;
  --teal-lt:      #F0FDFA;
  --text:         #0F172A;
  --text-sub:     #334155;
  --muted:        #64748B;
  --border:       #E2E8F0;
  --border-dk:    #CBD5E1;
}

/* ── Basis ── */
html, body, [class*="css"], .stApp {
  background-color: var(--bg-page) !important;
  color: var(--text) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  font-size: 14px !important;
  -webkit-font-smoothing: antialiased;
}

/* Hauptbereich etwas heller */
.main .block-container {
  background: var(--bg-page) !important;
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
  font-size: 13px !important;
}
[data-testid="stSidebar"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong {
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--text-sub) !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stNumberInput input {
  background: #FFFFFF !important;
  border: 1px solid var(--border-dk) !important;
  color: var(--text) !important;
  font-size: 13px !important;
  border-radius: 5px !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 16px 18px !important;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06) !important;
  transition: box-shadow 0.2s, border-color 0.2s;
}
[data-testid="metric-container"]:hover {
  box-shadow: 0 4px 12px rgba(29,78,216,0.1) !important;
  border-color: var(--blue) !important;
}
[data-testid="metric-container"] label {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'DM Mono', monospace !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  line-height: 1.2 !important;
}

/* ── Expander ── */
details {
  background: var(--bg-card) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  margin-bottom: 10px !important;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04) !important;
}
details summary {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 13px 18px !important;
  color: var(--text-sub) !important;
  cursor: pointer !important;
  letter-spacing: 0.01em !important;
}
details[open] {
  border-color: var(--blue) !important;
}
details[open] summary { color: var(--blue) !important; }

/* ── DataFrames / Tabellen ── */
.stDataFrame {
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
  box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
}
.stDataFrame thead th {
  background: var(--bg-page) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
  border-bottom: 2px solid var(--border-dk) !important;
  padding: 10px 12px !important;
}
.stDataFrame tbody td {
  font-family: 'DM Mono', monospace !important;
  font-size: 12px !important;
  color: var(--text) !important;
  padding: 8px 12px !important;
  border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr:nth-child(even) {
  background: #FAFBFF !important;
}
.stDataFrame tbody tr:hover {
  background: var(--blue-lt) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--blue) !important;
  border: none !important;
  color: #FFFFFF !important;
  border-radius: 6px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 16px !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 3px rgba(29,78,216,0.25) !important;
}
.stButton > button:hover {
  background: #1E40AF !important;
  box-shadow: 0 4px 12px rgba(29,78,216,0.35) !important;
  transform: translateY(-1px) !important;
}

/* Download-Buttons ── dezent grau */
.stDownloadButton > button {
  background: var(--bg-card) !important;
  border: 1.5px solid var(--border-dk) !important;
  color: var(--text-sub) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
  padding: 6px 14px !important;
}
.stDownloadButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber) !important;
  background: var(--amber-lt) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div {
  background: var(--blue) !important;
  border-radius: 4px !important;
}

/* ── Alerts ── */
.stAlert {
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  border-bottom: 2px solid var(--border) !important;
  background: transparent !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-bottom: 3px solid transparent !important;
  padding: 10px 18px !important;
  margin-bottom: -2px !important;
}
.stTabs [aria-selected="true"] {
  color: var(--blue) !important;
  border-bottom-color: var(--blue) !important;
  background: var(--blue-lt) !important;
  border-radius: 6px 6px 0 0 !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Custom UI-Komponenten ── */

/* Seitenheader */
.nexus-header {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 30px;
  color: var(--text);
  letter-spacing: -0.02em;
  line-height: 1.15;
}
.nexus-sub {
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  font-weight: 400;
  color: var(--muted);
  letter-spacing: 0.04em;
  margin-top: 5px;
}

/* Sektion-Trennlinie */
.section-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 0 12px 0;
  border-bottom: 2px solid var(--border);
  margin: 28px 0 18px 0;
}
.section-bar-label {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-sub);
  white-space: nowrap;
}
.section-bar-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}
.section-bar-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--blue);
  flex-shrink: 0;
}

/* KPI-Karten Grid */
.kpi-grid {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 16px 0;
}
.kpi-box {
  flex: 1;
  min-width: 120px;
  background: var(--bg-card);
  border: 1.5px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.05);
  transition: box-shadow 0.2s, border-color 0.2s;
}
.kpi-box:hover {
  box-shadow: 0 4px 12px rgba(15,23,42,0.1);
  border-color: var(--border-dk);
}
.kpi-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 7px;
}
.kpi-value {
  font-family: 'DM Mono', monospace;
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}

/* Farbvarianten */
.kpi-pos  { border-left: 3px solid var(--green) !important; }
.kpi-pos .kpi-value  { color: var(--green); }
.kpi-neg  { border-left: 3px solid var(--red) !important; }
.kpi-neg .kpi-value  { color: var(--red); }
.kpi-info { border-left: 3px solid var(--blue) !important; }
.kpi-info .kpi-value { color: var(--blue); }
.kpi-warn { border-left: 3px solid var(--amber) !important; }
.kpi-warn .kpi-value { color: var(--amber); }
.kpi-purple { border-left: 3px solid var(--purple) !important; }
.kpi-purple .kpi-value { color: var(--purple); }

/* Signal Badges */
.sig-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 4px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.sig-enter { background: var(--green-lt); color: var(--green); border: 1px solid #BBF7D0; }
.sig-exit  { background: var(--red-lt);   color: var(--red);   border: 1px solid #FECACA; }
.sig-hold  { background: #F8FAFC;         color: var(--muted); border: 1px solid var(--border); }

/* Ticker-Expander Header */
.ticker-title {
  font-family: 'Inter', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
}
.ticker-name {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: var(--muted);
  font-weight: 400;
}

/* Sidebar Logo-Header */
.sidebar-logo {
  padding: 14px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
}
.sidebar-logo-title {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 18px;
  color: var(--text);
  letter-spacing: -0.01em;
}
.sidebar-logo-sub {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--blue);
  letter-spacing: 0.12em;
  margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Hilfsfunktionen
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
    """Sektion-Trennlinie mit Punkt und Label."""
    st.markdown(f"""
    <div class="section-bar">
        <div class="section-bar-dot"></div>
        <div class="section-bar-label">{label}</div>
        <div class="section-bar-line"></div>
    </div>
    """, unsafe_allow_html=True)

def kpi_row(items: list):
    """Rendert eine Zeile farbiger KPI-Boxen.
    items = [(label, wert, css-klasse), ...]
    Klassen: kpi-pos, kpi-neg, kpi-info, kpi-warn, kpi-purple, ''
    """
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
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def _eur(v):
    if not np.isfinite(v): return "–"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.2f} €"

def _apply_theme(fig: go.Figure, height: int = 430) -> go.Figure:
    """Wendet das helle Plotly-Theme auf eine Figur an."""
    fig.update_layout(**PLOTLY_BASE, height=height)
    fig.update_xaxes(
        showspikes=True, spikecolor=THEME["text_muted"],
        spikethickness=1, spikedash="dot", spikemode="across",
    )
    fig.update_yaxes(
        showspikes=True, spikecolor=THEME["text_muted"],
        spikethickness=1, spikedash="dot",
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-title">📊 NEXUS</div>
      <div class="sidebar-logo-sub">2ND AI MODEL · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Universe**")
    ticker_source = st.selectbox("Ticker-Quelle", ["Manuell", "CSV-Upload"],
                                  label_visibility="collapsed")
    tickers_final: List[str] = []

    if ticker_source == "Manuell":
        tickers_input = st.text_input("Tickers (Komma-getrennt)",
                                       value="REGN, LULU, VOW3.DE, REI, DDL")
        tickers_final = _normalize_tickers([t for t in tickers_input.split(",") if t.strip()])
    else:
        st.caption("CSV mit Spalte **ticker** (oder erste Spalte)")
        uploads = st.file_uploader("CSV-Dateien", type=["csv"], accept_multiple_files=True)
        collected = []
        if uploads:
            for up in uploads:
                try: collected += parse_ticker_csv(up)
                except Exception as e: st.error(f"{up.name}: {e}")
        base  = _normalize_tickers(collected)
        extra = st.text_input("Weitere Ticker hinzufügen", value="", key="extra_csv")
        extras = _normalize_tickers([t for t in extra.split(",") if t.strip()]) if extra else []
        tickers_final = _normalize_tickers(base + extras)
        if tickers_final:
            st.caption(f"**{len(tickers_final)}** Ticker geladen")
            max_n = st.number_input("Max. Anzahl (0 = alle)", 0, len(tickers_final), 0, step=10)
            if max_n: tickers_final = tickers_final[:int(max_n)]
            tickers_final = st.multiselect("Auswahl verfeinern",
                                            tickers_final, default=tickers_final)

    if not tickers_final:
        tickers_final = _normalize_tickers(["REGN","VOW3.DE","LULU","REI","DDL"])

    st.download_button("⬇ Ticker-Liste als CSV",
                       to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
                       file_name="tickers.csv", mime="text/csv",
                       use_container_width=True)
    TICKERS = tickers_final

    st.divider()
    st.markdown("**Zeitraum**")
    col_d1, col_d2 = st.columns(2)
    START_DATE = col_d1.date_input("Von", pd.to_datetime("2025-01-01"))
    END_DATE   = col_d2.date_input("Bis", pd.to_datetime(datetime.now(LOCAL_TZ).date()))

    st.divider()
    st.markdown("**Modell**")
    c1s, c2s = st.columns(2)
    LOOKBACK = c1s.number_input("Lookback (Tage)", 10, 252, 35, step=5)
    HORIZON  = c2s.number_input("Horizon (Tage)", 1, 10, 5)
    THRESH   = st.number_input("Target Threshold", 0.0, 0.1, 0.046,
                                 step=0.005, format="%.3f")
    ENTRY_PROB = st.slider("Entry Wahrscheinlichkeit", 0.0, 1.0, 0.62, step=0.01)
    EXIT_PROB  = st.slider("Exit Wahrscheinlichkeit",  0.0, 1.0, 0.48, step=0.01)
    if EXIT_PROB >= ENTRY_PROB:
        st.error("⚠ Exit-Schwelle muss kleiner als Entry-Schwelle sein.")
        st.stop()
    MIN_HOLD_DAYS = st.number_input("Mindesthaltedauer (Tage)", 0, 252, 5, step=1)
    COOLDOWN_DAYS = st.number_input("Cooling Phase (Tage)", 0, 252, 0, step=1)

    st.divider()
    st.markdown("**Execution & Kosten**")
    c3s, c4s = st.columns(2)
    COMMISSION   = c3s.number_input("Commission", 0.0, 0.02, 0.004,
                                     step=0.0001, format="%.4f")
    SLIPPAGE_BPS = c4s.number_input("Slippage (bp)", 0, 50, 5, step=1)
    POS_FRAC     = st.slider("Positionsgrösse (%)", 0.1, 1.0, 1.0, step=0.1)
    INIT_CAP_PER_TICKER = st.number_input("Kapital pro Ticker (€)", 1000.0,
                                           1_000_000.0, 10_000.0,
                                           step=1000.0, format="%.0f")

    st.divider()
    st.markdown("**Intraday**")
    use_live            = st.checkbox("Intraday Tail verwenden", value=True)
    intraday_interval   = st.selectbox("Intervall", ["1m","2m","5m","15m"], index=2)
    fallback_last_session = st.checkbox("Fallback: letzte Session", value=False)
    exec_mode           = st.selectbox("Execution Mode",
                                        ["Next Open (backtest+live)",
                                         "Market-On-Close (live only)"])
    moc_cutoff_min      = st.number_input("MOC Cutoff (Min vor Close)", 5, 60, 15, step=5)
    intraday_chart_type = st.selectbox("Intraday Chart-Typ",
                                        ["Candlestick (OHLC)", "Close-Linie"], index=0)

    st.divider()
    st.markdown("**ML Hyperparameter**")
    c5s, c6s = st.columns(2)
    n_estimators  = c5s.number_input("n_estimators", 10, 500, 100, step=10)
    learning_rate = c6s.number_input("Lernrate", 0.01, 1.0, 0.1,
                                      step=0.01, format="%.2f")
    max_depth     = st.number_input("max_depth", 1, 10, 3, step=1)
    MODEL_PARAMS  = dict(n_estimators=int(n_estimators),
                         learning_rate=float(learning_rate),
                         max_depth=int(max_depth), random_state=42)

    st.divider()
    st.markdown("**Walk-Forward / OOS**")
    use_walk_forward = st.checkbox("Walk-Forward OOS aktivieren", value=False)
    wf_min_train     = st.number_input("WF Min. Trainingsbars", 40, 500, 120, step=10)

    st.divider()
    st.markdown("**Optionsdaten**")
    use_chain_live  = st.checkbox("Live-Optionskette (PCR/VOI)", value=True)
    atm_band_pct    = st.slider("ATM-Band ±%", 1, 15, 5) / 100.0
    max_days_to_exp = st.slider("Max. Restlaufzeit (Tage)", 7, 45, 21)
    n_expiries      = st.slider("Nächste n Verfälle", 1, 4, 2)

    st.divider()
    st.markdown("**Portfolio Forecast (MC)**")
    FORECAST_DAYS = st.number_input("Forecast Horizon (Tage)", 1, 30, 7, step=1)
    MC_SIMS       = st.number_input("MC Simulationen", 200, 5000, 1500, step=100)

    st.divider()
    col_r1, col_r2 = st.columns(2)
    if col_r1.button("🗑 Cache leeren", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    if col_r2.button("↺ Neu laden", use_container_width=True):
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Datenladen
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
def get_price_data_tail_intraday(ticker, years=3, use_tail=True, interval="5m",
                                  fallback_last_session=False,
                                  exec_mode_key="Next Open (backtest+live)",
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
        intraday = tk.history(period="1d", interval=interval,
                               auto_adjust=True, actions=False, prepost=False)
        if not intraday.empty:
            if intraday.index.tz is None:
                intraday.index = intraday.index.tz_localize("UTC")
            intraday.index = intraday.index.tz_convert(LOCAL_TZ)
            intraday = intraday.sort_index()
        else:
            intraday = pd.DataFrame()
    except Exception:
        intraday = pd.DataFrame()

    if exec_mode_key.startswith("Market-On-Close") and not intraday.empty:
        intraday = intraday.loc[
            :datetime.now(LOCAL_TZ) - timedelta(minutes=int(moc_cutoff_min_val))
        ]

    if intraday.empty and fallback_last_session:
        try:
            intr5 = tk.history(period="5d", interval=interval,
                                auto_adjust=True, actions=False, prepost=False)
            if not intr5.empty:
                if intr5.index.tz is None:
                    intr5.index = intr5.index.tz_localize("UTC")
                intr5.index = intr5.index.tz_convert(LOCAL_TZ)
                intraday = intr5.loc[str(intr5.index[-1].date())]
        except Exception:
            pass

    if not intraday.empty:
        lb = intraday.iloc[-1]
        dk = pd.Timestamp(lb.name.date(), tz=LOCAL_TZ)
        row = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(lb["Close"]),
            "Volume": float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan,
        }
        if dk in df.index:
            for k, v in row.items(): df.loc[dk, k] = v
        else:
            df.loc[dk] = row
        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"] = lb.name

    df.dropna(subset=["High","Low","Close","Open"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=180)
def get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval="5m"):
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval,
                       auto_adjust=True, actions=False, prepost=False)
    if intr.empty: return intr
    if intr.index.tz is None: intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr = intr.sort_index()
    keep = set(pd.Index(intr.index.normalize().unique())[-sessions:])
    return intr.loc[intr.index.normalize().isin(keep)].copy()


def load_all_prices(tickers, start, end, use_tail, interval,
                    fallback_last, exec_key, moc_cutoff):
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map:  Dict[str, dict] = {}
    if not tickers: return price_map, meta_map

    with st.spinner(f"Kursdaten werden geladen — {len(tickers)} Ticker …"):
        prog = st.progress(0.0)
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
            fmap = {
                ex.submit(get_price_data_tail_intraday, tk, 3, use_tail, interval,
                          fallback_last, exec_key, int(moc_cutoff)): tk
                for tk in tickers
            }
            done = 0
            for fut in as_completed(fmap):
                tk = fmap[fut]
                try:
                    df_full, meta = fut.result()
                    df_use = df_full.loc[str(start):str(end)].copy()
                    if not df_use.empty:
                        price_map[tk] = df_use
                        meta_map[tk]  = meta
                except Exception as e:
                    st.error(f"Fehler bei {tk}: {e}")
                finally:
                    done += 1
                    prog.progress(done / len(tickers))
    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Optionsketten-Aggregation
# ─────────────────────────────────────────────────────────────
def _atm_strike(ref_px, strikes):
    if not np.isfinite(ref_px) or strikes.size == 0: return np.nan
    return float(strikes[np.argmin(np.abs(strikes - ref_px))])

def _band_mask(strikes, atm, band):
    if not np.isfinite(atm):
        return pd.Series([False]*len(strikes), index=strikes.index)
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
         for e in exps
         if (pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize()
             - today).days <= max_days],
        key=lambda x: x[0]
    )
    exps_use = [e for _, e in exps_filt[:max(1, n_exps)]]
    if not exps_use: return pd.DataFrame()

    rows = []
    for e in exps_use:
        try:
            ch = tk.option_chain(e); calls, puts = ch.calls.copy(), ch.puts.copy()
        except Exception: continue
        for df_ in (calls, puts):
            for c in ["volume","openInterest","impliedVolatility","strike"]:
                if c not in df_.columns: df_[c] = np.nan
        strikes = np.sort(pd.concat([calls["strike"], puts["strike"]]).dropna().unique())
        atm = _atm_strike(ref_price, strikes)
        mC  = calls[_band_mask(calls["strike"], atm, atm_band)]
        mP  = puts [_band_mask(puts ["strike"], atm, atm_band)]
        rows.append({
            "exp":   e,
            "vol_c": float(np.nansum(mC["volume"])),
            "vol_p": float(np.nansum(mP["volume"])),
            "oi_c":  float(np.nansum(mC["openInterest"])),
            "oi_p":  float(np.nansum(mP["openInterest"])),
            "voi_c": float(np.nansum(mC["volume"])) / max(float(np.nansum(mC["openInterest"])),1.0),
            "voi_p": float(np.nansum(mP["volume"])) / max(float(np.nansum(mP["openInterest"])),1.0),
            "iv_c":  float(np.nanmean(mC["impliedVolatility"])) if len(mC) else np.nan,
            "iv_p":  float(np.nanmean(mP["impliedVolatility"])) if len(mP) else np.nan,
        })

    if not rows: return pd.DataFrame()
    agg = pd.DataFrame(rows).agg({
        "vol_c":"sum","vol_p":"sum","oi_c":"sum","oi_p":"sum",
        "voi_c":"mean","voi_p":"mean","iv_c":"mean","iv_p":"mean"
    })
    out = pd.DataFrame([{
        "PCR_vol":            float(agg["vol_p"]/max(agg["vol_c"],1.0)),
        "PCR_oi":             float(agg["oi_p"] /max(agg["oi_c"],1.0)),
        "VOI_call":           float(agg["voi_c"]),
        "VOI_put":            float(agg["voi_p"]),
        "IV_skew_p_minus_c":  float(agg["iv_p"] - agg["iv_c"]),
        "VOL_tot":            float(agg["vol_c"] + agg["vol_p"]),
        "OI_tot":             float(agg["oi_c"]  + agg["oi_p"]),
    }])
    out.index = [pd.Timestamp.today(tz=LOCAL_TZ).normalize()]
    return out


# ─────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────
def make_features(df, lookback, horizon, exog=None):
    if len(df) < (lookback + horizon + 5):
        raise ValueError("Zu wenige Bars für Lookback/Horizon.")
    feat = df.copy()
    feat["Range"]     = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)
    feat = feat.iloc[lookback-1:].copy()
    if exog is not None and not exog.empty:
        feat = feat.join(exog, how="left").ffill()
    feat["FutureRetExec"] = feat["Open"].shift(-horizon) / feat["Open"].shift(-1) - 1
    return feat

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
    cash_g = init_cap; cash_n = init_cap
    shares = 0.0; in_pos = False
    cb_g = 0.0; cb_n = 0.0
    last_ei: Optional[int] = None
    last_xi: Optional[int] = None
    eq_g, eq_n, trades, cum = [], [], [], 0.0

    for i in range(n):
        if i > 0:
            ot = float(df["Open"].iloc[i])
            sb = ot * (1 + slippage_bps/10000.0)
            ss = ot * (1 - slippage_bps/10000.0)
            pp = float(df["SignalProb"].iloc[i-1])
            de = df.index[i]
            cool = True
            if (not in_pos) and cooldown_days > 0 and last_xi is not None:
                cool = (i - last_xi) >= int(cooldown_days)

            if (not in_pos) and (pp > entry_thr) and cool:
                inv = cash_n * float(pos_frac)
                fee = inv * float(commission)
                sh  = max((inv - fee) / sb, 0.0)
                if sh > 0 and (sh*sb+fee) <= cash_n+1e-6:
                    shares = sh; cb_g = sh*sb; cb_n = sh*sb+fee
                    cash_g -= cb_g; cash_n -= cb_n
                    in_pos = True; last_ei = i
                    trades.append({
                        "Date": de, "Typ": "Entry", "Price": round(sb,4),
                        "Shares": round(sh,4), "Gross P&L": 0.0,
                        "Fees": round(fee,2), "Net P&L": 0.0,
                        "kum P&L": round(cum,2), "Prob": round(pp,4), "HoldDays": np.nan
                    })

            elif in_pos and pp < exit_thr:
                held = (i - last_ei) if last_ei is not None else 0
                if not (int(min_hold_days) > 0 and held < int(min_hold_days)):
                    gv = shares*ss; fe = gv*float(commission)
                    pnl_g = gv-cb_g; pnl_n = (gv-fe)-cb_n
                    cash_g += gv; cash_n += (gv-fe)
                    in_pos = False; shares = 0.0; cb_g = 0.0; cb_n = 0.0; cum += pnl_n
                    trades.append({
                        "Date": de, "Typ": "Exit", "Price": round(ss,4),
                        "Shares": 0.0, "Gross P&L": round(pnl_g,2),
                        "Fees": round(fe,2), "Net P&L": round(pnl_n,2),
                        "kum P&L": round(cum,2), "Prob": round(pp,4), "HoldDays": int(held)
                    })
                    last_xi = i; last_ei = None

        ct = float(df["Close"].iloc[i])
        eq_g.append(cash_g + (shares*ct if in_pos else 0.0))
        eq_n.append(cash_n + (shares*ct if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = eq_g
    df_bt["Equity_Net"]   = eq_n
    return df_bt, trades


# ─────────────────────────────────────────────────────────────
# Performance-Kennzahlen
# ─────────────────────────────────────────────────────────────
def _cagr(v):
    if len(v) < 2: return np.nan
    dt0, dt1 = pd.to_datetime(v.index[0]), pd.to_datetime(v.index[-1])
    yrs = max((dt1-dt0).days/365.25, 1e-9)
    return (v.iloc[-1]/v.iloc[0])**(1/yrs) - 1

def _sortino(rets):
    if rets.empty: return np.nan
    mean = rets.mean()*252; down = rets[rets<0]
    dd   = down.std()*np.sqrt(252) if len(down) else np.nan
    return mean/dd if dd and np.isfinite(dd) and dd > 0 else np.nan

def _winrate(trades):
    if not trades: return np.nan
    pnl = []; e = None
    for ev in trades:
        if ev["Typ"] == "Entry": e = ev
        elif ev["Typ"] == "Exit" and e is not None:
            pnl.append(float(ev.get("Net P&L", 0.0))); e = None
    return float((np.array(pnl, float) > 0).mean()) if pnl else np.nan

def compute_performance(df_bt, trades, init_cap):
    net_ret  = (df_bt["Equity_Net"].iloc[-1]/init_cap - 1)*100
    rets     = df_bt["Equity_Net"].pct_change().dropna()
    vol      = rets.std()*sqrt(252)*100
    sharpe   = (rets.mean()*sqrt(252))/(rets.std()+1e-12)
    dd       = (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax()) / df_bt["Equity_Net"].cummax()
    max_dd   = dd.min()*100
    calmar   = (net_ret/100)/abs(max_dd/100) if max_dd < 0 else np.nan
    gross_r  = (df_bt["Equity_Gross"].iloc[-1]/init_cap - 1)*100
    bh       = (df_bt["Close"].iloc[-1]/df_bt["Close"].iloc[0] - 1)*100
    fees_sum = float(sum(t.get("Fees",0.) for t in trades))
    cagr_v   = _cagr(df_bt["Equity_Net"])
    sort_v   = _sortino(rets)
    wr       = _winrate(trades)
    return {
        "Strategy Net (%)":   round(float(net_ret), 2),
        "Strategy Gross (%)": round(float(gross_r), 2),
        "Buy & Hold Net (%)": round(float(bh), 2),
        "Volatility (%)":     round(float(vol), 2),
        "Sharpe-Ratio":       round(float(sharpe), 2),
        "Sortino-Ratio":      round(float(sort_v), 2) if np.isfinite(sort_v) else np.nan,
        "Max Drawdown (%)":   round(float(max_dd), 2),
        "Calmar-Ratio":       round(float(calmar), 2) if np.isfinite(calmar) else np.nan,
        "Fees (€)":           round(fees_sum, 2),
        "Phase":              "Open" if trades and trades[-1]["Typ"]=="Entry" else "Flat",
        "Number of Trades":   int(sum(1 for t in trades if t["Typ"]=="Exit")),
        "Net P&L (€)":        round(float(df_bt["Equity_Net"].iloc[-1] - init_cap), 2),
        "CAGR (%)":           round(100*float(cagr_v) if np.isfinite(cagr_v) else np.nan, 2),
        "Winrate (%)":        round(100*float(wr) if np.isfinite(wr) else np.nan, 2),
        "InitCap (€)":        float(init_cap),
    }

def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk); ce = None
        for ev in tr:
            if ev["Typ"] == "Entry": ce = ev
            elif ev["Typ"] == "Exit" and ce is not None:
                ed = pd.to_datetime(ce["Date"]); xd = pd.to_datetime(ev["Date"])
                sh = float(ce.get("Shares",0.)); ep = float(ce.get("Price",np.nan))
                xp = float(ev.get("Price",np.nan))
                fe = float(ce.get("Fees",0.));   fx = float(ev.get("Fees",0.))
                pnl = float(ev.get("Net P&L",0.)); cost = sh*ep+fe
                rows.append({
                    "Ticker": tk, "Name": name,
                    "Entry Datum": ed, "Exit Datum": xd,
                    "Hold (Tage)": (xd-ed).days,
                    "Entry Prob": ce.get("Prob",np.nan),
                    "Exit Prob":  ev.get("Prob",np.nan),
                    "Stücke": round(sh,4),
                    "Entry Preis": round(ep,4), "Exit Preis": round(xp,4),
                    "PnL Netto (€)": round(pnl,2), "Gebühren (€)": round(fe+fx,2),
                    "Return (%)": round(pnl/cost*100,2) if cost else np.nan,
                })
                ce = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Forecast-Hilfen
# ─────────────────────────────────────────────────────────────
def estimate_expected_return(feat, forecast_days, threshold):
    if (feat is None or feat.empty
            or "Open" not in feat.columns
            or "SignalProb" not in feat.columns):
        return {}
    fr  = feat["Open"].shift(-int(forecast_days)) / feat["Open"].shift(-1) - 1
    tmp = pd.DataFrame({"FutureRet": fr}).dropna()
    if tmp.empty: return {}
    tmp["T"] = (tmp["FutureRet"] > float(threshold)).astype(int)
    mu1 = float(tmp.loc[tmp["T"]==1,"FutureRet"].mean()) if tmp["T"].sum() > 0 else 0.0
    mu0 = float(tmp.loc[tmp["T"]==0,"FutureRet"].mean()) if (1-tmp["T"]).sum() > 0 else 0.0
    p   = float(pd.to_numeric(feat["SignalProb"].iloc[-1], errors="coerce"))
    if not np.isfinite(p): p = 0.5
    return {"mu1": mu1, "mu0": mu0, "p": p, "exp_ret": p*mu1 + (1-p)*mu0}

def _ensure_psd(cov, eps=1e-12):
    cov = (cov+cov.T)/2
    try:
        w, v = np.linalg.eigh(cov); return (v*np.maximum(w,eps)) @ v.T
    except Exception:
        return np.diag(np.maximum(np.diag(cov), eps))

def portfolio_mc(exp_rets, cov, nav0, sims=1500, seed=42):
    tickers = exp_rets.index.tolist()
    cov     = cov.reindex(index=tickers, columns=tickers).fillna(0.)
    w       = np.ones(len(tickers)) / max(len(tickers), 1)
    rng     = np.random.default_rng(int(seed))
    draws   = rng.multivariate_normal(exp_rets.values, _ensure_psd(cov.values), size=int(sims))
    pr      = draws @ w; nv = nav0*(1+pr)
    q       = np.quantile(pr, [.05,.5,.95])
    qn      = np.quantile(nv, [.05,.5,.95])
    return {"q05":float(q[0]),"q50":float(q[1]),"q95":float(q[2]),
            "nq05":float(qn[0]),"nq50":float(qn[1]),"nq95":float(qn[2]),
            "port_rets": pr, "nav_paths": nv}


# ─────────────────────────────────────────────────────────────
# Model Training + Backtest
# ─────────────────────────────────────────────────────────────
def make_features_and_train(df, lookback, horizon, threshold, model_params,
                             entry_prob, exit_prob, init_capital, pos_frac,
                             min_hold_days=0, cooldown_days=0, exog_df=None,
                             walk_forward=False, wf_min_train=120):
    feat = make_features(df, lookback, horizon, exog=exog_df)
    hist = feat.iloc[:-1].dropna(subset=["FutureRetExec"]).copy()
    if len(hist) < 30: raise ValueError("Zu wenige Datenpunkte für Modell.")
    X_cols = ["Range","SlopeHigh","SlopeLow"]
    opt_c  = ["PCR_vol","PCR_oi","VOI_call","VOI_put",
               "IV_skew_p_minus_c","VOL_tot","OI_tot"]
    X_cols += [c for c in opt_c if c in feat.columns]
    hist["Target"] = (hist["FutureRetExec"] > threshold).astype(int)

    def make_pipe():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(**model_params)),
        ])

    if hist["Target"].nunique() < 2:
        feat["SignalProb"] = 0.5
    elif not walk_forward:
        p = make_pipe(); p.fit(hist[X_cols].values, hist["Target"].values)
        feat["SignalProb"] = p.predict_proba(feat[X_cols].values)[:,1]
    else:
        probs = np.full(len(feat), np.nan)
        mt = max(int(wf_min_train), lookback+horizon+10)
        for t in range(mt, len(feat)):
            tr = feat.iloc[:t].dropna(subset=["FutureRetExec"]).copy()
            if len(tr) < mt: continue
            tr["Target"] = (tr["FutureRetExec"] > threshold).astype(int)
            if tr["Target"].nunique() < 2: continue
            p = make_pipe(); p.fit(tr[X_cols].values, tr["Target"].values)
            probs[t] = p.predict_proba(feat[X_cols].iloc[[t]].values)[0,1]
        feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)

    df_bt, trades = backtest_next_open(
        feat.iloc[:-1], entry_prob, exit_prob, COMMISSION, SLIPPAGE_BPS,
        init_capital, pos_frac,
        min_hold_days=int(min_hold_days), cooldown_days=int(cooldown_days)
    )
    return feat, df_bt, trades, compute_performance(df_bt, trades, init_capital)


# ─────────────────────────────────────────────────────────────
# Chart-Builder — Helles Theme
# ─────────────────────────────────────────────────────────────
def chart_price_signal(feat, trades, ticker):
    """Candlestick/Preis + Signal-Wahrscheinlichkeit als Subpanel."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.04,
        subplot_titles=("", "Signal-Wahrscheinlichkeit"),
    )
    # Subtitles kleiner
    fig.layout.annotations[0].update(font=dict(size=11, color=THEME["text_muted"]))

    # Candlestick
    if all(c in feat.columns for c in ["Open","High","Low","Close"]):
        fig.add_trace(go.Candlestick(
            x=feat.index, open=feat["Open"], high=feat["High"],
            low=feat["Low"], close=feat["Close"],
            name="OHLC", showlegend=False,
            increasing=dict(
                line=dict(color=THEME["green"], width=1),
                fillcolor="rgba(21,128,61,0.18)",
            ),
            decreasing=dict(
                line=dict(color=THEME["red"], width=1),
                fillcolor="rgba(185,28,28,0.14)",
            ),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=feat.index, y=feat["Close"], mode="lines",
            line=dict(color=THEME["blue"], width=2), showlegend=False,
        ), row=1, col=1)

    # Entry/Exit Marker
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["Date"] = pd.to_datetime(tdf["Date"])
        for typ, sym, col, lbl in [
            ("Entry", "triangle-up",   THEME["green"], "Kauf"),
            ("Exit",  "triangle-down", THEME["red"],   "Verkauf"),
        ]:
            sub = tdf[tdf["Typ"] == typ]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Date"], y=sub["Price"],
                    mode="markers", name=lbl,
                    marker=dict(
                        symbol=sym, size=12, color=col,
                        line=dict(color="white", width=2),
                    ),
                    hovertemplate=(
                        f"<b>{lbl}</b><br>"
                        f"Datum: %{{x|%d.%m.%Y}}<br>"
                        f"Preis: %{{y:.2f}}<extra></extra>"
                    ),
                ), row=1, col=1)

    # Signal-Prob Panel
    prob = feat["SignalProb"]
    # Grüne Füllung über Entry-Schwelle, rote darunter
    fig.add_trace(go.Scatter(
        x=feat.index, y=prob, mode="lines",
        name="P(Signal)",
        line=dict(color=THEME["purple"], width=2),
        fill="tozeroy",
        fillcolor="rgba(109,40,217,0.08)",
        hovertemplate="%{x|%d.%m.%Y}  P = %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    # Schwellwert-Linien
    fig.add_hline(
        y=ENTRY_PROB, row=2, col=1,
        line_color=THEME["green"], line_dash="dash", line_width=1.5, opacity=0.75,
        annotation_text=f"  Entry {ENTRY_PROB:.2f}",
        annotation_font=dict(size=10, color=THEME["green"]),
        annotation_position="right",
    )
    fig.add_hline(
        y=EXIT_PROB, row=2, col=1,
        line_color=THEME["red"], line_dash="dash", line_width=1.5, opacity=0.75,
        annotation_text=f"  Exit {EXIT_PROB:.2f}",
        annotation_font=dict(size=10, color=THEME["red"]),
        annotation_position="right",
    )

    _apply_theme(fig, 520)
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b>  –  Kurs & Signalwahrscheinlichkeit",
            font=dict(size=14, color=THEME["text"]),
        ),
        xaxis_rangeslider_visible=False,
        yaxis2=dict(
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            tickformat=".2f",
            tickfont=dict(size=10),
        ),
    )
    return fig


def chart_equity(df_bt, ticker, init_cap):
    """Equity-Kurve + Drawdown Subpanel."""
    eq = df_bt["Equity_Net"]
    bh = init_cap * df_bt["Close"] / df_bt["Close"].iloc[0]
    dd = (eq - eq.cummax()) / eq.cummax() * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Equity-Kurve vs. Buy & Hold", "Drawdown (%)"),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color=THEME["text_muted"]))

    # Strategie
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, mode="lines", name="Strategie (netto)",
        line=dict(color=THEME["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(29,78,216,0.05)",
        hovertemplate="%{x|%d.%m.%Y}  %{y:,.2f} €<extra></extra>",
    ), row=1, col=1)

    # Buy & Hold
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh, mode="lines", name="Buy & Hold",
        line=dict(color=THEME["text_muted"], width=1.5, dash="dot"),
        hovertemplate="%{x|%d.%m.%Y}  B&H: %{y:,.2f} €<extra></extra>",
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, mode="lines", name="Drawdown",
        line=dict(color=THEME["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(185,28,28,0.08)",
        showlegend=False,
        hovertemplate="%{x|%d.%m.%Y}  DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    _apply_theme(fig, 470)
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b>  –  Equity & Drawdown",
            font=dict(size=14, color=THEME["text"]),
        ),
        yaxis=dict(title="NAV (€)", tickformat=",.0f",
                   titlefont=dict(size=12)),
        yaxis2=dict(title="DD (%)", tickformat=".1f",
                    titlefont=dict(size=12)),
    )
    return fig


def chart_intraday(intra, ticker, tdf, chart_type, interval):
    """Intraday-Chart letzte 5 Handelstage."""
    fig = go.Figure()

    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(
            x=intra.index, open=intra["Open"], high=intra["High"],
            low=intra["Low"], close=intra["Close"],
            showlegend=False,
            increasing=dict(
                line=dict(color=THEME["green"], width=1),
                fillcolor="rgba(21,128,61,0.18)",
            ),
            decreasing=dict(
                line=dict(color=THEME["red"], width=1),
                fillcolor="rgba(185,28,28,0.14)",
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=intra.index, y=intra["Close"], mode="lines",
            line=dict(color=THEME["blue"], width=2), showlegend=False,
        ))

    # Tages-Trennlinien
    for _, ds in intra.groupby(intra.index.normalize()):
        fig.add_vline(
            x=ds.index.min(), line_width=1, line_dash="dot",
            line_color=THEME["border_dark"], opacity=0.6,
        )

    # Trade-Marker
    if not tdf.empty:
        tdf2 = tdf.copy(); tdf2["Date"] = pd.to_datetime(tdf2["Date"])
        last_days = set(intra.index.normalize())
        ev = tdf2[tdf2["Date"].dt.normalize().isin(last_days)]
        for typ, col, sym, lbl in [
            ("Entry", THEME["green"], "triangle-up",   "Kauf"),
            ("Exit",  THEME["red"],   "triangle-down", "Verkauf"),
        ]:
            xs, ys = [], []
            for d, ds in intra.groupby(intra.index.normalize()):
                h = ev[(ev["Typ"]==typ) & (ev["Date"].dt.normalize()==d)]
                if h.empty: continue
                xs.append(ds.index.min()); ys.append(float(h["Price"].iloc[-1]))
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", name=lbl,
                    marker=dict(symbol=sym, size=12, color=col,
                                line=dict(color="white", width=2)),
                ))

    _apply_theme(fig, 410)
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker}</b>  –  Intraday, letzte 5 Handelstage ({interval})",
            font=dict(size=14, color=THEME["text"]),
        ),
        xaxis_rangeslider_visible=False,
    )
    return fig


def chart_corr(corr):
    """Korrelations-Heatmap hell."""
    n = corr.shape[0]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        text=corr.round(2).astype(str).values,
        texttemplate="%{text}",
        colorscale=[
            [0.0,  "#B91C1C"],   # starke neg. Korrelation — Rot
            [0.35, "#FCA5A5"],
            [0.5,  "#F8FAFC"],   # keine Korrelation — Fast Weiss
            [0.65, "#93C5FD"],
            [1.0,  "#1D4ED8"],   # starke pos. Korrelation — Blau
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title="ρ",
            title_font=dict(size=12, color=THEME["text_sub"]),
            tickfont=dict(size=11, color=THEME["text_sub"]),
            thickness=14,
        ),
        hovertemplate=(
            "<b>%{x}</b> vs <b>%{y}</b><br>"
            "ρ = %{z:.3f}<extra></extra>"
        ),
        textfont=dict(
            size=12 if n <= 8 else 10,
            color=THEME["text"],
        ),
    ))
    _apply_theme(fig, max(400, n*48))
    fig.update_layout(
        title=dict(
            text="Korrelationsmatrix (Kursrenditen)",
            font=dict(size=14, color=THEME["text"]),
        ),
        xaxis=dict(tickangle=-30, showgrid=False,
                   tickfont=dict(size=12, color=THEME["text_sub"])),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(size=12, color=THEME["text_sub"])),
    )
    return fig


def chart_portfolio_nav(nav):
    """Portfolio NAV-Kurve hell."""
    fig = go.Figure(go.Scatter(
        x=nav.index, y=nav.values, mode="lines",
        line=dict(color=THEME["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(29,78,216,0.07)",
        hovertemplate="%{x|%d.%m.%Y}  %{y:,.0f} €<extra></extra>",
        name="Portfolio NAV",
    ))
    _apply_theme(fig, 370)
    fig.update_layout(
        title=dict(
            text="Portfolio NAV — Equal-Weight",
            font=dict(size=14, color=THEME["text"]),
        ),
        yaxis=dict(title="NAV (€)", tickformat=",.0f"),
        showlegend=False,
    )
    return fig


def chart_mc_histogram(port_rets, q05, q50, q95, forecast_days, sims):
    """Monte-Carlo Rendite-Histogramm hell."""
    colors = [
        "rgba(21,128,61,0.6)" if v >= 0 else "rgba(185,28,28,0.6)"
        for v in port_rets
    ]
    fig = go.Figure(go.Histogram(
        x=port_rets*100, nbinsx=50,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        showlegend=False,
        hovertemplate="Return: %{x:.2f}%<br>Häufigkeit: %{y}<extra></extra>",
    ))
    annotations = []
    for val, lbl, col in [
        (q05*100,  f"5%: {q05*100:.2f}%",   THEME["red"]),
        (q50*100,  f"Median: {q50*100:.2f}%", THEME["blue"]),
        (q95*100,  f"95%: {q95*100:.2f}%",   THEME["green"]),
    ]:
        fig.add_vline(
            x=val, line_dash="dash", line_color=col, line_width=2, opacity=0.85,
        )
        annotations.append(dict(
            x=val, yref="paper", y=0.96, text=f"  {lbl}",
            showarrow=False,
            font=dict(size=11, color=col, family="'DM Mono', monospace"),
            xanchor="left",
        ))
    fig.update_layout(annotations=annotations)
    _apply_theme(fig, 370)
    fig.update_layout(
        title=dict(
            text=f"Simulierte Portfolio-Renditen  ·  {forecast_days} Tage  ·  {sims:,} Simulationen",
            font=dict(size=14, color=THEME["text"]),
        ),
        xaxis_title="Return (%)",
        yaxis_title="Häufigkeit",
        bargap=0.04,
    )
    return fig


def chart_histogram(data, xlabel, title, bins):
    """Allgemeines Trade-Histogramm hell."""
    mean_v = float(data.mean()) if len(data) else np.nan
    med_v  = float(data.median()) if len(data) else np.nan
    colors = [
        "rgba(21,128,61,0.6)" if v >= 0 else "rgba(185,28,28,0.6)"
        for v in data
    ]
    fig = go.Figure(go.Histogram(
        x=data, nbinsx=bins,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        showlegend=False,
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>Häufigkeit: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=THEME["text_muted"], line_width=1.5, opacity=0.5)
    annotations = []
    if np.isfinite(mean_v):
        fig.add_vline(x=mean_v, line_dash="dash",
                      line_color=THEME["blue"], line_width=2, opacity=0.8)
        annotations.append(dict(
            x=mean_v, yref="paper", y=0.95,
            text=f"  Ø {mean_v:.2f}",
            showarrow=False,
            font=dict(size=11, color=THEME["blue"], family="'DM Mono', monospace"),
            xanchor="left",
        ))
    if np.isfinite(med_v):
        fig.add_vline(x=med_v, line_dash="dot",
                      line_color=THEME["amber"], line_width=2, opacity=0.8)
        annotations.append(dict(
            x=med_v, yref="paper", y=0.85,
            text=f"  Median {med_v:.2f}",
            showarrow=False,
            font=dict(size=11, color=THEME["amber"], family="'DM Mono', monospace"),
            xanchor="left",
        ))
    if annotations:
        fig.update_layout(annotations=annotations)
    _apply_theme(fig, 350)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=THEME["text"])),
        xaxis_title=xlabel,
        yaxis_title="Anzahl Trades",
        bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Optimierer
# ─────────────────────────────────────────────────────────────
section("PARAMETER-OPTIMIERUNG")
with st.expander("⚙ Random-Search Optimizer  –  Walk-Forward Light", expanded=False):
    oc1, oc2 = st.columns(2)
    with oc1:
        n_trials       = st.number_input("Anzahl Trials", 10, 1000, 80, step=10)
        seed_opt       = st.number_input("Zufalls-Seed", 0, 10000, 42)
        lambda_trades  = st.number_input("Penalty λ pro Trade", 0.0, 1.0, 0.02, step=0.005)
        min_trades_req = st.number_input("Min. Trades (Filter)", 0, 10000, 5, step=1)
    with oc2:
        lb_lo, lb_hi   = st.slider("Lookback-Bereich", 10, 252, (30,120), step=5)
        hz_lo, hz_hi   = st.slider("Horizon-Bereich", 1, 10, (3,8))
        thr_lo, thr_hi = st.slider("Threshold-Bereich", 0.0, 0.10, (0.035,0.10),
                                    step=0.005, format="%.3f")
        en_lo,  en_hi  = st.slider("Entry Prob Bereich", 0.0, 1.0, (0.55,0.85), step=0.01)
        ex_lo,  ex_hi  = st.slider("Exit Prob Bereich",  0.0, 1.0, (0.30,0.60), step=0.01)

    @st.cache_data(show_spinner=False)
    def _opt_prices(tickers, start, end, use_tail, interval, fb, ek, moc):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fb, ek, moc)[0]

    if st.button("Suche starten", type="primary", use_container_width=True):
        import random
        from collections import Counter
        rng_opt = random.Random(int(seed_opt))
        pm = _opt_prices(
            tuple(TICKERS), str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
        )

        min_len = max(120, int(wf_min_train)+40)
        feasible = {tk: df.copy() for tk, df in (pm or {}).items()
                    if isinstance(df, pd.DataFrame) and len(df) >= min_len}
        if len(feasible) < 2:
            st.error("Zu wenige Ticker nach Vorfilter."); st.stop()

        rows_o, best_o = [], None
        prog_o  = st.progress(0.0)
        status_o= st.empty()
        err_c   = Counter()

        for trial in range(int(n_trials)):
            p = dict(
                lookback=rng_opt.randrange(lb_lo, lb_hi+1, 5),
                horizon =rng_opt.randrange(hz_lo, hz_hi+1),
                thresh  =rng_opt.uniform(thr_lo, thr_hi),
                entry   =rng_opt.uniform(en_lo, en_hi),
                exit    =rng_opt.uniform(ex_lo, ex_hi),
            )
            if p["exit"] >= p["entry"]:
                prog_o.progress((trial+1)/int(n_trials)); continue

            sharps_o, trad_o, ok_t = [], 0, 0
            for tk, df0 in feasible.items():
                if len(df0) < max(80, p["lookback"]+p["horizon"]+40): continue
                try:
                    feat_o, hist_o = build_feature_cache(
                        df0, int(p["lookback"]), int(p["horizon"]), float(p["thresh"])
                    )
                    if feat_o is None or hist_o is None or hist_o["Target"].nunique() < 2:
                        raise ValueError("Entartetes Target")
                    X_c   = [c for c in ["Range","SlopeHigh","SlopeLow"] if c in feat_o.columns]
                    n_h   = len(hist_o); split = max(int(wf_min_train), int(0.6*n_h))
                    if n_h-split < 30: raise ValueError("OOS zu kurz")
                    pp = Pipeline([
                        ("i", SimpleImputer(strategy="median")),
                        ("m", GradientBoostingClassifier(**MODEL_PARAMS)),
                    ])
                    pp.fit(hist_o.iloc[:split][X_c].values, hist_o.iloc[:split]["Target"].values)
                    feat_o["SignalProb"] = pp.predict_proba(feat_o[X_c].values)[:,1]
                    df_bt_o, tr_o = backtest_next_open(
                        feat_o.iloc[:-1], float(p["entry"]), float(p["exit"]),
                        COMMISSION, SLIPPAGE_BPS, float(INIT_CAP_PER_TICKER),
                        float(POS_FRAC), int(MIN_HOLD_DAYS), int(COOLDOWN_DAYS)
                    )
                    oos_eq = df_bt_o.loc[
                        df_bt_o.index.intersection(hist_o.index[split:])
                    ]["Equity_Net"]
                    if len(oos_eq) < 30: raise ValueError("OOS zu kurz")
                    r = oos_eq.pct_change().dropna()
                    if r.empty: raise ValueError("Leere Returns")
                    sharps_o.append(
                        float((r.mean()/(r.std(ddof=0)+1e-12))*np.sqrt(252))
                    )
                    trad_o += int(sum(1 for t in tr_o if t["Typ"]=="Exit"))
                    ok_t   += 1
                except Exception as e:
                    err_c[str(e)[:80]] += 1

            if ok_t < 2 or not sharps_o:
                prog_o.progress((trial+1)/int(n_trials)); continue
            sh_med = float(np.nanmedian(sharps_o))
            if not np.isfinite(sh_med) or trad_o < int(min_trades_req):
                prog_o.progress((trial+1)/int(n_trials)); continue
            score = sh_med - float(lambda_trades)*(trad_o/max(1, ok_t))
            rec   = dict(trial=trial, score=score, sharpe_med=sh_med,
                         trades=trad_o, ok_tickers=ok_t, **p)
            rows_o.append(rec)
            if best_o is None or score > best_o["score"]: best_o = rec
            if (trial+1) % 10 == 0:
                status_o.caption(
                    f"Trial {trial+1}/{n_trials}  –  Bester Score: {best_o['score']:.3f}"
                )
            prog_o.progress((trial+1)/int(n_trials))

        if not rows_o:
            st.error("Keine gültigen Ergebnisse gefunden.")
            if err_c:
                st.dataframe(
                    pd.DataFrame(err_c.most_common(10), columns=["Fehler","Anzahl"])
                )
        else:
            df_res_o = pd.DataFrame(rows_o).sort_values("score", ascending=False)
            st.success(
                f"Optimierung abgeschlossen  ·  "
                f"Bester Score: **{best_o['score']:.3f}**  ·  "
                f"Sharpe: **{best_o['sharpe_med']:.2f}**  ·  "
                f"Trades: **{best_o['trades']}**"
            )
            bc = st.columns(5)
            bc[0].metric("Lookback",      int(best_o["lookback"]))
            bc[1].metric("Horizon",       int(best_o["horizon"]))
            bc[2].metric("Threshold",     f"{best_o['thresh']:.3f}")
            bc[3].metric("Entry Prob",    f"{best_o['entry']:.2f}")
            bc[4].metric("Exit Prob",     f"{best_o['exit']:.2f}")
            st.dataframe(df_res_o.head(25), use_container_width=True)
            st.download_button(
                "⬇ Optimizer-Ergebnisse als CSV",
                to_csv_eu(df_res_o),
                file_name="optimizer_results.csv", mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────
# HAUPT-HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: white;
  border: 1.5px solid #E2E8F0;
  border-radius: 10px;
  padding: 24px 28px 20px;
  margin-bottom: 28px;
  box-shadow: 0 2px 8px rgba(15,23,42,0.06);
">
  <div style="
    font-family: 'Inter', sans-serif; font-weight: 700;
    font-size: 28px; color: #0F172A; letter-spacing: -0.02em;
  ">📊 NEXUS — 2nd AI Model</div>
  <div style="
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: #64748B; margin-top: 6px; letter-spacing: 0.04em;
  ">
    Gradient Boosting Classifier  ·  Walk-Forward OOS  ·  Portfolio MC Forecast
  </div>
</div>
""", unsafe_allow_html=True)

results:    List[dict]                = []
all_trades: Dict[str, List[dict]]     = {}
all_dfs:    Dict[str, pd.DataFrame]   = {}
all_feat:   Dict[str, pd.DataFrame]   = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
)

# Optionsketten laden
options_live: Dict[str, pd.DataFrame] = {}
if use_chain_live:
    with st.spinner("Optionsketten werden geladen …"):
        po  = st.progress(0.0)
        tks = list(price_map.keys())
        for i, tk in enumerate(tks):
            try:
                df_o = price_map[tk]
                if df_o is None or df_o.empty: continue
                ch = get_equity_chain_aggregates_for_today(
                    tk, float(df_o["Close"].iloc[-1]),
                    atm_band_pct, int(n_expiries), int(max_days_to_exp)
                )
                if not ch.empty: options_live[tk] = ch
            except Exception: pass
            finally: po.progress((i+1)/max(1, len(tks)))

live_forecasts_run: List[dict] = []
_decide = lambda p,en,ex: (
    "Enter / Add" if p > en else ("Exit / Reduce" if p < ex else "Hold / No Trade")
)


# ─────────────────────────────────────────────────────────────
# Per-Ticker Analyse
# ─────────────────────────────────────────────────────────────
for ticker in TICKERS:
    if ticker not in price_map: continue
    df   = price_map[ticker]
    meta = meta_map.get(ticker, {})
    name = get_ticker_name(ticker)

    with st.expander(f"📈  {ticker}  —  {name}", expanded=False):
        try:
            ts  = df.index[-1]
            sfx = (f"  ·  Intraday bis {meta['tail_ts'].strftime('%H:%M')}"
                   if meta.get("tail_is_intraday") else "")
            st.caption(
                f"Letzter Datenpunkt: **{ts.strftime('%d.%m.%Y %H:%M %Z')}**{sfx}"
                f"  ·  {'Walk-Forward OOS' if use_walk_forward else 'In-Sample'}"
                f"  ·  Ziel: FutureRetExec > {THRESH:.3f} in {HORIZON} Tagen"
            )

            exog_tk = None
            if (use_chain_live and ticker in options_live
                    and not options_live[ticker].empty):
                ch = options_live[ticker].copy()
                ch.index = [df.index[-1].normalize()]
                exog_tk = ch

            feat, df_bt, trades, metrics = make_features_and_train(
                df, int(LOOKBACK), int(HORIZON), float(THRESH), MODEL_PARAMS,
                float(ENTRY_PROB), float(EXIT_PROB),
                init_capital=float(INIT_CAP_PER_TICKER),
                pos_frac=float(POS_FRAC),
                min_hold_days=int(MIN_HOLD_DAYS),
                cooldown_days=int(COOLDOWN_DAYS),
                exog_df=exog_tk,
                walk_forward=bool(use_walk_forward),
                wf_min_train=int(wf_min_train),
            )
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker]    = df_bt
            all_feat[ticker]   = feat

            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            live_act   = _decide(live_prob, float(ENTRY_PROB), float(EXIT_PROB))

            row = {
                "Stand": pd.Timestamp(feat.index[-1]).strftime("%d.%m.%Y %H:%M"),
                "Ticker": ticker, "Name": name,
                f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob, 4),
                "Signal": live_act,
                "Kurs": round(live_close, 4),
                "Bar": "intraday" if meta.get("tail_is_intraday") else "daily",
            }
            if use_chain_live and exog_tk is not None:
                for col in ["PCR_vol","PCR_oi","VOI_call","VOI_put",
                            "IV_skew_p_minus_c","VOL_tot","OI_tot"]:
                    v = exog_tk.iloc[-1].get(col, np.nan)
                    if pd.notna(v): row[col] = round(float(v), 4)
            live_forecasts_run.append(row)

            # ── KPI-Karten ──────────────────────────────────
            mn = metrics
            kpi_row([
                ("Netto Return",  _pct(mn["Strategy Net (%)"]),
                 "kpi-pos" if mn["Strategy Net (%)"] > 0 else "kpi-neg"),
                ("Buy & Hold",    _pct(mn["Buy & Hold Net (%)"]),
                 "kpi-pos" if mn["Buy & Hold Net (%)"] > 0 else "kpi-neg"),
                ("Sharpe Ratio",  f"{mn['Sharpe-Ratio']:.2f}",        "kpi-info"),
                ("Sortino Ratio", f"{mn['Sortino-Ratio']:.2f}"
                 if np.isfinite(mn["Sortino-Ratio"]) else "–",        "kpi-info"),
                ("Max Drawdown",  _pct(mn["Max Drawdown (%)"]),       "kpi-neg"),
                ("Winrate",       f"{mn['Winrate (%)']:.1f}%"
                 if np.isfinite(mn["Winrate (%)"]) else "–",          ""),
                ("Anzahl Trades", f"{mn['Number of Trades']}",        ""),
                ("Phase",         mn["Phase"],
                 "kpi-info" if mn["Phase"]=="Open" else ""),
            ])

            # ── Charts: Preis + Intraday ─────────────────────
            cc1, cc2 = st.columns(2)
            tdf_loc = pd.DataFrame(trades)

            with cc1:
                st.plotly_chart(
                    chart_price_signal(feat, trades, ticker),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            intra = get_intraday_last_n_sessions(ticker, 5, 10, intraday_interval)
            with cc2:
                if intra.empty:
                    st.info("Keine Intraday-Daten für diesen Ticker verfügbar.")
                else:
                    st.plotly_chart(
                        chart_intraday(intra, ticker, tdf_loc,
                                       intraday_chart_type, intraday_interval),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            # ── Equity-Kurve ────────────────────────────────
            st.plotly_chart(
                chart_equity(df_bt, ticker, float(INIT_CAP_PER_TICKER)),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # ── Trade-Log ───────────────────────────────────
            with st.expander(f"Trade-Log  –  {ticker}", expanded=False):
                if tdf_loc.empty:
                    st.info("Keine Trades in diesem Zeitraum.")
                else:
                    td = tdf_loc.copy()
                    td["Ticker"] = ticker; td["Name"] = name
                    if "Date" in td.columns:
                        td["Date"] = pd.to_datetime(td["Date"]).dt.strftime("%d.%m.%Y")
                    td = td.rename(columns={
                        "Prob": "Signal Prob", "HoldDays": "Hold (Tage)",
                        "Net P&L": "PnL (€)", "kum P&L": "Kum. PnL (€)",
                    })
                    desired = ["Ticker","Name","Date","Typ","Price","Shares",
                               "Signal Prob","Hold (Tage)","PnL (€)","Kum. PnL (€)","Fees"]
                    sc = [c for c in desired if c in td.columns]

                    def _row_color(row):
                        t = str(row.get("Typ","")).lower()
                        if "entry" in t:
                            return [f"background-color: {THEME['green_light']}"]*len(row)
                        if "exit" in t:
                            return [f"background-color: {THEME['red_light']}"]*len(row)
                        return [""]*len(row)

                    st.dataframe(
                        td[sc].style.format({
                            "Price":"{:.2f}", "Shares":"{:.4f}",
                            "Signal Prob":"{:.4f}",
                            "PnL (€)":"{:.2f}", "Kum. PnL (€)":"{:.2f}",
                            "Fees":"{:.2f}",
                        }).apply(_row_color, axis=1),
                        use_container_width=True,
                    )
                    st.download_button(
                        f"⬇ Trades {ticker} als CSV",
                        to_csv_eu(td[sc]),
                        file_name=f"trades_{ticker}.csv", mime="text/csv",
                        key=f"dl_tr_{ticker}",
                    )

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")
            import traceback
            with st.expander("Details zum Fehler"):
                st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────
# Live Forecast Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
        .drop_duplicates(subset=["Ticker"], keep="last")
        .sort_values(["Stand","Ticker"])
        .reset_index(drop=True)
    )
    live_df["Kursziel"] = (
        pd.to_numeric(live_df["Kurs"], errors="coerce") * (1 + float(THRESH))
    ).round(2)

    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"
    if prob_col not in live_df.columns:
        cand = [c for c in live_df.columns if c.startswith("P(") and c.endswith("d)")]
        if cand: prob_col = cand[0]

    section(f"LIVE FORECAST BOARD  —  {HORIZON}-Tage Prognose")

    # Übersicht
    n_enter = (live_df["Signal"] == "Enter / Add").sum()
    n_exit  = (live_df["Signal"] == "Exit / Reduce").sum()
    n_hold  = len(live_df) - n_enter - n_exit
    kpi_row([
        ("Signale gesamt", f"{len(live_df)}",   ""),
        ("▲ Kaufsignal",   f"{n_enter}",          "kpi-pos"),
        ("▼ Verkaufssignal",f"{n_exit}",          "kpi-neg"),
        ("◆ Halten",       f"{n_hold}",           ""),
    ])

    desired_lf = ["Stand","Ticker","Name",prob_col,"Signal",
                  "Kurs","Kursziel","Bar"]
    if use_chain_live:
        desired_lf = ["Stand","Ticker","Name",prob_col,"Signal",
                      "PCR_oi","PCR_vol","VOI_call","VOI_put",
                      "Kurs","Kursziel","Bar"]
    show_lf = [c for c in desired_lf if c in live_df.columns]

    def _board_style(row):
        a = str(row.get("Signal","")).lower()
        if "enter" in a:
            return [f"background-color: {THEME['green_light']}"]*len(row)
        if "exit" in a:
            return [f"background-color: {THEME['red_light']}"]*len(row)
        return [""]*len(row)

    fmt_lf = {prob_col: "{:.4f}", "Kurs": "{:.2f}", "Kursziel": "{:.2f}"}
    for c in ["PCR_oi","PCR_vol","VOI_call","VOI_put"]:
        if c in live_df.columns: fmt_lf[c] = "{:.3f}"

    st.dataframe(
        live_df[show_lf].style.format(fmt_lf).apply(_board_style, axis=1),
        use_container_width=True,
        height=min(620, 45 + 38*len(live_df)),
    )
    st.download_button(
        "⬇ Forecasts als CSV",
        to_csv_eu(live_df),
        file_name=f"live_forecasts_{HORIZON}d.csv", mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# Summary & Portfolio
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (
        summary_df["Net P&L (€)"] / float(INIT_CAP_PER_TICKER) * 100
    )
    if "Phase" in summary_df.columns:
        summary_df["Phase"] = (
            summary_df["Phase"].astype(str).str.strip().str.lower()
            .map(lambda x: "Open" if x=="open" else
                           ("Flat" if x=="flat" else x.capitalize()))
        )

    total_net   = float(summary_df["Net P&L (€)"].sum())
    total_fees  = float(summary_df["Fees (€)"].sum())
    total_gross = total_net + total_fees
    total_cap   = float(INIT_CAP_PER_TICKER) * len(summary_df)

    section("PORTFOLIO SUMMARY")
    kpi_row([
        ("Netto P&L gesamt",  _eur(total_net),
         "kpi-pos" if total_net > 0 else "kpi-neg"),
        ("Brutto P&L gesamt", _eur(total_gross),
         "kpi-pos" if total_gross > 0 else "kpi-neg"),
        ("Handelskosten",     f"–{total_fees:,.2f} €",    "kpi-warn"),
        ("Trades gesamt",
         f"{int(summary_df['Number of Trades'].sum())}",   ""),
        ("Ø CAGR",
         f"{summary_df['CAGR (%)'].dropna().mean():.2f}%"
         if "CAGR (%)" in summary_df else "–",             "kpi-info"),
        ("Ø Winrate",
         f"{summary_df['Winrate (%)'].dropna().mean():.1f}%"
         if "Winrate (%)" in summary_df else "–",          ""),
    ])

    def _num_col(v):
        try:
            fv = float(v)
            if fv > 0: return f"color: {THEME['green']}; font-weight: 600"
            if fv < 0: return f"color: {THEME['red']}; font-weight: 600"
        except: pass
        return ""

    def _phase_col(v):
        if str(v) == "Open":
            return (f"background-color: {THEME['blue_light']}; "
                    f"color: {THEME['blue']}; font-weight: 600; "
                    f"border-radius: 4px; padding: 2px 8px;")
        return (f"background-color: {THEME['grid']}; "
                f"color: {THEME['text_muted']}; "
                f"border-radius: 4px; padding: 2px 8px;")

    pct_cols = ["Strategy Net (%)","Strategy Gross (%)","Buy & Hold Net (%)","Net P&L (%)","CAGR (%)"]
    styled_s = summary_df.style.format({
        "Strategy Net (%)":   "{:.2f}",
        "Strategy Gross (%)": "{:.2f}",
        "Buy & Hold Net (%)": "{:.2f}",
        "Volatility (%)":     "{:.2f}",
        "Sharpe-Ratio":       "{:.2f}",
        "Sortino-Ratio":      "{:.2f}",
        "Max Drawdown (%)":   "{:.2f}",
        "Calmar-Ratio":       "{:.2f}",
        "Fees (€)":           "{:.2f}",
        "Net P&L (%)":        "{:.2f}",
        "Net P&L (€)":        "{:.2f}",
        "CAGR (%)":           "{:.2f}",
        "Winrate (%)":        "{:.2f}",
        "InitCap (€)":        "{:.0f}",
    }).applymap(_num_col, subset=[c for c in pct_cols if c in summary_df.columns])

    if "Phase" in summary_df.columns:
        styled_s = styled_s.applymap(_phase_col, subset=["Phase"])

    st.dataframe(styled_s, use_container_width=True)
    st.download_button(
        "⬇ Summary als CSV",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary.csv", mime="text/csv",
    )

    # ── Offene Positionen ────────────────────────────────────
    section("OFFENE POSITIONEN")
    open_positions = []
    for ticker, tr in all_trades.items():
        if tr and tr[-1]["Typ"] == "Entry":
            le  = next(t for t in reversed(tr) if t["Typ"]=="Entry")
            lc  = float(all_dfs[ticker]["Close"].iloc[-1])
            upnl= (lc - float(le["Price"])) * float(le["Shares"])
            open_positions.append({
                "Ticker":          ticker,
                "Name":            get_ticker_name(ticker),
                "Einstiegsdatum":  pd.to_datetime(le["Date"]).strftime("%d.%m.%Y"),
                "Einstiegspreis":  round(float(le["Price"]), 2),
                "Aktueller Kurs":  round(lc, 2),
                "Signal Prob":     round(float(all_feat[ticker]["SignalProb"].iloc[-1]), 4),
                "Unr. PnL (€)":    round(upnl, 2),
            })

    if open_positions:
        op_df = pd.DataFrame(open_positions).sort_values("Einstiegsdatum", ascending=False)

        def _upnl_c(v):
            try:
                return (f"color: {THEME['green']}; font-weight: 600"
                        if float(v) >= 0
                        else f"color: {THEME['red']}; font-weight: 600")
            except: return ""

        st.dataframe(
            op_df.style.format({
                "Einstiegspreis": "{:.2f}",
                "Aktueller Kurs": "{:.2f}",
                "Signal Prob":    "{:.4f}",
                "Unr. PnL (€)":  "{:.2f}",
            }).applymap(_upnl_c, subset=["Unr. PnL (€)"]),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Offene Positionen als CSV",
            to_csv_eu(op_df),
            file_name="open_positions.csv", mime="text/csv",
        )
    else:
        st.success("✅ Keine offenen Positionen vorhanden.")

    # ── Round-Trips ──────────────────────────────────────────
    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        section("ABGESCHLOSSENE TRADES (ROUND-TRIPS)")
        rt_df["Entry Datum"] = pd.to_datetime(rt_df["Entry Datum"])
        rt_df["Exit Datum"]  = pd.to_datetime(rt_df["Exit Datum"])

        ret = pd.to_numeric(rt_df.get("Return (%)"),    errors="coerce").dropna()
        pnl = pd.to_numeric(rt_df.get("PnL Netto (€)"), errors="coerce").dropna()

        kpi_row([
            ("Anzahl Trades",  f"{len(ret)}",                                                 ""),
            ("Winrate",        f"{100*(ret>0).mean():.1f}%" if len(ret) else "–",             "kpi-pos"),
            ("Ø Return",       f"{ret.mean():.2f}%" if len(ret) else "–",                     "kpi-info"),
            ("Median Return",  f"{ret.median():.2f}%" if len(ret) else "–",                   ""),
            ("Std-Abweichung", f"{ret.std():.2f}%" if len(ret) else "–",                      "kpi-warn"),
        ])

        rt_disp = rt_df.copy()
        rt_disp["Entry Datum"] = rt_disp["Entry Datum"].dt.strftime("%d.%m.%Y")
        rt_disp["Exit Datum"]  = rt_disp["Exit Datum"].dt.strftime("%d.%m.%Y")
        if "Hold (Tage)" in rt_disp.columns:
            rt_disp["Hold (Tage)"] = rt_disp["Hold (Tage)"].round().astype("Int64")

        def _ret_style(v):
            try:
                return (f"color: {THEME['green']}; font-weight: 600"
                        if float(v) >= 0
                        else f"color: {THEME['red']}; font-weight: 600")
            except: return ""

        st.dataframe(
            rt_disp.sort_values("Exit Datum", ascending=False).style.format({
                "Stücke":          "{:.4f}",
                "Entry Preis":     "{:.2f}",
                "Exit Preis":      "{:.2f}",
                "PnL Netto (€)":   "{:.2f}",
                "Gebühren (€)":    "{:.2f}",
                "Return (%)":      "{:.2f}",
                "Entry Prob":      "{:.4f}",
                "Exit Prob":       "{:.4f}",
            }).applymap(_ret_style, subset=["Return (%)","PnL Netto (€)"]),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Round-Trips als CSV",
            to_csv_eu(rt_disp),
            file_name="round_trips.csv", mime="text/csv",
        )

        # ── Return-Verteilung ────────────────────────────────
        section("RETURN-VERTEILUNG DER TRADES")
        bins_rt = st.slider("Anzahl Histogramm-Balken", 10, 120, 30, step=5)
        hc1, hc2 = st.columns(2)
        with hc1:
            if not ret.empty:
                st.plotly_chart(
                    chart_histogram(ret, "Return (%)",
                                    "Trade-Renditen (%)", bins_rt),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
        with hc2:
            if not pnl.empty:
                st.plotly_chart(
                    chart_histogram(pnl, "PnL Netto (€)",
                                    "Trade P&L Netto (€)", bins_rt),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # ── Portfolio Equal-Weight ──────────────────────────────
    section("PORTFOLIO  —  EQUAL-WEIGHT PERFORMANCE")
    price_series = []
    for tk, dfbt in all_dfs.items():
        if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
            s = dfbt["Close"].copy(); s.name = tk
            try:
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
            except Exception: pass
            s.index = pd.to_datetime(s.index).normalize()
            price_series.append(s)

    prices_port = pd.DataFrame()
    port_ret    = pd.Series(dtype=float)

    if len(price_series) < 2:
        st.info("Mindestens 2 Ticker mit Kurshistorie erforderlich.")
    else:
        prices_port = pd.concat(price_series, axis=1, join="outer").sort_index()
        rets_ew     = prices_port.pct_change()
        valid       = rets_ew.notna().sum(axis=1) >= 2
        rets2       = rets_ew.loc[valid].copy()
        if not rets2.empty:
            w_row    = rets2.notna().astype(float)
            w_row    = w_row.div(w_row.sum(axis=1), axis=0)
            port_ret = (rets2.fillna(0.0) * w_row).sum(axis=1).dropna()

        if port_ret.empty:
            st.info("Portfolio-Returns sind leer.")
        else:
            ann_r    = (1+port_ret).prod()**(252/len(port_ret)) - 1
            ann_v    = port_ret.std(ddof=0)*np.sqrt(252)
            sh_p     = (port_ret.mean()/(port_ret.std(ddof=0)+1e-12))*np.sqrt(252)
            nav0     = float(INIT_CAP_PER_TICKER) * len(summary_df)
            nav      = nav0 * (1+port_ret).cumprod()
            max_dd_p = float(((nav/nav.cummax())-1).min())

            kpi_row([
                ("Return p.a.",    f"{ann_r*100:.2f}%",
                 "kpi-pos" if ann_r > 0 else "kpi-neg"),
                ("Volatilität p.a.", f"{ann_v*100:.2f}%", "kpi-warn"),
                ("Sharpe Ratio",   f"{sh_p:.2f}",          "kpi-info"),
                ("Max. Drawdown",  f"{max_dd_p*100:.2f}%", "kpi-neg"),
            ])
            st.plotly_chart(
                chart_portfolio_nav(nav),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.download_button(
                "⬇ Portfolio-Returns (täglich) als CSV",
                to_csv_eu(pd.DataFrame({
                    "Datum": port_ret.index,
                    "Portfolio Return": port_ret.values,
                })),
                file_name="portfolio_returns_daily.csv", mime="text/csv",
            )

    # ── Korrelationsmatrix ───────────────────────────────────
    if not prices_port.empty and prices_port.shape[1] >= 2:
        section("PORTFOLIO-KORRELATION")
        ka, kb, kc, kd = st.columns(4)
        freq_lbl    = ka.selectbox("Frequenz",        ["täglich","wöchentlich","monatlich"])
        corr_meth   = kb.selectbox("Methode",         ["Pearson","Spearman","Kendall"])
        min_obs_c   = kc.slider("Min. gemeinsame Datenpunkte", 10, 300, 20, step=5)
        use_ffill_c = kd.checkbox("Lücken auffüllen (FFill)", value=True)

        pc = prices_port.copy()
        if use_ffill_c:        pc = pc.ffill()
        if freq_lbl == "wöchentlich": pc = pc.resample("W").last()
        elif freq_lbl == "monatlich": pc = pc.resample("M").last()

        rc = pc.pct_change().dropna(how="all")
        rc = rc[rc.notna().sum()[rc.notna().sum() >= int(min_obs_c)].index]

        if rc.shape[1] >= 2:
            corr = rc.corr(method=corr_meth.lower())
            st.plotly_chart(
                chart_corr(corr),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            m = corr.values.copy(); np.fill_diagonal(m, np.nan)
            off = m[np.isfinite(m)]
            cov_c    = rc.cov()
            vols_c   = np.sqrt(np.diag(cov_c.values))
            n_c      = len(vols_c); w_c = np.ones(n_c)/n_c
            pv       = float(w_c @ cov_c.values @ w_c)
            denom    = float(np.sum((w_c[:,None]*w_c[None,:]*vols_c[:,None]*vols_c[None,:])))+1e-12
            diag_p   = float(np.sum((w_c**2)*(vols_c**2)))
            ipc      = (pv - diag_p)/denom

            kpi_row([
                ("Ø Paar-Korrelation", f"{np.mean(off):.2f}"   if off.size else "–", ""),
                ("Median",             f"{np.median(off):.2f}" if off.size else "–", ""),
                ("Streuung (σ)",       f"{np.std(off):.2f}"    if off.size else "–", ""),
                ("IPC (normiert)",     f"{ipc:.2f}"            if np.isfinite(ipc) else "–", "kpi-info"),
            ])
            st.caption(
                f"Basis: **{len(rc)}** gemeinsame Datenpunkte  ·  "
                f"Frequenz: {freq_lbl}  ·  Methode: {corr_meth}"
            )
            st.download_button(
                "⬇ Korrelationsmatrix als CSV",
                to_csv_eu(corr.reset_index().rename(columns={"index":"Ticker"})),
                file_name="correlation_matrix.csv", mime="text/csv",
            )
        else:
            st.info("Zu wenige gemeinsame Datenpunkte für Korrelationsmatrix.")

    # ── Portfolio Forecast (MC) ──────────────────────────────
    section(f"PORTFOLIO-PROGNOSE  —  {int(FORECAST_DAYS)} Tage  ·  MC = {int(MC_SIMS):,}")

    rows_fc = []
    for tk, feat in all_feat.items():
        est = estimate_expected_return(feat, int(FORECAST_DAYS), float(THRESH))
        if not est: continue
        rows_fc.append({
            "Ticker":            tk,
            "Name":              get_ticker_name(tk),
            "P (Signal)":        est["p"],
            "μ1 (Ziel=1)":       est["mu1"],
            "μ0 (Ziel=0)":       est["mu0"],
            f"E[r {FORECAST_DAYS}d]": est["exp_ret"],
        })

    if not rows_fc:
        st.info("Prognose: Nicht genug Daten für Schätzung.")
    else:
        fc_df  = pd.DataFrame(rows_fc).set_index("Ticker")
        ercol  = f"E[r {FORECAST_DAYS}d]"

        st.dataframe(
            fc_df.sort_values(ercol, ascending=False).style.format({
                "P (Signal)":  "{:.4f}",
                "μ1 (Ziel=1)": "{:.4f}",
                "μ0 (Ziel=0)": "{:.4f}",
                ercol:         "{:.4f}",
            }).applymap(_num_col, subset=[ercol]),
            use_container_width=True,
        )

        exp_rets = fc_df[ercol].astype(float).dropna()
        if not prices_port.empty:
            exp_rets = exp_rets.reindex(
                prices_port.columns.intersection(exp_rets.index)
            ).dropna()

        if len(exp_rets) < 2:
            st.info("Für Portfolio-MC werden mindestens 2 Ticker benötigt.")
        else:
            if not prices_port.empty:
                dr    = prices_port[exp_rets.index].pct_change().dropna(how="all")
                cov_h = dr.cov(min_periods=60) * float(FORECAST_DAYS)
            else:
                cov_h = pd.DataFrame(
                    np.diag(np.full(len(exp_rets), (0.02)**2)),
                    index=exp_rets.index, columns=exp_rets.index
                )

            nav0_fc = float(INIT_CAP_PER_TICKER) * len(summary_df)
            out_mc  = portfolio_mc(exp_rets, cov_h, nav0_fc,
                                    sims=int(MC_SIMS), seed=42)

            kpi_row([
                ("Erwartete Rendite EW",
                 f"{exp_rets.mean()*100:.2f}%",                              "kpi-info"),
                ("5%-Quantil",
                 f"{out_mc['q05']*100:.2f}%  ·  {out_mc['nq05']:,.0f} €",   "kpi-neg"),
                ("Median (50%)",
                 f"{out_mc['q50']*100:.2f}%  ·  {out_mc['nq50']:,.0f} €",   ""),
                ("95%-Quantil",
                 f"{out_mc['q95']*100:.2f}%  ·  {out_mc['nq95']:,.0f} €",   "kpi-pos"),
            ])

            st.plotly_chart(
                chart_mc_histogram(
                    out_mc["port_rets"], out_mc["q05"],
                    out_mc["q50"], out_mc["q95"],
                    int(FORECAST_DAYS), int(MC_SIMS)
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

else:
    st.warning("⚠ Noch keine Ergebnisse verfügbar. Bitte Ticker-Eingaben und Datenabdeckung prüfen.")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
  margin-top: 52px;
  padding: 18px 24px;
  background: white;
  border: 1px solid {THEME['border']};
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
">
  <div style="
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {THEME['text_muted']};
    letter-spacing: 0.06em;
  ">NEXUS · 2nd AI Model · v2.0</div>
  <div style="
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {THEME['text_muted']};
    letter-spacing: 0.04em;
  ">Gradient Boosting · Walk-Forward OOS · Monte Carlo Forecast</div>
</div>
""", unsafe_allow_html=True)
