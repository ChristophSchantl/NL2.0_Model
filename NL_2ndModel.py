from pathlib import Path

code = r'''# streamlit_app_v3_1.py
# ─────────────────────────────────────────────────────────────
# NEXUS — 2nd AI Model v3.1
# Signal-basierte Strategie · pro Ticker separates Konto
# Robusteres Feature-Set · Walk-Forward OOS als Standard
# Probability Calibration optional · Portfolio auf Strategie-Returns
# Optionen als Modellinput vollständig entfernt
# ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import traceback
from math import sqrt
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


# ─────────────────────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────────────────────
THEME = {
    "bg":         "#F7F8FA",
    "bg_card":    "#FFFFFF",
    "bg_panel":   "#F1F4F8",
    "accent1":    "#C8A96B",
    "accent2":    "#D97706",
    "accent3":    "#2563EB",
    "accent4":    "#7C3AED",
    "red":        "#DC2626",
    "green":      "#16A34A",
    "text":       "#111827",
    "muted":      "#6B7280",
    "border":     "#E5E7EB",
    "grid":       "#EAECEF",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(
        family="'Inter', 'Segoe UI', sans-serif",
        color=THEME["text"],
        size=12
    ),
    xaxis=dict(
        gridcolor=THEME["grid"],
        gridwidth=1,
        showline=True,
        linecolor=THEME["border"],
        zeroline=False,
        tickfont=dict(size=11, color=THEME["muted"]),
        title_font=dict(size=12, color=THEME["muted"]),
    ),
    yaxis=dict(
        gridcolor=THEME["grid"],
        gridwidth=1,
        showline=True,
        linecolor=THEME["border"],
        zeroline=False,
        tickfont=dict(size=11, color=THEME["muted"]),
        title_font=dict(size=12, color=THEME["muted"]),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        font_color=THEME["text"],
        font_family="'Inter', sans-serif",
        bordercolor=THEME["border"],
    ),
    margin=dict(t=56, b=44, l=58, r=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0)",
        font=dict(size=11, color=THEME["muted"]),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

st.set_page_config(
    page_title="NEXUS — 2nd AI Model v3.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCAL_TZ = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 6
pd.options.display.float_format = "{:,.4f}".format

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

:root {
  --bg: #F7F8FA;
  --bg-card: #FFFFFF;
  --bg-panel: #F1F4F8;
  --primary: #C8A96B;
  --primary-dark: #B28A45;
  --secondary: #2563EB;
  --violet: #7C3AED;
  --green: #16A34A;
  --red: #DC2626;
  --text: #111827;
  --muted: #6B7280;
  --border: #E5E7EB;
  --grid: #EAECEF;
  --shadow: 0 6px 22px rgba(17, 24, 39, 0.06);
  --radius: 18px;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
}

.stApp {
  background: linear-gradient(180deg, #FAFBFC 0%, #F5F7FA 100%) !important;
  color: var(--text) !important;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 2rem !important;
  max-width: 1500px !important;
}

[data-testid="stSidebar"] {
  background: #FFFFFF !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

div[data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stNumberInput input,
.stDateInput input,
textarea {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

.stButton > button,
.stDownloadButton > button {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
  font-weight: 600 !important;
  padding: 0.58rem 1rem !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--primary) !important;
  color: var(--primary-dark) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid*="primary"] {
  background: linear-gradient(135deg, #C8A96B 0%, #B28A45 100%) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 8px 18px rgba(200,169,107,0.22) !important;
}

[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 18px 18px !important;
  box-shadow: var(--shadow) !important;
}

.stDataFrame, .stTable {
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  overflow: hidden !important;
  background: #FFFFFF !important;
  box-shadow: var(--shadow) !important;
}

details {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  margin-bottom: 14px !important;
  box-shadow: var(--shadow) !important;
}

.nexus-header {
  font-family: 'Playfair Display', serif !important;
  font-weight: 700 !important;
  font-size: 40px !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
  line-height: 1.0;
}
.nexus-sub {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  color: var(--muted) !important;
  font-weight: 500 !important;
  margin-top: 8px;
}

.section-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0 14px 0;
  margin: 28px 0 16px 0;
}
.section-bar-label {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--muted);
}
.section-bar-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}
.section-bar-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: linear-gradient(135deg, #C8A96B 0%, #B28A45 100%);
  box-shadow: 0 0 0 4px rgba(200,169,107,0.12);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px;
  margin: 16px 0 20px 0;
}
.kpi-box {
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: var(--shadow);
}
.kpi-label {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  margin-bottom: 8px;
}
.kpi-value {
  font-size: 22px;
  font-weight: 750;
  color: var(--text);
}
.kpi-pos .kpi-value { color: var(--green); }
.kpi-neg .kpi-value { color: var(--red); }
.kpi-info .kpi-value { color: var(--secondary); }
.kpi-warn .kpi-value { color: var(--primary-dark); }

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 10px 14px !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(
        index=False,
        sep=";",
        decimal=",",
        date_format="%d.%m.%Y",
        float_format=float_format
    ).encode("utf-8-sig")


def _normalize_tickers(items: List[str]) -> List[str]:
    cleaned = []
    for x in items or []:
        if not isinstance(x, str):
            continue
        s = x.strip().upper()
        if s:
            cleaned.append(s)
    return list(dict.fromkeys(cleaned))


def parse_ticker_csv(path_or_buffer) -> List[str]:
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception:
        df = pd.read_csv(path_or_buffer, sep=";")
    if df.empty:
        return []
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("ticker", "symbol", "symbols", "code"):
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
        try:
            info = tk.get_info()
        except Exception:
            info = getattr(tk, "info", {}) or {}
        for k in ("shortName", "longName", "displayName", "companyName", "name"):
            if k in info and info[k]:
                return str(info[k])
    except Exception:
        pass
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
        f'<div class="kpi-box {cls}"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div></div>'
        for lbl, val, cls in items
    )
    st.markdown(f'<div class="kpi-grid">{parts}</div>', unsafe_allow_html=True)


def _pct(v, decimals=2):
    if not np.isfinite(v):
        return "–"
    s = "+" if v > 0 else ""
    return f"{s}{v:.{decimals}f}%"


def _eur(v):
    if not np.isfinite(v):
        return "–"
    s = "+" if v > 0 else ""
    return f"{s}{v:,.2f}€"


def _apply_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**PLOTLY_BASE, height=height)
    fig.update_xaxes(
        showspikes=True,
        spikecolor=THEME["muted"],
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor=THEME["muted"],
        spikethickness=1,
        spikedash="dot",
    )
    return fig


def _safe_div(a, b, default=np.nan):
    try:
        if b is None or abs(float(b)) < 1e-12:
            return default
        return float(a) / float(b)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 20px;border-bottom:1px solid #E5E7EB;margin-bottom:18px;">
      <div style="font-family:'Playfair Display',serif;font-weight:700;font-size:24px;color:#111827;letter-spacing:-0.02em;">NEXUS</div>
      <div style="font-family:'Inter',sans-serif;font-size:11px;color:#B28A45;letter-spacing:0.12em;margin-top:4px;font-weight:700;">
        2ND AI MODEL · v3.1
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
                try:
                    collected += parse_ticker_csv(up)
                except Exception as e:
                    st.error(f"{up.name}: {e}")
        base = _normalize_tickers(collected)
        extra = st.text_input("Weitere Ticker", value="", key="extra_csv")
        extras = _normalize_tickers([t for t in extra.split(",") if t.strip()]) if extra else []
        tickers_final = _normalize_tickers(base + extras)
        if tickers_final:
            st.caption(f"**{len(tickers_final)}** Ticker")
            max_n = st.number_input("Max (0=alle)", 0, len(tickers_final), 0, step=10)
            if max_n:
                tickers_final = tickers_final[:int(max_n)]
            tickers_final = st.multiselect("Auswahl", tickers_final, default=tickers_final)

    if not tickers_final:
        tickers_final = _normalize_tickers(["REGN", "VOW3.DE", "LULU", "REI", "DDL"])

    st.download_button(
        "⬇ Ticker-CSV",
        to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
        file_name="tickers.csv",
        mime="text/csv",
        use_container_width=True
    )
    TICKERS = tickers_final

    st.divider()
    st.markdown("**ZEITRAUM**")
    col_d1, col_d2 = st.columns(2)
    START_DATE = col_d1.date_input("Von", pd.to_datetime("2022-01-01"))
    END_DATE = col_d2.date_input("Bis", pd.to_datetime(datetime.now(LOCAL_TZ).date()))

    st.divider()
    st.markdown("**MODELL**")
    c1s, c2s = st.columns(2)
    LOOKBACK = c1s.number_input("Lookback", 10, 252, 35, step=5)
    HORIZON = c2s.number_input("Horizon (Hold-Tage)", 1, 20, 5)
    THRESH = st.number_input("Target Threshold", 0.0, 0.20, 0.030, step=0.005, format="%.3f")
    ENTRY_PROB = st.slider("Entry Prob", 0.0, 1.0, 0.62, step=0.01)
    EXIT_PROB = st.slider("Exit Prob", 0.0, 1.0, 0.48, step=0.01)
    if EXIT_PROB >= ENTRY_PROB:
        st.error("⚠ Exit < Entry required")
        st.stop()
    MIN_HOLD_DAYS = st.number_input("Min. Hold (Tage)", 0, 252, 5, step=1)
    COOLDOWN_DAYS = st.number_input("Cooldown (Tage)", 0, 252, 0, step=1)

    st.divider()
    st.markdown("**EXECUTION**")
    c3s, c4s = st.columns(2)
    COMMISSION = c3s.number_input("Commission", 0.0, 0.02, 0.004, step=0.0001, format="%.4f")
    SLIPPAGE_BPS = c4s.number_input("Slippage bp", 0, 50, 5, step=1)
    POS_FRAC = st.slider("Position Size", 0.1, 1.0, 1.0, step=0.1)
    INIT_CAP_PER_TICKER = st.number_input(
        "Capital/Ticker (€)", 1000.0, 1_000_000.0, 10_000.0, step=1000.0, format="%.0f"
    )

    st.divider()
    st.markdown("**INTRADAY**")
    use_live = st.checkbox("Intraday Tail", value=True)
    intraday_interval = st.selectbox("Intervall", ["1m", "2m", "5m", "15m"], index=2)
    fallback_last_session = st.checkbox("Session Fallback", value=False)
    exec_mode = st.selectbox("Execution", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
    moc_cutoff_min = st.number_input("MOC Cutoff (min)", 5, 60, 15, step=5)
    intraday_chart_type = st.selectbox("Intraday Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

    st.divider()
    st.markdown("**ML PARAMS**")
    c5s, c6s = st.columns(2)
    max_iter = c5s.number_input("max_iter", 50, 600, 220, step=10)
    learning_rate = c6s.number_input("lr", 0.01, 0.50, 0.05, step=0.01, format="%.2f")
    c7s, c8s = st.columns(2)
    max_depth = c7s.number_input("max_depth", 2, 12, 4, step=1)
    min_leaf = c8s.number_input("min_samples_leaf", 5, 100, 20, step=1)
    calibration_on = st.checkbox("Probability Calibration", value=True)
    MODEL_PARAMS = dict(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_leaf),
        random_state=42
    )

    st.divider()
    st.markdown("**WALK-FORWARD / OOS**")
    use_walk_forward = st.checkbox("Walk-Forward OOS", value=True)
    wf_min_train = st.number_input("WF min_train Bars", 60, 700, 180, step=10)
    wf_refit_step = st.number_input("WF Refit Every n Bars", 1, 50, 5, step=1)

    st.divider()
    st.markdown("**PORTFOLIO FORECAST**")
    FORECAST_DAYS = st.number_input("Forecast Horizon (Tage)", 1, 30, 7, step=1)
    MC_SIMS = st.number_input("MC Simulationen", 200, 5000, 1500, step=100)

    st.divider()
    col_r1, col_r2 = st.columns(2)
    if col_r1.button("🗑 Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if col_r2.button("↺ Rerun", use_container_width=True):
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
def get_price_data_tail_intraday(
    ticker,
    start_date,
    end_date,
    use_tail=True,
    interval="5m",
    fallback_last_session=False,
    exec_mode_key="Next Open (backtest+live)",
    moc_cutoff_min_val=15
):
    tk = yf.Ticker(ticker)

    start_pad = pd.Timestamp(start_date) - pd.Timedelta(days=450)
    end_pad = pd.Timestamp(end_date) + pd.Timedelta(days=5)

    df = tk.history(
        start=start_pad.strftime("%Y-%m-%d"),
        end=end_pad.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        actions=False
    )
    if df.empty:
        raise ValueError(f"Keine Daten: {ticker}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)
    df = df.sort_index().drop_duplicates()

    meta = {"tail_is_intraday": False, "tail_ts": None}

    if not use_tail:
        df.dropna(subset=["High", "Low", "Close", "Open"], inplace=True)
        return df, meta

    try:
        intraday = tk.history(period="1d", interval=interval, auto_adjust=True, actions=False, prepost=False)
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
        intraday = intraday.loc[:datetime.now(LOCAL_TZ) - timedelta(minutes=int(moc_cutoff_min_val))]

    if intraday.empty and fallback_last_session:
        try:
            intr5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
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
            "Open": float(intraday["Open"].iloc[0]),
            "High": float(intraday["High"].max()),
            "Low": float(intraday["Low"].min()),
            "Close": float(lb["Close"]),
            "Volume": float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan
        }
        if dk in df.index:
            for k, v in row.items():
                df.loc[dk, k] = v
        else:
            df.loc[dk] = row
        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"] = lb.name

    df.dropna(subset=["High", "Low", "Close", "Open"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=180)
def get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval="5m"):
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr = intr.sort_index()
    keep = set(pd.Index(intr.index.normalize().unique())[-sessions:])
    return intr.loc[intr.index.normalize().isin(keep)].copy()


def load_all_prices(tickers, start, end, use_tail, interval, fallback_last, exec_key, moc_cutoff):
    price_map, meta_map = {}, {}
    if not tickers:
        return price_map, meta_map

    with st.spinner(f"📡 Kursdaten · {len(tickers)} Ticker"):
        prog = st.progress(0.0)
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
            fmap = {
                ex.submit(
                    get_price_data_tail_intraday,
                    tk, start, end, use_tail, interval, fallback_last, exec_key, int(moc_cutoff)
                ): tk for tk in tickers
            }
            done = 0
            for fut in as_completed(fmap):
                tk = fmap[fut]
                try:
                    df_full, meta = fut.result()
                    df_use = df_full.loc[str(start):str(end)].copy()
                    if not df_use.empty:
                        price_map[tk] = df_use
                        meta_map[tk] = meta
                except Exception as e:
                    st.error(f"❌ {tk}: {e}")
                finally:
                    done += 1
                    prog.progress(done / len(tickers))
    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────
def add_price_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    feat = df.copy().sort_index()

    feat["ret_1d"] = feat["Close"].pct_change(1)
    feat["ret_5d"] = feat["Close"].pct_change(5)
    feat["ret_20d"] = feat["Close"].pct_change(20)

    feat["vol_20d"] = feat["ret_1d"].rolling(20).std() * np.sqrt(252)
    feat["vol_60d"] = feat["ret_1d"].rolling(60).std() * np.sqrt(252)

    tr1 = (feat["High"] - feat["Low"]).abs()
    tr2 = (feat["High"] - feat["Close"].shift(1)).abs()
    tr3 = (feat["Low"] - feat["Close"].shift(1)).abs()
    feat["TR"] = np.maximum.reduce([tr1.fillna(0).values, tr2.fillna(0).values, tr3.fillna(0).values])
    feat["ATR_14"] = pd.Series(feat["TR"], index=feat.index).rolling(14).mean()
    feat["ATR_pct"] = feat["ATR_14"] / feat["Close"]

    feat["Range"] = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["Range_pct"] = feat["Range"] / feat["Close"]

    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"] = feat["Low"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeClose"] = feat["Close"].rolling(lookback).apply(slope, raw=True)

    feat["MA20"] = feat["Close"].rolling(20).mean()
    feat["MA50"] = feat["Close"].rolling(50).mean()
    feat["dist_ma20"] = feat["Close"] / feat["MA20"] - 1
    feat["dist_ma50"] = feat["Close"] / feat["MA50"] - 1

    feat["high_20"] = feat["High"].rolling(20).max()
    feat["low_20"] = feat["Low"].rolling(20).min()
    feat["dist_high20"] = feat["Close"] / feat["high_20"] - 1
    feat["dist_low20"] = feat["Close"] / feat["low_20"] - 1

    feat["gap_open"] = feat["Open"] / feat["Close"].shift(1) - 1
    feat["oc_ret"] = feat["Close"] / feat["Open"] - 1
    feat["hl_spread"] = (feat["High"] - feat["Low"]) / feat["Close"]
    feat["vol_rel20"] = feat["Volume"] / feat["Volume"].rolling(20).median()
    feat["vol_chg_5d"] = feat["Volume"].pct_change(5)

    return feat


def make_features(df: pd.DataFrame, lookback: int, horizon: int):
    if len(df) < max(lookback + horizon + 25, 120):
        raise ValueError("Zu wenige Bars.")
    feat = add_price_features(df.copy(), lookback)

    # Saubere Execution-Definition:
    # Signal am Ende von t
    # Entry am Open von t+1
    # Exit am Open von t+1+horizon
    feat["EntryOpen"] = feat["Open"].shift(-1)
    feat["ExitOpen"] = feat["Open"].shift(-(horizon + 1))
    feat["FutureRetExec"] = feat["ExitOpen"] / feat["EntryOpen"] - 1

    return feat


BASE_FEATURES = [
    "Range_pct",
    "SlopeHigh",
    "SlopeLow",
    "SlopeClose",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_60d",
    "ATR_pct",
    "dist_ma20",
    "dist_ma50",
    "dist_high20",
    "dist_low20",
    "gap_open",
    "oc_ret",
    "hl_spread",
    "vol_rel20",
    "vol_chg_5d",
]


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────
def backtest_next_open(
    df, entry_thr, exit_thr, commission, slippage_bps,
    init_cap, pos_frac, min_hold_days=0, cooldown_days=0
):
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte.")

    cash_g = init_cap
    cash_n = init_cap
    shares = 0.0
    in_pos = False
    cb_g = 0.0
    cb_n = 0.0
    last_ei = None
    last_xi = None
    eq_g, eq_n, trades = [], [], []
    cum = 0.0

    for i in range(n):
        if i > 0:
            ot = float(df["Open"].iloc[i])
            sb = ot * (1 + slippage_bps / 10000.0)
            ss = ot * (1 - slippage_bps / 10000.0)
            pp = float(df["SignalProb"].iloc[i - 1])
            de = df.index[i]

            cool = True
            if (not in_pos) and cooldown_days > 0 and last_xi is not None:
                cool = (i - last_xi) >= int(cooldown_days)

            if (not in_pos) and (pp > entry_thr) and cool:
                inv = cash_n * float(pos_frac)
                fee = inv * float(commission)
                sh = max((inv - fee) / sb, 0.0)
                if sh > 0 and (sh * sb + fee) <= cash_n + 1e-6:
                    shares = sh
                    cb_g = sh * sb
                    cb_n = sh * sb + fee
                    cash_g -= cb_g
                    cash_n -= cb_n
                    in_pos = True
                    last_ei = i
                    trades.append({
                        "Date": de,
                        "Typ": "Entry",
                        "Price": round(sb, 4),
                        "Shares": round(sh, 4),
                        "Gross P&L": 0.0,
                        "Fees": round(fee, 2),
                        "Net P&L": 0.0,
                        "kum P&L": round(cum, 2),
                        "Prob": round(pp, 4),
                        "HoldDays": np.nan
                    })

            elif in_pos and pp < exit_thr:
                held = (i - last_ei) if last_ei is not None else 0
                if not (int(min_hold_days) > 0 and held < int(min_hold_days)):
                    gv = shares * ss
                    fe = gv * float(commission)
                    pnl_g = gv - cb_g
                    pnl_n = (gv - fe) - cb_n
                    cash_g += gv
                    cash_n += (gv - fe)
                    in_pos = False
                    shares = 0.0
                    cb_g = 0.0
                    cb_n = 0.0
                    cum += pnl_n
                    trades.append({
                        "Date": de,
                        "Typ": "Exit",
                        "Price": round(ss, 4),
                        "Shares": 0.0,
                        "Gross P&L": round(pnl_g, 2),
                        "Fees": round(fe, 2),
                        "Net P&L": round(pnl_n, 2),
                        "kum P&L": round(cum, 2),
                        "Prob": round(pp, 4),
                        "HoldDays": int(held)
                    })
                    last_xi = i
                    last_ei = None

        ct = float(df["Close"].iloc[i])
        eq_g.append(cash_g + (shares * ct if in_pos else 0.0))
        eq_n.append(cash_n + (shares * ct if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = eq_g
    df_bt["Equity_Net"] = eq_n
    return df_bt, trades


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────
def _cagr(v):
    if len(v) < 2:
        return np.nan
    dt0, dt1 = pd.to_datetime(v.index[0]), pd.to_datetime(v.index[-1])
    yrs = max((dt1 - dt0).days / 365.25, 1e-9)
    return (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1


def _sortino(rets):
    if rets.empty:
        return np.nan
    mean = rets.mean() * 252
    down = rets[rets < 0]
    dd = down.std() * np.sqrt(252) if len(down) else np.nan
    return mean / dd if dd and np.isfinite(dd) and dd > 0 else np.nan


def _winrate(trades):
    if not trades:
        return np.nan
    pnl = []
    e = None
    for ev in trades:
        if ev["Typ"] == "Entry":
            e = ev
        elif ev["Typ"] == "Exit" and e is not None:
            pnl.append(float(ev.get("Net P&L", 0.0)))
            e = None
    return float((np.array(pnl, float) > 0).mean()) if pnl else np.nan


def compute_performance(df_bt, trades, init_cap):
    net_ret = (df_bt["Equity_Net"].iloc[-1] / init_cap - 1) * 100
    rets = df_bt["Equity_Net"].pct_change().dropna()
    vol = rets.std(ddof=0) * sqrt(252) * 100
    sharpe = (rets.mean() * sqrt(252)) / (rets.std(ddof=0) + 1e-12)
    dd = (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax()) / df_bt["Equity_Net"].cummax()
    max_dd = dd.min() * 100
    cagr = _cagr(df_bt["Equity_Net"])
    calmar = cagr / abs(max_dd / 100) if np.isfinite(cagr) and max_dd < 0 else np.nan

    return {
        "Strategy Net (%)": round(float(net_ret), 2),
        "Strategy Gross (%)": round(float((df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100), 2),
        "Buy & Hold (%)": round(float((df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100), 2),
        "Volatility (%)": round(float(vol), 2),
        "Sharpe-Ratio": round(float(sharpe), 2),
        "Sortino-Ratio": round(float(_sortino(rets)), 2) if np.isfinite(_sortino(rets)) else np.nan,
        "Max Drawdown (%)": round(float(max_dd), 2),
        "Calmar-Ratio": round(float(calmar), 2) if np.isfinite(calmar) else np.nan,
        "Fees (€)": round(float(sum(t.get("Fees", 0.0) for t in trades)), 2),
        "Phase": "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat",
        "Closed Trades": int(sum(1 for t in trades if t["Typ"] == "Exit")),
        "Net P&L (€)": round(float(df_bt["Equity_Net"].iloc[-1] - init_cap), 2),
        "CAGR (%)": round(100 * float(cagr) if np.isfinite(cagr) else np.nan, 2),
        "Winrate (%)": round(100 * float(_winrate(trades)) if np.isfinite(_winrate(trades)) else np.nan, 2),
        "InitCap (€)": float(init_cap),
    }


def compute_round_trips(all_trades):
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        ce = None
        for ev in tr:
            if ev["Typ"] == "Entry":
                ce = ev
            elif ev["Typ"] == "Exit" and ce is not None:
                ed = pd.to_datetime(ce["Date"])
                xd = pd.to_datetime(ev["Date"])
                sh = float(ce.get("Shares", 0.0))
                ep = float(ce.get("Price", np.nan))
                xp = float(ev.get("Price", np.nan))
                fe = float(ce.get("Fees", 0.0))
                fx = float(ev.get("Fees", 0.0))
                pnl = float(ev.get("Net P&L", 0.0))
                cost = sh * ep + fe
                rows.append({
                    "Ticker": tk,
                    "Name": name,
                    "Entry Date": ed,
                    "Exit Date": xd,
                    "Hold (days)": (xd - ed).days,
                    "Entry Prob": ce.get("Prob", np.nan),
                    "Exit Prob": ev.get("Prob", np.nan),
                    "Shares": round(sh, 4),
                    "Entry Price": round(ep, 4),
                    "Exit Price": round(xp, 4),
                    "PnL Net (€)": round(pnl, 2),
                    "Fees (€)": round(fe + fx, 2),
                    "Return (%)": round(pnl / cost * 100, 2) if cost else np.nan
                })
                ce = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────
def estimate_expected_return(feat, forecast_days, threshold):
    if feat is None or feat.empty or "Open" not in feat.columns or "SignalProb" not in feat.columns:
        return {}
    fr = feat["Open"].shift(-(forecast_days + 1)) / feat["Open"].shift(-1) - 1
    tmp = pd.DataFrame({"FutureRet": fr}).dropna()
    if tmp.empty:
        return {}
    tmp["T"] = (tmp["FutureRet"] > float(threshold)).astype(int)
    mu1 = float(tmp.loc[tmp["T"] == 1, "FutureRet"].mean()) if tmp["T"].sum() > 0 else 0.0
    mu0 = float(tmp.loc[tmp["T"] == 0, "FutureRet"].mean()) if (tmp["T"] == 0).sum() > 0 else 0.0
    p = float(pd.to_numeric(feat["SignalProb"].iloc[-1], errors="coerce"))
    if not np.isfinite(p):
        p = 0.5
    return {"mu1": mu1, "mu0": mu0, "p": p, "exp_ret": p * mu1 + (1 - p) * mu0}


def _ensure_psd(cov, eps=1e-12):
    cov = (cov + cov.T) / 2
    try:
        w, v = np.linalg.eigh(cov)
        return (v * np.maximum(w, eps)) @ v.T
    except Exception:
        return np.diag(np.maximum(np.diag(cov), eps))


def portfolio_mc(exp_rets, cov, nav0, sims=1500, seed=42):
    tickers = exp_rets.index.tolist()
    cov = cov.reindex(index=tickers, columns=tickers).fillna(0.0)
    w = np.ones(len(tickers)) / max(len(tickers), 1)
    rng = np.random.default_rng(int(seed))
    draws = rng.multivariate_normal(exp_rets.values, _ensure_psd(cov.values), size=int(sims))
    pr = draws @ w
    nv = nav0 * (1 + pr)
    q = np.quantile(pr, [.05, .5, .95])
    qn = np.quantile(nv, [.05, .5, .95])
    return {
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
        "nq05": float(qn[0]),
        "nq50": float(qn[1]),
        "nq95": float(qn[2]),
        "port_rets": pr,
        "nav_paths": nv
    }


# ─────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────
def _build_estimator(model_params, calibrated=False):
    base = HistGradientBoostingClassifier(**model_params)
    if calibrated:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    return base


def _train_predict_proba(X_train, y_train, X_pred, model_params, calibrated=False):
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_pred), 0.5, dtype=float)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_pred_imp = imputer.transform(X_pred)

    est = _build_estimator(model_params, calibrated=calibrated)
    est.fit(X_train_imp, y_train)
    proba = est.predict_proba(X_pred_imp)[:, 1]
    return np.asarray(proba, dtype=float)


def make_features_and_train(
    df, lookback, horizon, threshold, model_params,
    entry_prob, exit_prob, init_capital, pos_frac,
    min_hold_days=0, cooldown_days=0,
    walk_forward=True, wf_min_train=180, wf_refit_step=5,
    calibration=True
):
    feat = make_features(df, lookback, horizon)
    hist = feat.iloc[:-1].copy()
    hist = hist.dropna(subset=["FutureRetExec"]).copy()

    if len(hist) < max(90, lookback + horizon + 30):
        raise ValueError("Zu wenige Datenpunkte.")

    X_cols = [c for c in BASE_FEATURES if c in hist.columns]
    hist["Target"] = (hist["FutureRetExec"] > threshold).astype(int)

    if len(X_cols) < 6:
        raise ValueError("Zu wenige Features nach Bereinigung.")
    if hist["Target"].nunique() < 2:
        feat["SignalProb"] = 0.5
    elif not walk_forward:
        probs = _train_predict_proba(
            hist[X_cols].values,
            hist["Target"].values,
            feat[X_cols].values,
            model_params,
            calibrated=calibration
        )
        feat["SignalProb"] = probs
    else:
        probs = np.full(len(feat), np.nan)
        min_train = max(int(wf_min_train), lookback + horizon + 40)

        last_fit_t = None
        imputer = None
        estimator = None

        for t in range(min_train, len(feat)):
            tr = feat.iloc[:t].dropna(subset=["FutureRetExec"]).copy()
            tr["Target"] = (tr["FutureRetExec"] > threshold).astype(int)

            if len(tr) < min_train:
                continue
            if tr["Target"].nunique() < 2:
                probs[t] = 0.5
                continue
            if tr["Target"].value_counts().min() < 10:
                probs[t] = 0.5
                continue

            should_refit = (estimator is None) or (last_fit_t is None) or ((t - last_fit_t) >= int(wf_refit_step))
            if should_refit:
                imputer = SimpleImputer(strategy="median")
                X_train_imp = imputer.fit_transform(tr[X_cols].values)
                estimator = _build_estimator(model_params, calibrated=calibration)
                estimator.fit(X_train_imp, tr["Target"].values)
                last_fit_t = t

            X_one = imputer.transform(feat[X_cols].iloc[[t]].values)
            probs[t] = estimator.predict_proba(X_one)[0, 1]

        feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)

    df_bt, trades = backtest_next_open(
        feat.iloc[:-1].copy(), entry_prob, exit_prob, COMMISSION,
        SLIPPAGE_BPS, init_capital, pos_frac,
        min_hold_days=int(min_hold_days), cooldown_days=int(cooldown_days)
    )

    return feat, df_bt, trades, compute_performance(df_bt, trades, init_capital)


# ─────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────
def chart_price_signal(feat, trades, ticker):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30], vertical_spacing=0.03
    )

    fig.add_trace(go.Candlestick(
        x=feat.index,
        open=feat["Open"],
        high=feat["High"],
        low=feat["Low"],
        close=feat["Close"],
        name="OHLC",
        showlegend=False,
        increasing=dict(line=dict(color=THEME["green"], width=1), fillcolor="rgba(22,163,74,0.18)"),
        decreasing=dict(line=dict(color=THEME["red"], width=1), fillcolor="rgba(220,38,38,0.16)"),
    ), row=1, col=1)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["Date"] = pd.to_datetime(tdf["Date"])
        for typ, sym, col in [("Entry", "triangle-up", THEME["green"]), ("Exit", "triangle-down", THEME["red"])]:
            sub = tdf[tdf["Typ"] == typ]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["Date"],
                    y=sub["Price"],
                    mode="markers",
                    name=typ,
                    marker=dict(symbol=sym, size=11, color=col, line=dict(color="white", width=1.5)),
                    hovertemplate=f"<b>{typ}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra></extra>",
                ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=feat.index,
        y=feat["SignalProb"],
        mode="lines",
        name="Signal Prob",
        line=dict(color=THEME["accent4"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>P=%{y:.4f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=ENTRY_PROB, row=2, col=1, line_color=THEME["green"], line_dash="dash", line_width=1, opacity=0.6)
    fig.add_hline(y=EXIT_PROB, row=2, col=1, line_color=THEME["red"], line_dash="dash", line_width=1, opacity=0.6)

    _apply_theme(fig, 500)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Preis & Signal", font=dict(size=13, color=THEME["muted"])),
        xaxis_rangeslider_visible=False,
        yaxis2=dict(range=[0, 1], title="P", tickformat=".2f"),
    )
    return fig


def chart_equity(df_bt, ticker, init_cap):
    eq = df_bt["Equity_Net"]
    bh = init_cap * df_bt["Close"] / df_bt["Close"].iloc[0]
    dd = (eq - eq.cummax()) / eq.cummax() * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04
    )

    fig.add_trace(go.Scatter(
        x=eq.index,
        y=eq,
        mode="lines",
        name="Strategie",
        line=dict(color=THEME["accent1"], width=2),
        fill="tozeroy",
        fillcolor="rgba(200,169,107,0.10)",
        hovertemplate="%{x|%Y-%m-%d}  %{y:,.2f}€<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bh.index,
        y=bh,
        mode="lines",
        name="Buy & Hold",
        line=dict(color=THEME["muted"], width=1.5, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}  B&H: %{y:,.2f}€<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd,
        mode="lines",
        name="Drawdown",
        line=dict(color=THEME["red"], width=1),
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.10)",
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}  DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    _apply_theme(fig, 450)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Equity & Drawdown", font=dict(size=13, color=THEME["muted"])),
        yaxis=dict(title="NAV (€)", tickformat=",.0f"),
        yaxis2=dict(title="DD %", tickformat=".1f"),
    )
    return fig


def chart_intraday(intra, ticker, tdf, chart_type, interval):
    fig = go.Figure()

    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(
            x=intra.index,
            open=intra["Open"],
            high=intra["High"],
            low=intra["Low"],
            close=intra["Close"],
            showlegend=False,
            increasing=dict(line=dict(color=THEME["green"], width=1), fillcolor="rgba(22,163,74,0.18)"),
            decreasing=dict(line=dict(color=THEME["red"], width=1), fillcolor="rgba(220,38,38,0.16)"),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=intra.index, y=intra["Close"], mode="lines",
            line=dict(color=THEME["accent1"], width=1.5), showlegend=False
        ))

    for _, ds in intra.groupby(intra.index.normalize()):
        fig.add_vline(
            x=ds.index.min(),
            line_width=1,
            line_dash="dot",
            line_color=THEME["border"],
            opacity=0.5
        )

    if not tdf.empty:
        tdf2 = tdf.copy()
        tdf2["Date"] = pd.to_datetime(tdf2["Date"])
        last_days = set(intra.index.normalize())
        ev = tdf2[tdf2["Date"].dt.normalize().isin(last_days)]
        for typ, col, sym in [("Entry", THEME["green"], "triangle-up"), ("Exit", THEME["red"], "triangle-down")]:
            xs, ys = [], []
            for d, ds in intra.groupby(intra.index.normalize()):
                h = ev[(ev["Typ"] == typ) & (ev["Date"].dt.normalize() == d)]
                if h.empty:
                    continue
                xs.append(ds.index.min())
                ys.append(float(h["Price"].iloc[-1]))
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers", name=typ,
                    marker=dict(symbol=sym, size=11, color=col, line=dict(color="white", width=1.5))
                ))

    _apply_theme(fig, 400)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  ·  Intraday 5d ({interval})", font=dict(size=13, color=THEME["muted"])),
        xaxis_rangeslider_visible=False,
    )
    return fig


def chart_corr(corr):
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        text=corr.round(2).astype(str).values,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#FCA5A5"],
            [0.5, "#F8FAFC"],
            [1.0, "#86EFAC"]
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="ρ", thickness=10),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>ρ = %{z:.3f}<extra></extra>",
        textfont=dict(size=10 if corr.shape[0] <= 8 else 8),
    ))
    n = corr.shape[0]
    _apply_theme(fig, max(380, n * 44))
    fig.update_layout(
        title=dict(text="Korrelationsmatrix", font=dict(size=13, color=THEME["muted"])),
        xaxis=dict(tickangle=-30, showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig


def chart_portfolio_nav(nav):
    fig = go.Figure(go.Scatter(
        x=nav.index,
        y=nav.values,
        mode="lines",
        line=dict(color=THEME["accent1"], width=2),
        fill="tozeroy",
        fillcolor="rgba(200,169,107,0.10)",
        hovertemplate="%{x|%Y-%m-%d}  %{y:,.0f}€<extra></extra>",
        name="NAV",
    ))
    _apply_theme(fig, 360)
    fig.update_layout(
        title=dict(text="Portfolio NAV · Equal-Weight", font=dict(size=13, color=THEME["muted"])),
        yaxis=dict(title="NAV (€)", tickformat=",.0f"),
        showlegend=False,
    )
    return fig


def chart_mc_histogram(port_rets, q05, q50, q95, forecast_days, sims):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_rets * 100,
        nbinsx=50,
        marker=dict(
            color=["rgba(22,163,74,0.55)" if v >= 0 else "rgba(220,38,38,0.45)" for v in port_rets],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        showlegend=False,
        hovertemplate="Return: %{x:.2f}%<br>Anzahl: %{y}<extra></extra>",
    ))

    for val, lbl, col in [
        (q05 * 100, "5%", THEME["red"]),
        (q50 * 100, "50%", THEME["accent3"]),
        (q95 * 100, "95%", THEME["green"])
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=col,
            line_width=1.5,
            opacity=0.8,
            annotation_text=f"  {lbl}: {val:.2f}%",
            annotation_font=dict(color=col, size=9),
            annotation_position="top right"
        )

    _apply_theme(fig, 360)
    fig.update_layout(
        title=dict(text=f"MC Portfolio Returns  ·  {forecast_days}d  ·  {sims} Sim.", font=dict(size=13, color=THEME["muted"])),
        xaxis_title="Return (%)",
        yaxis_title="Häufigkeit",
        bargap=0.04,
    )
    return fig


def chart_histogram(data, xlabel, title, bins):
    mean_v = float(data.mean()) if len(data) else np.nan
    fig = go.Figure(go.Histogram(
        x=data,
        nbinsx=bins,
        marker=dict(
            color=["rgba(22,163,74,0.55)" if v >= 0 else "rgba(220,38,38,0.45)" for v in data],
            line=dict(color="rgba(0,0,0,0)", width=0)
        ),
        showlegend=False,
        hovertemplate=f"{xlabel}: %{{x:.2f}}<br>n: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=THEME["muted"], line_width=1, opacity=0.4)
    if np.isfinite(mean_v):
        fig.add_vline(
            x=mean_v,
            line_dash="dot",
            line_color=THEME["accent2"],
            line_width=1.5,
            annotation_text=f"  Ø {mean_v:.2f}",
            annotation_font=dict(color=THEME["accent2"], size=9),
            annotation_position="top right"
        )
    _apply_theme(fig, 340)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=THEME["muted"])),
        xaxis_title=xlabel,
        yaxis_title="n",
        bargap=0.05
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────
section("PARAMETER-OPTIMIERUNG")
with st.expander("⚙ Random-Search Optimizer  ·  Robuster OOS-Score", expanded=False):
    oc1, oc2 = st.columns(2)
    with oc1:
        n_trials = st.number_input("Trials", 10, 1000, 80, step=10)
        seed_opt = st.number_input("Seed", 0, 10000, 42)
        min_trades_req = st.number_input("Min. Closed Trades", 0, 10000, 5, step=1)
        min_valid_tickers = st.number_input("Min. gültige Ticker", 1, 200, 2, step=1)
    with oc2:
        lb_lo, lb_hi = st.slider("Lookback", 10, 252, (20, 90), step=5)
        hz_lo, hz_hi = st.slider("Horizon", 1, 20, (3, 8))
        thr_lo, thr_hi = st.slider("Threshold", 0.0, 0.15, (0.015, 0.060), step=0.005, format="%.3f")
        en_lo, en_hi = st.slider("Entry Prob", 0.0, 1.0, (0.55, 0.80), step=0.01)
        ex_lo, ex_hi = st.slider("Exit Prob", 0.0, 1.0, (0.30, 0.55), step=0.01)

    @st.cache_data(show_spinner=False)
    def _opt_prices(tickers, start, end, use_tail, interval, fb, ek, moc):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fb, ek, moc)[0]

    def _oos_score(df_bt_o: pd.DataFrame):
        if df_bt_o.empty or "Equity_Net" not in df_bt_o.columns:
            return np.nan, np.nan, np.nan
        r = df_bt_o["Equity_Net"].pct_change().dropna()
        if len(r) < 20:
            return np.nan, np.nan, np.nan
        vol = r.std(ddof=0)
        sharpe = (r.mean() / (vol + 1e-12)) * np.sqrt(252)
        dd = ((df_bt_o["Equity_Net"] / df_bt_o["Equity_Net"].cummax()) - 1).min()
        end_ret = df_bt_o["Equity_Net"].iloc[-1] / df_bt_o["Equity_Net"].iloc[0] - 1
        return float(sharpe), float(dd), float(end_ret)

    if st.button("🚀 Suche starten", type="primary", use_container_width=True):
        import random
        from collections import Counter

        rng_opt = random.Random(int(seed_opt))
        pm = _opt_prices(
            tuple(TICKERS), str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
        )

        feasible = {
            tk: df.copy() for tk, df in (pm or {}).items()
            if isinstance(df, pd.DataFrame) and len(df) >= max(int(wf_min_train) + 40, 140)
        }

        if len(feasible) < int(min_valid_tickers):
            st.error("Zu wenige Ticker nach Prefilter.")
            st.stop()

        rows_o, best_o = [], None
        prog_o = st.progress(0.0)
        status_o = st.empty()
        err_c = Counter()

        for trial in range(int(n_trials)):
            p = dict(
                lookback=rng_opt.randrange(lb_lo, lb_hi + 1, 5),
                horizon=rng_opt.randrange(hz_lo, hz_hi + 1),
                thresh=rng_opt.uniform(thr_lo, thr_hi),
                entry=rng_opt.uniform(en_lo, en_hi),
                exit=rng_opt.uniform(ex_lo, ex_hi)
            )
            if p["exit"] >= p["entry"]:
                prog_o.progress((trial + 1) / int(n_trials))
                continue

            sharps_o, dds_o, rets_o, trad_o, ok_t = [], [], [], 0, 0

            for tk, df0 in feasible.items():
                if len(df0) < max(130, p["lookback"] + p["horizon"] + 50):
                    continue
                try:
                    feat_o, df_bt_o, tr_o, _ = make_features_and_train(
                        df=df0,
                        lookback=int(p["lookback"]),
                        horizon=int(p["horizon"]),
                        threshold=float(p["thresh"]),
                        model_params=MODEL_PARAMS,
                        entry_prob=float(p["entry"]),
                        exit_prob=float(p["exit"]),
                        init_capital=float(INIT_CAP_PER_TICKER),
                        pos_frac=float(POS_FRAC),
                        min_hold_days=int(MIN_HOLD_DAYS),
                        cooldown_days=int(COOLDOWN_DAYS),
                        walk_forward=True,
                        wf_min_train=int(wf_min_train),
                        wf_refit_step=int(wf_refit_step),
                        calibration=bool(calibration_on)
                    )

                    oos_start_idx = max(int(wf_min_train), int(0.55 * len(df_bt_o)))
                    oos_slice = df_bt_o.iloc[oos_start_idx:].copy()
                    sh, dd, rr = _oos_score(oos_slice)
                    closed_trades = int(sum(1 for t in tr_o if t["Typ"] == "Exit"))

                    if not np.isfinite(sh) or closed_trades < int(min_trades_req):
                        raise ValueError("schwacher_oos_block")

                    sharps_o.append(sh)
                    dds_o.append(dd)
                    rets_o.append(rr)
                    trad_o += closed_trades
                    ok_t += 1
                except Exception as e:
                    err_c[str(e)[:80]] += 1

            if ok_t < int(min_valid_tickers) or not sharps_o:
                prog_o.progress((trial + 1) / int(n_trials))
                continue

            sharpe_med = float(np.nanmedian(sharps_o))
            ret_med = float(np.nanmedian(rets_o)) if rets_o else np.nan
            dd_med = float(np.nanmedian(dds_o)) if dds_o else np.nan

            # Robuster Score ohne Overtrading-Penalty:
            # priorisiert OOS-Sharpe, dann OOS-Return, dann Drawdown-Stabilität
            score = sharpe_med + 0.35 * ret_med + 0.15 * dd_med

            rec = dict(
                trial=trial,
                score=score,
                sharpe_med=sharpe_med,
                ret_med=ret_med,
                dd_med=dd_med,
                trades=trad_o,
                ok_tickers=ok_t,
                **p
            )
            rows_o.append(rec)

            if best_o is None or score > best_o["score"]:
                best_o = rec

            if (trial + 1) % 10 == 0 and best_o is not None:
                status_o.caption(
                    f"Trial {trial+1}/{n_trials} · Best Score: {best_o['score']:.3f} · "
                    f"Sharpe: {best_o['sharpe_med']:.2f}"
                )

            prog_o.progress((trial + 1) / int(n_trials))

        if not rows_o:
            st.error("Keine gültigen Ergebnisse.")
            if err_c:
                st.dataframe(pd.DataFrame(err_c.most_common(10), columns=["Error", "Count"]))
        else:
            df_res_o = pd.DataFrame(rows_o).sort_values("score", ascending=False)
            st.success(
                f"✅ Best Score: **{best_o['score']:.3f}** · "
                f"Sharpe: **{best_o['sharpe_med']:.2f}** · "
                f"Median Return: **{best_o['ret_med']*100:.2f}%**"
            )
            bc = st.columns(5)
            bc[0].metric("Lookback", int(best_o["lookback"]))
            bc[1].metric("Horizon", int(best_o["horizon"]))
            bc[2].metric("Threshold", f"{best_o['thresh']:.3f}")
            bc[3].metric("Entry Prob", f"{best_o['entry']:.2f}")
            bc[4].metric("Exit Prob", f"{best_o['exit']:.2f}")
            st.dataframe(df_res_o.head(25), use_container_width=True)
            st.download_button(
                "⬇ Optimizer-Ergebnisse",
                to_csv_eu(df_res_o),
                file_name="optimizer_results.csv",
                mime="text/csv"
            )


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:6px 0 18px;border-bottom:1px solid #E5E7EB;margin-bottom:30px;">
  <div class="nexus-header">NEXUS</div>
  <div class="nexus-sub">2nd AI Model · HistGradientBoosting · Walk-Forward OOS · Portfolio MC Forecast</div>
</div>
""", unsafe_allow_html=True)

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs: Dict[str, pd.DataFrame] = {}
all_feat: Dict[str, pd.DataFrame] = {}
all_strategy_rets: Dict[str, pd.Series] = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
)

live_forecasts_run: List[dict] = []
_decide = lambda p, en, ex: "Enter / Add" if p > en else ("Exit / Reduce" if p < ex else "Hold / No Trade")


# ─────────────────────────────────────────────────────────────
# Per-Ticker Loop
# ─────────────────────────────────────────────────────────────
for ticker in TICKERS:
    if ticker not in price_map:
        continue

    df = price_map[ticker]
    meta = meta_map.get(ticker, {})
    name = get_ticker_name(ticker)

    with st.expander(f"⚡  {ticker}  ·  {name}", expanded=False):
        try:
            ts = df.index[-1]
            sfx = f" · intraday {meta['tail_ts'].strftime('%H:%M')}" if meta.get("tail_is_intraday") and meta.get("tail_ts") is not None else ""
            st.caption(
                f"🕐 {ts.strftime('%Y-%m-%d %H:%M %Z')}{sfx}  ·  "
                f"{'WF-OOS' if use_walk_forward else 'In-Sample'}  ·  "
                f"Target: FutureRetExec > {THRESH:.3f} in {HORIZON}d"
            )

            feat, df_bt, trades, metrics = make_features_and_train(
                df=df,
                lookback=int(LOOKBACK),
                horizon=int(HORIZON),
                threshold=float(THRESH),
                model_params=MODEL_PARAMS,
                entry_prob=float(ENTRY_PROB),
                exit_prob=float(EXIT_PROB),
                init_capital=float(INIT_CAP_PER_TICKER),
                pos_frac=float(POS_FRAC),
                min_hold_days=int(MIN_HOLD_DAYS),
                cooldown_days=int(COOLDOWN_DAYS),
                walk_forward=bool(use_walk_forward),
                wf_min_train=int(wf_min_train),
                wf_refit_step=int(wf_refit_step),
                calibration=bool(calibration_on)
            )

            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat
            all_strategy_rets[ticker] = df_bt["Equity_Net"].pct_change()

            live_prob = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            live_act = _decide(live_prob, float(ENTRY_PROB), float(EXIT_PROB))

            live_forecasts_run.append({
                "AsOf": pd.Timestamp(feat.index[-1]).strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "Name": name,
                f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob, 4),
                "Action": live_act,
                "Close": round(live_close, 4),
                "Bar": "intraday" if meta.get("tail_is_intraday") else "daily"
            })

            mn = metrics
            phase_cls = "kpi-info" if mn["Phase"] == "Open" else ""
            kpi_row([
                ("NETTO", _pct(mn["Strategy Net (%)"]), "kpi-pos" if mn["Strategy Net (%)"] > 0 else "kpi-neg"),
                ("BUY & HOLD", _pct(mn["Buy & Hold (%)"]), "kpi-pos" if mn["Buy & Hold (%)"] > 0 else "kpi-neg"),
                ("SHARPE", f"{mn['Sharpe-Ratio']:.2f}", "kpi-info"),
                ("SORTINO", f"{mn['Sortino-Ratio']:.2f}" if np.isfinite(mn["Sortino-Ratio"]) else "–", "kpi-info"),
                ("MAX DD", _pct(mn["Max Drawdown (%)"]), "kpi-neg"),
                ("WINRATE", f"{mn['Winrate (%)']:.1f}%" if np.isfinite(mn["Winrate (%)"]) else "–", ""),
                ("TRADES", f"{mn['Closed Trades']}", ""),
                ("PHASE", mn["Phase"], phase_cls),
            ])

            cc1, cc2 = st.columns(2)
            tdf_loc = pd.DataFrame(trades)

            with cc1:
                st.plotly_chart(
                    chart_price_signal(feat, trades, ticker),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

            intra = get_intraday_last_n_sessions(ticker, 5, 10, intraday_interval)
            with cc2:
                if intra.empty:
                    st.info("Keine Intraday-Daten.")
                else:
                    st.plotly_chart(
                        chart_intraday(intra, ticker, tdf_loc, intraday_chart_type, intraday_interval),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )

            st.plotly_chart(
                chart_equity(df_bt, ticker, float(INIT_CAP_PER_TICKER)),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            with st.expander(f"🗒 Trade-Log  ·  {ticker}", expanded=False):
                if tdf_loc.empty:
                    st.info("Keine Trades.")
                else:
                    td = tdf_loc.copy()
                    td["Ticker"] = ticker
                    td["Name"] = name
                    if "Date" in td.columns:
                        td["Date"] = pd.to_datetime(td["Date"]).dt.strftime("%d.%m.%Y")
                    td = td.rename(columns={
                        "Prob": "Signal Prob",
                        "HoldDays": "Hold (days)",
                        "Net P&L": "PnL",
                        "kum P&L": "CumPnL"
                    })
                    desired = ["Ticker", "Name", "Date", "Typ", "Price", "Shares", "Signal Prob", "Hold (days)", "PnL", "CumPnL", "Fees"]
                    sc = [c for c in desired if c in td.columns]

                    def _rc(row):
                        t = str(row.get("Typ", "")).lower()
                        if "entry" in t:
                            return [f"color: {THEME['green']}"] * len(row)
                        if "exit" in t:
                            return [f"color: {THEME['red']}"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        td[sc].style.format({
                            "Price": "{:.2f}",
                            "Shares": "{:.4f}",
                            "Signal Prob": "{:.4f}",
                            "PnL": "{:.2f}",
                            "CumPnL": "{:.2f}",
                            "Fees": "{:.2f}"
                        }).apply(_rc, axis=1),
                        use_container_width=True
                    )
                    st.download_button(
                        f"⬇ Trades {ticker}",
                        to_csv_eu(td[sc]),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key=f"dl_tr_{ticker}"
                    )

        except Exception as e:
            st.error(f"❌ {ticker}: {e}")
            st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────────────────────
# Live Forecast Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
        .drop_duplicates(subset=["Ticker"], keep="last")
        .sort_values(["AsOf", "Ticker"])
        .reset_index(drop=True)
    )
    live_df["Target_px"] = (pd.to_numeric(live_df["Close"], errors="coerce") * (1 + float(THRESH))).round(2)
    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"

    section(f"LIVE FORECAST BOARD  ·  {HORIZON}d")

    n_enter = (live_df["Action"] == "Enter / Add").sum()
    n_exit = (live_df["Action"] == "Exit / Reduce").sum()
    n_hold = len(live_df) - n_enter - n_exit

    kpi_row([
        ("Signale gesamt", f"{len(live_df)}", ""),
        ("▲ Enter", f"{n_enter}", "kpi-pos"),
        ("▼ Exit", f"{n_exit}", "kpi-neg"),
        ("◆ Hold", f"{n_hold}", ""),
    ])

    desired_lf = ["AsOf", "Ticker", "Name", prob_col, "Action", "Close", "Target_px", "Bar"]
    show_lf = [c for c in desired_lf if c in live_df.columns]

    def _board_color(row):
        a = str(row.get("Action", "")).lower()
        if "enter" in a:
            return [f"background: rgba(22,163,74,0.06)"] * len(row)
        if "exit" in a:
            return [f"background: rgba(220,38,38,0.06)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        live_df[show_lf].style.format({prob_col: "{:.4f}", "Close": "{:.2f}", "Target_px": "{:.2f}"}).apply(_board_color, axis=1),
        use_container_width=True,
        height=min(600, 40 + 35 * len(live_df)),
    )

    st.download_button(
        "⬇ Forecasts CSV",
        to_csv_eu(live_df),
        file_name=f"live_forecasts_{HORIZON}d.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────
# Summary & Portfolio
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / float(INIT_CAP_PER_TICKER)) * 100

    if "Phase" in summary_df.columns:
        summary_df["Phase"] = (
            summary_df["Phase"].astype(str).str.strip().str.lower()
            .map(lambda x: "Open" if x == "open" else ("Flat" if x == "flat" else x.capitalize()))
        )

    total_net = float(summary_df["Net P&L (€)"].sum())
    total_fees = float(summary_df["Fees (€)"].sum())
    total_gross = total_net + total_fees

    section("PORTFOLIO SUMMARY")
    kpi_row([
        ("Netto P&L", _eur(total_net), "kpi-pos" if total_net > 0 else "kpi-neg"),
        ("Brutto P&L", _eur(total_gross), "kpi-pos" if total_gross > 0 else "kpi-neg"),
        ("Trading Costs", f"–{total_fees:,.2f}€", "kpi-neg"),
        ("Trades", f"{int(summary_df['Closed Trades'].sum())}", ""),
        ("Ø CAGR", f"{summary_df['CAGR (%)'].dropna().mean():.2f}%" if "CAGR (%)" in summary_df else "–", "kpi-info"),
        ("Ø Winrate", f"{summary_df['Winrate (%)'].dropna().mean():.1f}%" if "Winrate (%)" in summary_df else "–", ""),
    ])

    def _num_col(v):
        try:
            return f"color: {THEME['green']}" if float(v) > 0 else (f"color: {THEME['red']}" if float(v) < 0 else "")
        except Exception:
            return ""

    def _phase_col(v):
        if str(v) == "Open":
            return f"background: rgba(37,99,235,0.10); color: {THEME['accent3']}; border-radius:8px; padding:1px 6px;"
        if str(v) == "Flat":
            return f"background: rgba(107,114,128,0.08); color:{THEME['muted']}; border-radius:8px; padding:1px 6px;"
        return ""

    pct_c = ["Strategy Net (%)", "Strategy Gross (%)", "Buy & Hold (%)", "Net P&L (%)", "CAGR (%)"]
    styled_s = (
        summary_df.style.format({
            "Strategy Net (%)": "{:.2f}",
            "Strategy Gross (%)": "{:.2f}",
            "Buy & Hold (%)": "{:.2f}",
            "Volatility (%)": "{:.2f}",
            "Sharpe-Ratio": "{:.2f}",
            "Sortino-Ratio": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}",
            "Calmar-Ratio": "{:.2f}",
            "Fees (€)": "{:.2f}",
            "Net P&L (%)": "{:.2f}",
            "Net P&L (€)": "{:.2f}",
            "CAGR (%)": "{:.2f}",
            "Winrate (%)": "{:.2f}",
            "InitCap (€)": "{:.0f}"
        })
        .applymap(_num_col, subset=[c for c in pct_c if c in summary_df.columns])
    )
    if "Phase" in summary_df.columns:
        styled_s = styled_s.applymap(_phase_col, subset=["Phase"])

    st.dataframe(styled_s, use_container_width=True)
    st.download_button(
        "⬇ Summary CSV",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary.csv",
        mime="text/csv"
    )

    section("OFFENE POSITIONEN")
    open_positions = []
    for ticker, tr in all_trades.items():
        if tr and tr[-1]["Typ"] == "Entry":
            le = next(t for t in reversed(tr) if t["Typ"] == "Entry")
            lc = float(all_dfs[ticker]["Close"].iloc[-1])
            upnl = (lc - float(le["Price"])) * float(le["Shares"])
            open_positions.append({
                "Ticker": ticker,
                "Name": get_ticker_name(ticker),
                "Entry Date": pd.to_datetime(le["Date"]).strftime("%Y-%m-%d"),
                "Entry Price": round(float(le["Price"]), 2),
                "Close": round(lc, 2),
                "Signal Prob": round(float(all_feat[ticker]["SignalProb"].iloc[-1]), 4),
                "uPnL (€)": round(upnl, 2),
            })

    if open_positions:
        op_df = pd.DataFrame(open_positions).sort_values("Entry Date", ascending=False)

        def _upnl_c(v):
            try:
                return f"color:{THEME['green']}" if float(v) >= 0 else f"color:{THEME['red']}"
            except Exception:
                return ""

        st.dataframe(
            op_df.style.format({
                "Entry Price": "{:.2f}",
                "Close": "{:.2f}",
                "Signal Prob": "{:.4f}",
                "uPnL (€)": "{:.2f}"
            }).applymap(_upnl_c, subset=["uPnL (€)"]),
            use_container_width=True
        )
        st.download_button(
            "⬇ Open Positions",
            to_csv_eu(op_df),
            file_name="open_positions.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Keine offenen Positionen.")

    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        section("ROUND-TRIPS")
        rt_df["Entry Date"] = pd.to_datetime(rt_df["Entry Date"])
        rt_df["Exit Date"] = pd.to_datetime(rt_df["Exit Date"])

        ret = pd.to_numeric(rt_df.get("Return (%)"), errors="coerce").dropna()
        pnl = pd.to_numeric(rt_df.get("PnL Net (€)"), errors="coerce").dropna()

        kpi_row([
            ("Trades", f"{len(ret)}", ""),
            ("Winrate", f"{100 * (ret > 0).mean():.1f}%" if len(ret) else "–", "kpi-pos"),
            ("Ø Return", f"{ret.mean():.2f}%" if len(ret) else "–", "kpi-info"),
            ("Median", f"{ret.median():.2f}%" if len(ret) else "–", ""),
            ("Std", f"{ret.std():.2f}%" if len(ret) else "–", "kpi-warn"),
        ])

        rt_disp = rt_df.copy()
        rt_disp["Entry Date"] = rt_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        rt_disp["Exit Date"] = rt_disp["Exit Date"].dt.strftime("%Y-%m-%d")
        if "Hold (days)" in rt_disp.columns:
            rt_disp["Hold (days)"] = rt_disp["Hold (days)"].round().astype("Int64")

        def _ret_c(v):
            try:
                return f"color:{THEME['green']};font-weight:500" if float(v) >= 0 else f"color:{THEME['red']};font-weight:500"
            except Exception:
                return ""

        st.dataframe(
            rt_disp.sort_values("Exit Date", ascending=False).style.format({
                "Shares": "{:.4f}",
                "Entry Price": "{:.2f}",
                "Exit Price": "{:.2f}",
                "PnL Net (€)": "{:.2f}",
                "Fees (€)": "{:.2f}",
                "Return (%)": "{:.2f}",
                "Entry Prob": "{:.4f}",
                "Exit Prob": "{:.4f}"
            }).applymap(_ret_c, subset=["Return (%)", "PnL Net (€)"]),
            use_container_width=True
        )

        st.download_button(
            "⬇ Round-Trips CSV",
            to_csv_eu(rt_disp),
            file_name="round_trips.csv",
            mime="text/csv"
        )

        section("RETURN-VERTEILUNG")
        bins_rt = st.slider("Bins", 10, 120, 30, step=5, key="rt_bins")
        hc1, hc2 = st.columns(2)
        with hc1:
            if not ret.empty:
                st.plotly_chart(
                    chart_histogram(ret, "Return (%)", "Trade Returns (%)", bins_rt),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
        with hc2:
            if not pnl.empty:
                st.plotly_chart(
                    chart_histogram(pnl, "PnL Net (€)", "Trade P&L (€)", bins_rt),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

    section("PORTFOLIO  ·  EQUAL-WEIGHT STRATEGY RETURNS")
    strat_series = []
    for tk, s in all_strategy_rets.items():
        if isinstance(s, pd.Series) and len(s.dropna()) >= 10:
            ss = s.copy()
            ss.name = tk
            try:
                if getattr(ss.index, "tz", None) is not None:
                    ss.index = ss.index.tz_localize(None)
            except Exception:
                pass
            ss.index = pd.to_datetime(ss.index).normalize()
            strat_series.append(ss)

    strat_port = pd.DataFrame()
    port_ret = pd.Series(dtype=float)

    if len(strat_series) < 2:
        st.info("Mindestens 2 Ticker nötig.")
    else:
        strat_port = pd.concat(strat_series, axis=1, join="outer").sort_index()
        valid = strat_port.notna().sum(axis=1) >= 2
        rets2 = strat_port.loc[valid].copy()

        if not rets2.empty:
            w_row = rets2.notna().astype(float)
            w_row = w_row.div(w_row.sum(axis=1), axis=0)
            port_ret = (rets2.fillna(0.0) * w_row).sum(axis=1).dropna()

        if port_ret.empty:
            st.info("Portfolio-Returns leer.")
        else:
            ann_r = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
            ann_v = port_ret.std(ddof=0) * np.sqrt(252)
            sh_p = (port_ret.mean() / (port_ret.std(ddof=0) + 1e-12)) * np.sqrt(252)
            nav0 = float(INIT_CAP_PER_TICKER) * len(summary_df)
            nav = nav0 * (1 + port_ret).cumprod()
            max_dd_p = float(((nav / nav.cummax()) - 1).min())

            kpi_row([
                ("Return p.a.", f"{ann_r*100:.2f}%", "kpi-pos" if ann_r > 0 else "kpi-neg"),
                ("Vol p.a.", f"{ann_v*100:.2f}%", "kpi-warn"),
                ("Sharpe", f"{sh_p:.2f}", "kpi-info"),
                ("Max DD", f"{max_dd_p*100:.2f}%", "kpi-neg"),
            ])

            st.plotly_chart(
                chart_portfolio_nav(nav),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.download_button(
                "⬇ Portfolio Returns CSV",
                to_csv_eu(pd.DataFrame({"Date": port_ret.index, "PortRet": port_ret.values})),
                file_name="portfolio_returns.csv",
                mime="text/csv"
            )

    if not strat_port.empty and strat_port.shape[1] >= 2:
        section("KORRELATION  ·  STRATEGY RETURNS")
        ka, kb, kc, kd = st.columns(4)
        freq_lbl = ka.selectbox("Frequenz", ["täglich", "wöchentlich", "monatlich"], index=0)
        corr_meth = kb.selectbox("Methode", ["Pearson", "Spearman", "Kendall"], index=0)
        min_obs_c = kc.slider("Min. Punkte", 10, 300, 20, step=5)
        use_ffill_c = kd.checkbox("FFill Lücken", value=False)

        pc = strat_port.copy()
        if use_ffill_c:
            pc = pc.ffill()
        if freq_lbl == "wöchentlich":
            pc = (1 + pc).resample("W").prod() - 1
        elif freq_lbl == "monatlich":
            pc = (1 + pc).resample("M").prod() - 1

        rc = pc.dropna(how="all")
        keep_cols = rc.notna().sum()[rc.notna().sum() >= int(min_obs_c)].index
        rc = rc[keep_cols]

        if rc.shape[1] >= 2:
            corr = rc.corr(method=corr_meth.lower())
            st.plotly_chart(chart_corr(corr), use_container_width=True, config={"displayModeBar": False})

            m = corr.values.copy()
            np.fill_diagonal(m, np.nan)
            off = m[np.isfinite(m)]
            cov_c = rc.cov()
            vols_c = np.sqrt(np.diag(cov_c.values))
            n_c = len(vols_c)
            w_c = np.ones(n_c) / n_c
            pv = float(w_c @ cov_c.values @ w_c)
            denom = float(np.sum((w_c[:, None] * w_c[None, :] * vols_c[:, None] * vols_c[None, :]))) + 1e-12
            diag_p = float(np.sum((w_c ** 2) * (vols_c ** 2)))
            ipc = (pv - diag_p) / denom

            kpi_row([
                ("Ø Paar-ρ", f"{np.mean(off):.2f}" if off.size else "–", ""),
                ("Median-ρ", f"{np.median(off):.2f}" if off.size else "–", ""),
                ("σ(ρ)", f"{np.std(off):.2f}" if off.size else "–", ""),
                ("IPC (norm.)", f"{ipc:.2f}" if np.isfinite(ipc) else "–", "kpi-info"),
            ])

            st.caption(f"Basis: {len(rc)} Zeitpunkte · {freq_lbl} · {corr_meth}")
            st.download_button(
                "⬇ Korrelationsmatrix CSV",
                to_csv_eu(corr.reset_index().rename(columns={"index": "Ticker"})),
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )
        else:
            st.info("Zu wenig Overlap für Korrelationsmatrix.")

    section(f"PORTFOLIO FORECAST  ·  {int(FORECAST_DAYS)}d  ·  MC={int(MC_SIMS)}")

    rows_fc = []
    for tk, feat in all_feat.items():
        est = estimate_expected_return(feat, int(FORECAST_DAYS), float(THRESH))
        if not est:
            continue
        rows_fc.append({
            "Ticker": tk,
            "Name": get_ticker_name(tk),
            "p (Prob)": est["p"],
            "μ1": est["mu1"],
            "μ0": est["mu0"],
            f"E[r {int(FORECAST_DAYS)}d]": est["exp_ret"]
        })

    if not rows_fc:
        st.info("Forecast: Nicht genug Daten.")
    else:
        fc_df = pd.DataFrame(rows_fc).set_index("Ticker")
        ercol = f"E[r {int(FORECAST_DAYS)}d]"
        st.dataframe(
            fc_df.sort_values(ercol, ascending=False).style.format({
                "p (Prob)": "{:.4f}",
                "μ1": "{:.4f}",
                "μ0": "{:.4f}",
                ercol: "{:.4f}"
            }).applymap(_num_col, subset=[ercol]),
            use_container_width=True
        )

        exp_rets = fc_df[ercol].astype(float).dropna()

        if strat_port.empty:
            st.info("Für Portfolio-MC mind. 2 Ticker nötig.")
        else:
            exp_rets = exp_rets.reindex(strat_port.columns.intersection(exp_rets.index)).dropna()

            if len(exp_rets) < 2:
                st.info("Für Portfolio-MC mind. 2 Ticker nötig.")
            else:
                dr = strat_port[exp_rets.index].dropna(how="all")
                cov_h = dr.cov(min_periods=60) * float(FORECAST_DAYS)
                nav0_fc = float(INIT_CAP_PER_TICKER) * len(summary_df)
                out_mc = portfolio_mc(exp_rets, cov_h, nav0_fc, sims=int(MC_SIMS), seed=42)

                kpi_row([
                    ("E[Return EW]", f"{exp_rets.mean()*100:.2f}%", "kpi-info"),
                    ("5%-Quantil", f"{out_mc['q05']*100:.2f}%  ·  {out_mc['nq05']:,.0f}€", "kpi-neg"),
                    ("Median", f"{out_mc['q50']*100:.2f}%  ·  {out_mc['nq50']:,.0f}€", ""),
                    ("95%-Quantil", f"{out_mc['q95']*100:.2f}%  ·  {out_mc['nq95']:,.0f}€", "kpi-pos"),
                ])

                st.plotly_chart(
                    chart_mc_histogram(
                        out_mc["port_rets"], out_mc["q05"], out_mc["q50"],
                        out_mc["q95"], int(FORECAST_DAYS), int(MC_SIMS)
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
else:
    st.warning("⚠ Keine Ergebnisse — Ticker & Datenabdeckung prüfen.")


# Footer
st.markdown(f"""
<div style="
    margin-top:56px;
    padding:22px 0;
    border-top:1px solid {THEME['border']};
    text-align:center;
    color:{THEME['muted']};
    font-family:'Inter',sans-serif;
    font-size:11px;
    font-weight:600;
    letter-spacing:0.06em;
    text-transform:uppercase;">
  NEXUS 2ND AI MODEL v3.1 · HistGradientBoosting · Walk-Forward OOS · No Options Input
</div>
""", unsafe_allow_html=True)
'''

path = Path("/mnt/data/streamlit_app_v3_1.py")
path.write_text(code, encoding="utf-8")
print(path)
