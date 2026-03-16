# streamlit_app_v3_1.py
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

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
  border: 1px solid #E5E7EB;
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 6px 22px rgba(17, 24, 39, 0.06);
}
.kpi-label {
  font-size: 11px;
  color: #6B7280;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  margin-bottom: 8px;
}
.kpi-value {
  font-size: 22px;
  font-weight: 750;
  color: #111827;
}
.kpi-pos .kpi-value { color: #16A34A; }
.kpi-neg .kpi-value { color: #DC2626; }
.kpi-info .kpi-value { color: #2563EB; }
.kpi-warn .kpi-value { color: #B28A45; }
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
    return fig

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
    INIT_CAP_PER_TICKER = st.number_input("Capital/Ticker (€)", 1000.0, 1_000_000.0, 10_000.0, step=1000.0, format="%.0f")

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

# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
def get_price_data_tail_intraday(ticker, start_date, end_date, use_tail=True, interval="5m", fallback_last_session=False, exec_mode_key="Next Open (backtest+live)", moc_cutoff_min_val=15):
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
                ex.submit(get_price_data_tail_intraday, tk, start, end, use_tail, interval, fallback_last, exec_key, int(moc_cutoff)): tk for tk in tickers
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
# Features / Model
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


BASE_FEATURES = [
    "Range_pct", "SlopeHigh", "SlopeLow", "SlopeClose", "ret_1d", "ret_5d", "ret_20d",
    "vol_20d", "vol_60d", "ATR_pct", "dist_ma20", "dist_ma50", "dist_high20", "dist_low20",
    "gap_open", "oc_ret", "hl_spread", "vol_rel20", "vol_chg_5d"
]


def make_features(df: pd.DataFrame, lookback: int, horizon: int):
    if len(df) < max(lookback + horizon + 25, 120):
        raise ValueError("Zu wenige Bars.")
    feat = add_price_features(df.copy(), lookback)
    feat["EntryOpen"] = feat["Open"].shift(-1)
    feat["ExitOpen"] = feat["Open"].shift(-(horizon + 1))
    feat["FutureRetExec"] = feat["ExitOpen"] / feat["EntryOpen"] - 1
    return feat


def _build_estimator(model_params, calibrated=False):
    base = HistGradientBoostingClassifier(**model_params)
    if calibrated:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    return base


def backtest_next_open(df, entry_thr, exit_thr, commission, slippage_bps, init_cap, pos_frac, min_hold_days=0, cooldown_days=0):
    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte.")

    cash_g, cash_n = init_cap, init_cap
    shares, in_pos = 0.0, False
    cb_g = cb_n = 0.0
    last_ei = last_xi = None
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
                    trades.append({"Date": de, "Typ": "Entry", "Price": round(sb, 4), "Shares": round(sh, 4), "Gross P&L": 0.0, "Fees": round(fee, 2), "Net P&L": 0.0, "kum P&L": round(cum, 2), "Prob": round(pp, 4), "HoldDays": np.nan})
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
                    cb_g = cb_n = 0.0
                    cum += pnl_n
                    trades.append({"Date": de, "Typ": "Exit", "Price": round(ss, 4), "Shares": 0.0, "Gross P&L": round(pnl_g, 2), "Fees": round(fe, 2), "Net P&L": round(pnl_n, 2), "kum P&L": round(cum, 2), "Prob": round(pp, 4), "HoldDays": int(held)})
                    last_xi = i
                    last_ei = None

        ct = float(df["Close"].iloc[i])
        eq_g.append(cash_g + (shares * ct if in_pos else 0.0))
        eq_n.append(cash_n + (shares * ct if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = eq_g
    df_bt["Equity_Net"] = eq_n
    return df_bt, trades


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
                    "Ticker": tk, "Name": name, "Entry Date": ed, "Exit Date": xd, "Hold (days)": (xd - ed).days,
                    "Entry Prob": ce.get("Prob", np.nan), "Exit Prob": ev.get("Prob", np.nan), "Shares": round(sh, 4),
                    "Entry Price": round(ep, 4), "Exit Price": round(xp, 4), "PnL Net (€)": round(pnl, 2),
                    "Fees (€)": round(fe + fx, 2), "Return (%)": round(pnl / cost * 100, 2) if cost else np.nan
                })
                ce = None
    return pd.DataFrame(rows)


def _train_probs(X_train, y_train, X_pred, model_params, calibrated=False):
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_pred), 0.5, dtype=float)
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_pred_imp = imp.transform(X_pred)
    est = _build_estimator(model_params, calibrated)
    est.fit(X_train_imp, y_train)
    return est.predict_proba(X_pred_imp)[:, 1]


def make_features_and_train(df, lookback, horizon, threshold, model_params, entry_prob, exit_prob, init_capital, pos_frac, min_hold_days=0, cooldown_days=0, walk_forward=True, wf_min_train=180, wf_refit_step=5, calibration=True):
    feat = make_features(df, lookback, horizon)
    hist = feat.iloc[:-1].copy().dropna(subset=["FutureRetExec"])
    if len(hist) < max(90, lookback + horizon + 30):
        raise ValueError("Zu wenige Datenpunkte.")

    X_cols = [c for c in BASE_FEATURES if c in hist.columns]
    hist["Target"] = (hist["FutureRetExec"] > threshold).astype(int)
    if len(X_cols) < 6:
        raise ValueError("Zu wenige Features nach Bereinigung.")

    if hist["Target"].nunique() < 2:
        feat["SignalProb"] = 0.5
    elif not walk_forward:
        feat["SignalProb"] = _train_probs(hist[X_cols].values, hist["Target"].values, feat[X_cols].values, model_params, calibration)
    else:
        probs = np.full(len(feat), np.nan)
        min_train = max(int(wf_min_train), lookback + horizon + 40)
        imp = None
        est = None
        last_fit_t = None
        for t in range(min_train, len(feat)):
            tr = feat.iloc[:t].dropna(subset=["FutureRetExec"]).copy()
            tr["Target"] = (tr["FutureRetExec"] > threshold).astype(int)
            if len(tr) < min_train:
                continue
            if tr["Target"].nunique() < 2 or tr["Target"].value_counts().min() < 10:
                probs[t] = 0.5
                continue
            should_refit = (est is None) or (last_fit_t is None) or ((t - last_fit_t) >= int(wf_refit_step))
            if should_refit:
                imp = SimpleImputer(strategy="median")
                X_train_imp = imp.fit_transform(tr[X_cols].values)
                est = _build_estimator(model_params, calibration)
                est.fit(X_train_imp, tr["Target"].values)
                last_fit_t = t
            X_one = imp.transform(feat[X_cols].iloc[[t]].values)
            probs[t] = est.predict_proba(X_one)[0, 1]
        feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)

    df_bt, trades = backtest_next_open(feat.iloc[:-1].copy(), entry_prob, exit_prob, COMMISSION, SLIPPAGE_BPS, init_capital, pos_frac, int(min_hold_days), int(cooldown_days))
    return feat, df_bt, trades, compute_performance(df_bt, trades, init_capital)

# ─────────────────────────────────────────────────────────────
# Minimal Charts
# ─────────────────────────────────────────────────────────────
def chart_price_signal(feat, trades, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=feat.index, open=feat["Open"], high=feat["High"], low=feat["Low"], close=feat["Close"], showlegend=False), row=1, col=1)
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["Date"] = pd.to_datetime(tdf["Date"])
        for typ, sym, col in [("Entry", "triangle-up", THEME["green"]), ("Exit", "triangle-down", THEME["red"])]:
            sub = tdf[tdf["Typ"] == typ]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Price"], mode="markers", marker=dict(symbol=sym, size=10, color=col), name=typ), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat.index, y=feat["SignalProb"], mode="lines", line=dict(color=THEME["accent4"], width=1.5), name="Signal Prob"), row=2, col=1)
    fig.add_hline(y=ENTRY_PROB, row=2, col=1, line_color=THEME["green"], line_dash="dash")
    fig.add_hline(y=EXIT_PROB, row=2, col=1, line_color=THEME["red"], line_dash="dash")
    _apply_theme(fig, 500)
    fig.update_layout(title=f"{ticker} · Preis & Signal", xaxis_rangeslider_visible=False, yaxis2=dict(range=[0, 1]))
    return fig


def chart_equity(df_bt, ticker, init_cap):
    eq = df_bt["Equity_Net"]
    bh = init_cap * df_bt["Close"] / df_bt["Close"].iloc[0]
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Strategie", line=dict(color=THEME["accent1"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh, mode="lines", name="Buy & Hold", line=dict(color=THEME["muted"], width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name="Drawdown", line=dict(color=THEME["red"], width=1)), row=2, col=1)
    _apply_theme(fig, 450)
    fig.update_layout(title=f"{ticker} · Equity & Drawdown")
    return fig


def chart_intraday(intra, ticker, chart_type, interval):
    fig = go.Figure()
    if chart_type == "Candlestick (OHLC)":
        fig.add_trace(go.Candlestick(x=intra.index, open=intra["Open"], high=intra["High"], low=intra["Low"], close=intra["Close"], showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=intra.index, y=intra["Close"], mode="lines", line=dict(color=THEME["accent1"], width=1.5), showlegend=False))
    _apply_theme(fig, 400)
    fig.update_layout(title=f"{ticker} · Intraday 5d ({interval})", xaxis_rangeslider_visible=False)
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

    if st.button("🚀 Suche starten", type="primary", use_container_width=True):
        import random
        from collections import Counter
        rng_opt = random.Random(int(seed_opt))
        pm = load_all_prices(TICKERS, str(START_DATE), str(END_DATE), use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min))[0]
        feasible = {tk: df.copy() for tk, df in (pm or {}).items() if isinstance(df, pd.DataFrame) and len(df) >= max(int(wf_min_train) + 40, 140)}
        if len(feasible) < int(min_valid_tickers):
            st.error("Zu wenige Ticker nach Prefilter.")
            st.stop()
        rows_o, best_o = [], None
        prog_o = st.progress(0.0)
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
                    feat_o, df_bt_o, tr_o, _ = make_features_and_train(df0, int(p["lookback"]), int(p["horizon"]), float(p["thresh"]), MODEL_PARAMS, float(p["entry"]), float(p["exit"]), float(INIT_CAP_PER_TICKER), float(POS_FRAC), int(MIN_HOLD_DAYS), int(COOLDOWN_DAYS), True, int(wf_min_train), int(wf_refit_step), bool(calibration_on))
                    oos_start_idx = max(int(wf_min_train), int(0.55 * len(df_bt_o)))
                    oos_slice = df_bt_o.iloc[oos_start_idx:].copy()
                    r = oos_slice["Equity_Net"].pct_change().dropna()
                    closed_trades = int(sum(1 for t in tr_o if t["Typ"] == "Exit"))
                    if len(r) < 20 or closed_trades < int(min_trades_req):
                        raise ValueError("schwacher_oos_block")
                    sh = float((r.mean() / (r.std(ddof=0) + 1e-12)) * np.sqrt(252))
                    dd = float(((oos_slice["Equity_Net"] / oos_slice["Equity_Net"].cummax()) - 1).min())
                    rr = float(oos_slice["Equity_Net"].iloc[-1] / oos_slice["Equity_Net"].iloc[0] - 1)
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
            score = sharpe_med + 0.35 * ret_med + 0.15 * dd_med
            rec = dict(trial=trial, score=score, sharpe_med=sharpe_med, ret_med=ret_med, dd_med=dd_med, trades=trad_o, ok_tickers=ok_t, **p)
            rows_o.append(rec)
            if best_o is None or score > best_o["score"]:
                best_o = rec
            prog_o.progress((trial + 1) / int(n_trials))

        if not rows_o:
            st.error("Keine gültigen Ergebnisse.")
            if err_c:
                st.dataframe(pd.DataFrame(err_c.most_common(10), columns=["Error", "Count"]))
        else:
            df_res_o = pd.DataFrame(rows_o).sort_values("score", ascending=False)
            st.success(f"✅ Best Score: **{best_o['score']:.3f}** · Sharpe: **{best_o['sharpe_med']:.2f}** · Median Return: **{best_o['ret_med']*100:.2f}%**")
            st.dataframe(df_res_o.head(25), use_container_width=True)
            st.download_button("⬇ Optimizer-Ergebnisse", to_csv_eu(df_res_o), file_name="optimizer_results.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# Main
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

price_map, meta_map = load_all_prices(TICKERS, str(START_DATE), str(END_DATE), use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min))

for ticker in TICKERS:
    if ticker not in price_map:
        continue
    df = price_map[ticker]
    meta = meta_map.get(ticker, {})
    name = get_ticker_name(ticker)
    with st.expander(f"⚡  {ticker}  ·  {name}", expanded=False):
        try:
            feat, df_bt, trades, metrics = make_features_and_train(df, int(LOOKBACK), int(HORIZON), float(THRESH), MODEL_PARAMS, float(ENTRY_PROB), float(EXIT_PROB), float(INIT_CAP_PER_TICKER), float(POS_FRAC), int(MIN_HOLD_DAYS), int(COOLDOWN_DAYS), bool(use_walk_forward), int(wf_min_train), int(wf_refit_step), bool(calibration_on))
            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat
            all_strategy_rets[ticker] = df_bt["Equity_Net"].pct_change()

            kpi_row([
                ("NETTO", _pct(metrics["Strategy Net (%)"]), "kpi-pos" if metrics["Strategy Net (%)"] > 0 else "kpi-neg"),
                ("BUY & HOLD", _pct(metrics["Buy & Hold (%)"]), "kpi-pos" if metrics["Buy & Hold (%)"] > 0 else "kpi-neg"),
                ("SHARPE", f"{metrics['Sharpe-Ratio']:.2f}", "kpi-info"),
                ("MAX DD", _pct(metrics["Max Drawdown (%)"]), "kpi-neg"),
                ("WINRATE", f"{metrics['Winrate (%)']:.1f}%" if np.isfinite(metrics["Winrate (%)"]) else "–", ""),
                ("TRADES", f"{metrics['Closed Trades']}", ""),
                ("PHASE", metrics["Phase"], "kpi-info" if metrics["Phase"] == "Open" else ""),
            ])

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_price_signal(feat, trades, ticker), use_container_width=True, config={"displayModeBar": False})
            intra = get_intraday_last_n_sessions(ticker, 5, 10, intraday_interval)
            with c2:
                if intra.empty:
                    st.info("Keine Intraday-Daten.")
                else:
                    st.plotly_chart(chart_intraday(intra, ticker, intraday_chart_type, intraday_interval), use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(chart_equity(df_bt, ticker, float(INIT_CAP_PER_TICKER)), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"❌ {ticker}: {e}")
            st.code(traceback.format_exc(), language="python")

if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    section("PORTFOLIO SUMMARY")
    total_net = float(summary_df["Net P&L (€)"].sum())
    total_fees = float(summary_df["Fees (€)"].sum())
    kpi_row([
        ("Netto P&L", _eur(total_net), "kpi-pos" if total_net > 0 else "kpi-neg"),
        ("Trading Costs", f"–{total_fees:,.2f}€", "kpi-neg"),
        ("Trades", f"{int(summary_df['Closed Trades'].sum())}", ""),
        ("Ø CAGR", f"{summary_df['CAGR (%)'].dropna().mean():.2f}%", "kpi-info"),
    ])
    st.dataframe(summary_df, use_container_width=True)
    st.download_button("⬇ Summary CSV", to_csv_eu(summary_df.reset_index()), file_name="strategy_summary.csv", mime="text/csv")
else:
    st.warning("⚠ Keine Ergebnisse — Ticker & Datenabdeckung prüfen.")
