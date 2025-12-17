# streamlit_app.py
# ─────────────────────────────────────────────────────────────
# NEXT LEVEL 2ND MODELL – Signal-basierte Strategie (Full Version, FIXED)
# Fokus: OOS-Validität, sauberes Timing, robuste Targets, bessere Features
# - Training/Backtest NUR mit finalen Daily Bars (kein Intraday-Patch im Modell)
# - Intraday NUR für Anzeige
# - Walk-Forward standardmäßig ON + Embargo (Label-Overlap vermeiden)
# - Target wahlweise: Fix-Threshold, Vol-adjust (k·σ), Quantil
# - Features erweitert (Returns, Vol, Gap, ATR, MA-Spread, RSI-light)
# - Optional: Proba-Kalibrierung
# - Forecast: Bootstrap (statt Normal-MC)
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NEXT LEVEL 2ND AI-MODELL (FIXED)", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 6  # yfinance rate-limit sensibel
pd.options.display.float_format = "{:,.4f}".format


# ─────────────────────────────────────────────────────────────
# Helpers (CSV)
# ─────────────────────────────────────────────────────────────
def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(
        index=False, sep=";", decimal=",", date_format="%d.%m.%Y", float_format=float_format
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
    for key in ("ticker", "symbol", "symbols", "isin", "code"):
        if key in cols_lower:
            col = cols_lower[key]
            return _normalize_tickers(df[col].astype(str).tolist())
    first = df.columns[0]
    return _normalize_tickers(df[first].astype(str).tolist())


# ─────────────────────────────────────────────────────────────
# UI – Sidebar Controls
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameter")

ticker_source = st.sidebar.selectbox("Ticker-Quelle", ["Manuell (Textfeld)", "CSV-Upload"], index=0)
tickers_final: List[str] = []

if ticker_source == "Manuell (Textfeld)":
    tickers_input = st.sidebar.text_input("Tickers (Komma-getrennt)", value="REGN, LULU, VOW3.DE, REI, DDL")
    tickers_final = _normalize_tickers([t for t in tickers_input.split(",") if t.strip()])
else:
    st.sidebar.caption("Lade CSVs mit Spalte **ticker** (oder erste Spalte).")
    uploads = st.sidebar.file_uploader("CSV-Dateien", type=["csv"], accept_multiple_files=True)
    collected = []
    if uploads:
        for up in uploads:
            try:
                collected += parse_ticker_csv(up)
            except Exception as e:
                st.sidebar.error(f"Fehler beim Lesen von '{up.name}': {e}")
    base = _normalize_tickers(collected)
    extra_csv = st.sidebar.text_input("Weitere Ticker manuell hinzufügen (Komma-getrennt)", value="", key="extra_csv")
    extras = _normalize_tickers([t for t in extra_csv.split(",") if t.strip()]) if extra_csv else []
    tickers_final = _normalize_tickers(base + extras)

    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker: {len(tickers_final)}")
        if st.sidebar.checkbox("Zufällig mischen", value=False, help="Reihenfolge zufällig (seed=42)"):
            import random
            random.seed(42)
            random.shuffle(tickers_final)
        max_n = st.sidebar.number_input("Max. Anzahl (0 = alle)", min_value=0, max_value=len(tickers_final), value=0, step=10)
        if max_n and max_n < len(tickers_final):
            tickers_final = tickers_final[:int(max_n)]
        tickers_final = st.sidebar.multiselect("Auswahl verfeinern", options=tickers_final, default=tickers_final)

if not tickers_final:
    tickers_final = _normalize_tickers(["REGN", "VOW3.DE", "LULU", "REI", "DDL"])

st.sidebar.download_button(
    "Kombinierte Ticker als CSV",
    to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
    file_name="tickers_combined.csv", mime="text/csv"
)

TICKERS = tickers_final

# Core Parameter
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
END_DATE   = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", 10, 252, 60, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)", 1, 10, 5)

# ───── Target Definition (NEU, robust)
st.sidebar.markdown("**Target Definition (robust)**")
target_mode = st.sidebar.selectbox(
    "Target-Modus",
    ["Fix (FutureRet > THRESH)", "Vol-adjust (FutureRet > k·σ)", "Quantil (FutureRet > q)"],
    index=1
)
THRESH = st.sidebar.number_input(
    "THRESH (nur Fix)", 0.0, 0.20, 0.046, step=0.005, format="%.3f",
    help="Nur relevant im Fix-Modus."
)
k_vol = st.sidebar.number_input(
    "k (nur Vol-adjust)", 0.0, 10.0, 1.0, step=0.1, format="%.1f",
    help="Target=1 wenn FutureRet > k * rolling_sigma (auf Open→Open Returns)."
)
q_quant = st.sidebar.slider(
    "q (nur Quantil)", 0.50, 0.95, 0.70, step=0.01,
    help="Target=1 wenn FutureRet oberhalb Trainings-Quantil q liegt (im jeweiligen Train-Fenster)."
)

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", 0.0, 1.0, 0.62, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  0.0, 1.0, 0.48, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()

MIN_HOLD_DAYS = st.sidebar.number_input("Mindesthaltedauer (Handelstage)", 0, 252, 5, step=1)
COOLDOWN_DAYS = st.sidebar.number_input("Cooling Phase nach Exit (Handelstage)", 0, 252, 0, step=1)

COMMISSION   = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", 0.0, 0.02, 0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", 0, 50, 5, step=1)
POS_FRAC     = st.sidebar.slider("Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1)

INIT_CAP_PER_TICKER = st.sidebar.number_input("Initial Capital pro Ticker (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# Intraday – Anzeige only
st.sidebar.markdown("**Intraday (Anzeige only)**")
show_intraday = st.sidebar.checkbox("Intraday-Charts anzeigen (kein Einfluss auf Modell)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall (5-Tage-Chart)", ["1m", "2m", "5m", "15m"], index=2)
intraday_chart_type = st.sidebar.selectbox("Intraday-Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

# Modellparameter
st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 150, step=10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 1.0, 0.05, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     1, 10, 3, step=1)
MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42
)

# OOS / Walk-forward (standardmäßig ON)
st.sidebar.markdown("**OOS / Walk-Forward**")
use_walk_forward = st.sidebar.checkbox("Walk-Forward Probas verwenden (empfohlen)", value=True)
wf_min_train = st.sidebar.number_input("WF min_train Bars", 40, 800, 200, step=10)
embargo_bars = st.sidebar.number_input(
    "Embargo Bars (Label-Overlap vermeiden)", 0, 30, int(HORIZON), step=1,
    help="Train endet bei t-(embargo). Typisch = Horizon."
)

# Kalibrierung
st.sidebar.markdown("**Probability Calibration**")
use_calibration = st.sidebar.checkbox("Probas kalibrieren (sigmoid)", value=False)
calib_cv = st.sidebar.number_input("Kalibrierung CV-Folds", 2, 10, 3, step=1)

# Forecast
st.sidebar.markdown("**Portfolio Forecast**")
FORECAST_DAYS = st.sidebar.number_input("Forecast Horizon (Tage)", 1, 30, 7, step=1)
BOOT_SIMS = st.sidebar.number_input("Bootstrap Simulationen", 200, 20000, 5000, step=200)
block_len = st.sidebar.number_input("Block-Länge (Tage, Bootstrap)", 1, 30, 5, step=1)

# Housekeeping
c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear()
    st.rerun()
if c2.button("↻ Rerun"):
    st.rerun()


# ─────────────────────────────────────────────────────────────
# Misc Helpers
# ─────────────────────────────────────────────────────────────
def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0


def last_timestamp_info(df: pd.DataFrame):
    ts = df.index[-1]
    st.caption(f"Letzter Daily-Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}")


@st.cache_data(show_spinner=False, ttl=24*60*60)
def get_ticker_name(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        info = {}
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


def show_styled_or_plain(df: pd.DataFrame, styler):
    try:
        st.markdown(styler.to_html(), unsafe_allow_html=True)
    except Exception:
        st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Data Loading (Daily only for model/backtest)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=6*60*60)
def get_price_data_daily(ticker: str, years: int = 5) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)
    df = df.sort_index().drop_duplicates()
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return df


@st.cache_data(show_spinner=False, ttl=15*60)
def get_intraday_last_n_sessions(ticker: str, sessions: int = 5, days_buffer: int = 10, interval: str = "5m") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr = intr.sort_index()
    unique_dates = pd.Index(intr.index.normalize().unique())
    keep_dates = set(unique_dates[-sessions:])
    mask = intr.index.normalize().isin(keep_dates)
    return intr.loc[mask].copy()


def load_all_prices_daily(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    price_map: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return price_map

    st.info(f"Daily-Kurse laden für {len(tickers)} Ticker … (parallel)")
    prog = st.progress(0.0)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        future_map = {ex.submit(get_price_data_daily, tk, 5): tk for tk in tickers}
        done = 0
        for fut in as_completed(future_map):
            tk = future_map[fut]
            try:
                df_full = fut.result()
                df_use = df_full.loc[str(start):str(end)].copy()
                if not df_use.empty:
                    price_map[tk] = df_use
            except Exception as e:
                st.error(f"Fehler beim Laden von {tk}: {e}")
            finally:
                done += 1
                prog.progress(done / len(tickers))

    return price_map


# ─────────────────────────────────────────────────────────────
# Features & Target (sauberes Timing)
# ─────────────────────────────────────────────────────────────
def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rs = up.rolling(period).mean() / (dn.rolling(period).mean() + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    if len(df) < (lookback + horizon + 30):
        raise ValueError("Zu wenige Bars. Zeitraum verlängern oder Parameter senken.")

    feat = df.copy()

    # Baseline (dein Original)
    feat["Range"]     = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)

    # Simple, robuste Add-ons
    feat["Ret_1d"]  = feat["Close"].pct_change(1)
    feat["Ret_5d"]  = feat["Close"].pct_change(5)
    feat["Ret_20d"] = feat["Close"].pct_change(20)

    feat["Gap"] = feat["Open"] / feat["Close"].shift(1) - 1.0

    tr = _true_range(feat)
    feat["ATR_14"] = tr.rolling(14).mean()
    feat["ATRpct_14"] = feat["ATR_14"] / (feat["Close"] + 1e-12)

    feat["Vol_20"] = feat["Ret_1d"].rolling(20).std(ddof=0)
    feat["Vol_60"] = feat["Ret_1d"].rolling(60).std(ddof=0)

    ma20 = feat["Close"].rolling(20).mean()
    ma60 = feat["Close"].rolling(60).mean()
    feat["MA20_spread"] = feat["Close"] / (ma20 + 1e-12) - 1.0
    feat["MA60_spread"] = feat["Close"] / (ma60 + 1e-12) - 1.0

    feat["RSI_14"] = _rsi(feat["Close"], 14)

    # Align
    feat = feat.iloc[max(lookback-1, 60):].copy()

    # Label/Realisation (Next Open Execution – konsistent)
    # FutureRetExec(t) = Open(t+horizon) / Open(t+1) - 1
    feat["FutureRetExec"] = feat["Open"].shift(-horizon) / feat["Open"].shift(-1) - 1.0

    return feat


def build_target(train_future_ret: pd.Series, mode: str, thr: float, k: float, q: float, vol_ref: Optional[pd.Series]) -> pd.Series:
    fr = pd.to_numeric(train_future_ret, errors="coerce")

    if mode.startswith("Fix"):
        return (fr > float(thr)).astype(int)

    if mode.startswith("Vol-adjust"):
        if vol_ref is None:
            raise ValueError("Vol-adjust: vol_ref fehlt.")
        # vol_ref ist eine rolling sigma (z.B. Open→Open oder Close→Close Proxy)
        v = pd.to_numeric(vol_ref, errors="coerce")
        return (fr > float(k) * v).astype(int)

    # Quantil
    qq = float(np.nanquantile(fr.values, float(q))) if np.isfinite(fr).sum() > 10 else np.nan
    if not np.isfinite(qq):
        return pd.Series(np.nan, index=fr.index)
    return (fr > qq).astype(int)


def make_pipe(model_params: dict, calibrate: bool, calib_folds: int):
    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(**model_params)),
    ])
    if not calibrate:
        return base
    # CalibratedClassifierCV braucht estimator, nicht Pipeline – wir kalibrieren das Modell nach Imputer
    # Pragmatiker-Variante: Pipeline für X->impute und dahinter Calibrator mit Base-Model
    # -> wir machen’s sauber mit zweistufigem Fit im Code (siehe train/predict unten).
    return base


# ─────────────────────────────────────────────────────────────
# Backtest (Next Open) – pro Ticker separates Konto
# ─────────────────────────────────────────────────────────────
def backtest_next_open(
    df: pd.DataFrame,
    entry_thr: float,
    exit_thr: float,
    commission: float,
    slippage_bps: int,
    init_cap: float,
    pos_frac: float,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
) -> Tuple[pd.DataFrame, List[dict]]:

    df = df.copy()
    n = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = init_cap
    cash_net   = init_cap
    shares     = 0.0
    in_pos     = False

    cost_basis_gross = 0.0
    cost_basis_net   = 0.0

    last_entry_idx: Optional[int] = None
    last_exit_idx:  Optional[int] = None

    equity_gross, equity_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        if i > 0:
            open_today = float(df["Open"].iloc[i])
            slip_buy  = open_today * (1 + slippage_bps / 10000.0)
            slip_sell = open_today * (1 - slippage_bps / 10000.0)
            prob_prev = float(df["SignalProb"].iloc[i-1])
            date_exec = df.index[i]

            cool_ok = True
            if (not in_pos) and cooldown_days > 0 and last_exit_idx is not None:
                bars_since_exit = i - last_exit_idx
                cool_ok = bars_since_exit >= int(cooldown_days)

            can_enter = (not in_pos) and (prob_prev > entry_thr) and cool_ok
            if can_enter:
                invest_net    = cash_net * float(pos_frac)
                fee_entry     = invest_net * float(commission)
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)

                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-6:
                    shares = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry
                    cash_gross -= cost_basis_gross
                    cash_net   -= cost_basis_net
                    in_pos = True
                    last_entry_idx = i
                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 4),
                        "Shares": round(shares, 4), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0,
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldDays": np.nan
                    })

            elif in_pos and prob_prev < exit_thr:
                held_bars = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_days) > 0 and held_bars < int(min_hold_days):
                    pass
                else:
                    gross_value = shares * slip_sell
                    fee_exit    = gross_value * float(commission)
                    pnl_gross   = gross_value - cost_basis_gross
                    pnl_net     = (gross_value - fee_exit) - cost_basis_net

                    cash_gross += gross_value
                    cash_net   += (gross_value - fee_exit)

                    in_pos = False
                    shares = 0.0
                    cost_basis_gross = 0.0
                    cost_basis_net   = 0.0

                    cum_pl_net += pnl_net
                    trades.append({
                        "Date": date_exec, "Typ": "Exit", "Price": round(slip_sell, 4),
                        "Shares": 0.0, "Gross P&L": round(pnl_gross, 2),
                        "Fees": round(fee_exit, 2), "Net P&L": round(pnl_net, 2),
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldDays": int(held_bars)
                    })
                    last_exit_idx = i
                    last_entry_idx = None

        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net + (shares * close_today if in_pos else 0.0))

    df_bt = df.copy()
    df_bt["Equity_Gross"] = equity_gross
    df_bt["Equity_Net"]   = equity_net
    return df_bt, trades


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────
def _cagr_from_path(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    dt0 = pd.to_datetime(values.index[0])
    dt1 = pd.to_datetime(values.index[-1])
    years = max((dt1 - dt0).days / 365.25, 1e-9)
    return (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1


def _sortino(rets: pd.Series) -> float:
    if rets.empty:
        return np.nan
    mean = rets.mean() * 252
    downside = rets[rets < 0]
    dd = downside.std(ddof=0) * np.sqrt(252) if len(downside) else np.nan
    return mean / dd if dd and np.isfinite(dd) and dd > 0 else np.nan


def _winrate_roundtrips(trades: List[dict]) -> float:
    if not trades:
        return np.nan
    pnl = []
    entry = None
    for ev in trades:
        if ev["Typ"] == "Entry":
            entry = ev
        elif ev["Typ"] == "Exit" and entry is not None:
            pnl.append(float(ev.get("Net P&L", 0.0)))
            entry = None
    if not pnl:
        return np.nan
    pnl = np.array(pnl, dtype=float)
    return float((pnl > 0).mean())


def compute_performance(df_bt: pd.DataFrame, trades: List[dict], init_cap: float) -> dict:
    net_ret = (df_bt["Equity_Net"].iloc[-1] / init_cap - 1) * 100
    rets = df_bt["Equity_Net"].pct_change().dropna()
    vol_ann = rets.std(ddof=0) * sqrt(252) * 100
    sharpe = (rets.mean() * sqrt(252)) / (rets.std(ddof=0) + 1e-12)
    dd = (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax()) / df_bt["Equity_Net"].cummax()
    max_dd = dd.min() * 100
    calmar = (net_ret/100) / abs(max_dd/100) if max_dd < 0 else np.nan
    gross_ret = (df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100
    bh_ret = (df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100
    fees = float(sum(t.get("Fees", 0.0) for t in trades))
    phase = "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat"
    completed = int(sum(1 for t in trades if t["Typ"] == "Exit"))
    net_eur = float(df_bt["Equity_Net"].iloc[-1] - init_cap)
    cagr = _cagr_from_path(df_bt["Equity_Net"])
    sortino = _sortino(rets)
    winrate = _winrate_roundtrips(trades)
    return {
        "Strategy Net (%)": round(float(net_ret), 2),
        "Strategy Gross (%)": round(float(gross_ret), 2),
        "Buy & Hold Net (%)": round(float(bh_ret), 2),
        "Volatility (%)": round(float(vol_ann), 2),
        "Sharpe-Ratio": round(float(sharpe), 2),
        "Sortino-Ratio": round(float(sortino), 2) if np.isfinite(sortino) else np.nan,
        "Max Drawdown (%)": round(float(max_dd), 2),
        "Calmar-Ratio": round(float(calmar), 2) if np.isfinite(calmar) else np.nan,
        "Fees (€)": round(float(fees), 2),
        "Phase": phase,
        "Number of Trades": completed,
        "Net P&L (€)": round(float(net_eur), 2),
        "CAGR (%)": round(100*(float(cagr) if np.isfinite(cagr) else np.nan), 2),
        "Winrate (%)": round(100*(float(winrate) if np.isfinite(winrate) else np.nan), 2),
        "InitCap (€)": float(init_cap),
    }


# ─────────────────────────────────────────────────────────────
# Walk-Forward Training (OOS) + Embargo + optional Calibration
# ─────────────────────────────────────────────────────────────
def fit_predict_proba(
    X_train: np.ndarray, y_train: np.ndarray,
    X_pred: np.ndarray,
    model_params: dict,
    calibrate: bool,
    calib_folds: int
) -> np.ndarray:
    # Pipeline: impute + model
    base = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(**model_params)),
    ])

    if not calibrate:
        base.fit(X_train, y_train)
        return base.predict_proba(X_pred)[:, 1]

    # Calibrate on top of base pipeline
    # Wichtig: Calibrator darf nur auf Train laufen, NICHT auf Zukunft.
    calib = CalibratedClassifierCV(base, method="sigmoid", cv=int(calib_folds))
    calib.fit(X_train, y_train)
    return calib.predict_proba(X_pred)[:, 1]


def make_features_and_train(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    target_mode: str,
    thr_fix: float,
    k_vol: float,
    q_quant: float,
    model_params: dict,
    walk_forward: bool,
    wf_min_train: int,
    embargo: int,
    calibrate: bool,
    calib_folds: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict, dict]:

    feat = make_features(df, lookback, horizon)

    # Historie (ohne letzte Zeile – die hat kein komplettes Label)
    hist = feat.iloc[:-1].dropna(subset=["FutureRetExec"]).copy()
    if len(hist) < max(60, int(wf_min_train)):
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing. Zeitraum erweitern.")

    # Vol-Referenz für Vol-adjust: rolling sigma auf Open→Open (Proxy)
    open_ret_1d = feat["Open"].pct_change(1)
    vol_ref = open_ret_1d.rolling(20).std(ddof=0)  # simple, robust
    hist["VolRef"] = vol_ref.loc[hist.index]

    # Feature Columns
    X_cols = [
        "Range", "SlopeHigh", "SlopeLow",
        "Ret_1d", "Ret_5d", "Ret_20d",
        "Gap", "ATRpct_14",
        "Vol_20", "Vol_60",
        "MA20_spread", "MA60_spread",
        "RSI_14",
    ]
    X_cols = [c for c in X_cols if c in feat.columns]

    # Target bauen (im Train-Fenster)
    # -> Für Quantil hängt die Schwelle vom Train ab; für Fix/Vol-adjust können wir global bauen,
    #    aber in WF wird es trotzdem pro Train berechnet (robuster).
    def _build_y(train_df: pd.DataFrame) -> pd.Series:
        fr = train_df["FutureRetExec"]
        if target_mode.startswith("Vol-adjust"):
            return build_target(fr, target_mode, thr_fix, k_vol, q_quant, train_df["VolRef"])
        return build_target(fr, target_mode, thr_fix, k_vol, q_quant, None)

    # Probas
    probs = np.full(len(feat), np.nan, dtype=float)

    # In-Sample (nicht empfohlen)
    if not walk_forward:
        train_df = hist.copy()
        y = _build_y(train_df)
        train_df = train_df.assign(Target=y).dropna(subset=["Target"])
        if train_df["Target"].nunique() < 2:
            feat["SignalProb"] = 0.5
        else:
            X_train = train_df[X_cols].values
            y_train = train_df["Target"].values.astype(int)
            p_all = fit_predict_proba(X_train, y_train, feat[X_cols].values, model_params, calibrate, calib_folds)
            feat["SignalProb"] = p_all
    else:
        min_train = max(int(wf_min_train), 120)
        embargo_use = max(int(embargo), 0)

        # Walk forward: für jeden t ein Modell mit Daten bis (t - embargo_use)
        for t in range(min_train, len(feat)):
            end_train = t - embargo_use
            if end_train <= 30:
                continue

            train_window = feat.iloc[:end_train].dropna(subset=["FutureRetExec"]).copy()
            if len(train_window) < min_train:
                continue

            # y im Train-Fenster
            train_window["VolRef"] = vol_ref.loc[train_window.index]
            y = _build_y(train_window)
            train_window = train_window.assign(Target=y).dropna(subset=["Target"])

            if train_window["Target"].nunique() < 2:
                continue

            X_train = train_window[X_cols].values
            y_train = train_window["Target"].values.astype(int)

            X_pred = feat[X_cols].iloc[[t]].values
            probs[t] = fit_predict_proba(X_train, y_train, X_pred, model_params, calibrate, calib_folds)[0]

        feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)

    # Backtest nutzt feat bis vor letzter Zeile
    feat_bt = feat.iloc[:-1].copy()
    df_bt, trades = backtest_next_open(
        feat_bt,
        float(ENTRY_PROB), float(EXIT_PROB),
        float(COMMISSION), int(SLIPPAGE_BPS),
        float(INIT_CAP_PER_TICKER),
        float(POS_FRAC),
        min_hold_days=int(MIN_HOLD_DAYS),
        cooldown_days=int(COOLDOWN_DAYS),
    )
    metrics = compute_performance(df_bt, trades, float(INIT_CAP_PER_TICKER))

    # Diagnostics: OOS quality (Brier + bucket edge)
    diag = {}
    try:
        # Nur dort, wo wir Label kennen (hist)
        diag_df = feat.iloc[:-1].copy()
        diag_df["VolRef"] = vol_ref.loc[diag_df.index]
        y_all = _build_y(diag_df).astype(float)
        p_all = pd.to_numeric(diag_df["SignalProb"], errors="coerce")

        m = np.isfinite(y_all.values) & np.isfinite(p_all.values)
        if m.sum() > 80 and len(np.unique(y_all.values[m].astype(int))) > 1:
            yv = y_all.values[m].astype(int)
            pv = p_all.values[m].clip(0.0, 1.0)

            diag["Brier"] = float(brier_score_loss(yv, pv))

            # Bucket edge: oberstes Dezil vs Rest
            q90 = np.quantile(pv, 0.90)
            top = pv >= q90
            mu_top = float(np.nanmean(diag_df.loc[m].loc[top, "FutureRetExec"]))
            mu_all = float(np.nanmean(diag_df.loc[m, "FutureRetExec"]))
            diag["Mu_FutureRetExec_top10p"] = mu_top
            diag["Mu_FutureRetExec_all"] = mu_all
            diag["Edge_top10p_minus_all"] = mu_top - mu_all
    except Exception:
        pass

    return feat, df_bt, trades, metrics, diag


# ─────────────────────────────────────────────────────────────
# Portfolio Forecast – Bootstrap (Block)
# ─────────────────────────────────────────────────────────────
def bootstrap_portfolio_forecast(
    rets: pd.DataFrame,
    nav0: float,
    horizon_days: int,
    sims: int,
    block: int,
    seed: int = 42
) -> dict:
    """
    Block-Bootstrap auf Return-Vektoren (multivariate historisch, ohne Normalannahme).
    rets: DataFrame (daily returns, aligned), Spalten=Tickers
    """
    rng = np.random.default_rng(int(seed))
    rets = rets.dropna(how="any")
    if rets.empty or len(rets) < max(120, horizon_days * 10):
        raise ValueError("Zu wenig Return-Historie für Bootstrap-Forecast.")

    tickers = rets.columns.tolist()
    n = len(rets)
    w = np.ones(len(tickers), dtype=float) / len(tickers)

    H = int(horizon_days)
    B = max(1, int(block))
    steps = int(np.ceil(H / B))

    port_rets = np.empty(int(sims), dtype=float)

    rets_np = rets.values
    for s in range(int(sims)):
        idxs = []
        for _ in range(steps):
            start = rng.integers(0, max(1, n - B))
            idxs.extend(range(start, start + B))
        idxs = idxs[:H]
        path = rets_np[idxs, :]  # H x N
        # Horizon return (compounded)
        port_path = (1.0 + (path @ w)).prod() - 1.0
        port_rets[s] = port_path

    nav_paths = nav0 * (1.0 + port_rets)
    q = np.quantile(port_rets, [0.05, 0.50, 0.95])
    q_nav = np.quantile(nav_paths, [0.05, 0.50, 0.95])

    return dict(
        port_ret_q05=float(q[0]), port_ret_q50=float(q[1]), port_ret_q95=float(q[2]),
        nav_q05=float(q_nav[0]), nav_q50=float(q_nav[1]), nav_q95=float(q_nav[2]),
        port_rets=port_rets, nav_paths=nav_paths,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 36px;'>📈 NEXT LEVEL 2ND AI-MODELL (FIXED)</h1>", unsafe_allow_html=True)
st.caption(
    "Wichtig: Modell/Backtest nutzt ausschließlich finale Daily Bars. Intraday wird nur visualisiert. "
    "Walk-Forward + Embargo ist Standard, damit die Probas OOS-ähnlicher sind."
)

price_map = load_all_prices_daily(TICKERS, str(START_DATE), str(END_DATE))

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs:   Dict[str, pd.DataFrame] = {}
all_feat:  Dict[str, pd.DataFrame] = {}
all_diag:  Dict[str, dict] = {}

if not price_map:
    st.warning("Keine Preisdaten geladen. Prüfe Ticker/Zeitraum.")
    st.stop()

for ticker in TICKERS:
    if ticker not in price_map:
        continue

    df = price_map[ticker]
    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker} — {get_ticker_name(ticker)}")

        try:
            last_timestamp_info(df)

            feat, df_bt, trades, metrics, diag = make_features_and_train(
                df=df,
                lookback=int(LOOKBACK),
                horizon=int(HORIZON),
                target_mode=str(target_mode),
                thr_fix=float(THRESH),
                k_vol=float(k_vol),
                q_quant=float(q_quant),
                model_params=MODEL_PARAMS,
                walk_forward=bool(use_walk_forward),
                wf_min_train=int(wf_min_train),
                embargo=int(embargo_bars),
                calibrate=bool(use_calibration),
                calib_folds=int(calib_cv),
            )

            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat
            all_diag[ticker] = diag

            # Live decision (auf letzter Daily-Bar)
            def decide_action(p: float, entry_thr: float, exit_thr: float) -> str:
                if p > entry_thr:  return "Enter / Add"
                if p < exit_thr:   return "Exit / Reduce"
                return "Hold / No Trade"

            live_ts    = pd.Timestamp(feat.index[-1])
            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1])

            c1m, c2m, c3m, c4m, c5m, c6m = st.columns(6)
            c1m.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2m.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3m.metric("Sharpe",              f"{metrics['Sharpe-Ratio']:.2f}")
            c4m.metric("Sortino",             f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5m.metric("Max DD (%)",          f"{metrics['Max Drawdown (%)']:.2f}")
            c6m.metric("Aktionssignal",       decide_action(live_prob, float(ENTRY_PROB), float(EXIT_PROB)))

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("P(Signal) (letzter Tag)", f"{live_prob:.4f}")
            if diag:
                d2.metric("Brier (lower=better)", f"{diag.get('Brier', np.nan):.4f}" if np.isfinite(diag.get("Brier", np.nan)) else "–")
                d3.metric("Edge top10% (FutureRet)", f"{diag.get('Edge_top10p_minus_all', np.nan)*100:.2f}%" if np.isfinite(diag.get("Edge_top10p_minus_all", np.nan)) else "–")
                d4.metric("μ(top10%)", f"{diag.get('Mu_FutureRetExec_top10p', np.nan)*100:.2f}%" if np.isfinite(diag.get("Mu_FutureRetExec_top10p", np.nan)) else "–")
            else:
                d2.metric("Brier", "–"); d3.metric("Edge top10%", "–"); d4.metric("μ(top10%)", "–")

            st.caption(
                f"Target-Modus: {target_mode} | WF={'ON' if use_walk_forward else 'OFF'} | Embargo={int(embargo_bars)} | "
                f"Calibration={'ON' if use_calibration else 'OFF'}"
            )

            # Charts
            chart_cols = st.columns(2)

            # Price + Prob color segments
            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.35)", width=1),
                hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
            ))
            signal_probs = df_plot["SignalProb"].clip(0, 1)
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i:i+2]
                seg_y = df_plot["Close"].iloc[i:i+2]
                color_seg = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, float(norm.iloc[i]))[0]
                price_fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", showlegend=False,
                    line=dict(color=color_seg, width=2), hoverinfo="skip"
                ))
            price_fig.update_layout(
                title=f"{ticker}: Daily Preis + SignalProb",
                xaxis_title="Datum", yaxis_title="Preis",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            # Intraday chart (Anzeige only)
            with chart_cols[1]:
                if not show_intraday:
                    st.info("Intraday-Anzeige ist deaktiviert.")
                else:
                    intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
                    if intra.empty:
                        st.info("Keine Intraday-Daten verfügbar (Ticker/Intervall/Zeitraum).")
                    else:
                        intr_fig = go.Figure()
                        if intraday_chart_type == "Candlestick (OHLC)":
                            intr_fig.add_trace(go.Candlestick(
                                x=intra.index, open=intra["Open"], high=intra["High"],
                                low=intra["Low"], close=intra["Close"],
                                name="OHLC (intraday)",
                                increasing_line_width=1, decreasing_line_width=1
                            ))
                        else:
                            intr_fig.add_trace(go.Scatter(
                                x=intra.index, y=intra["Close"], mode="lines", name="Close (intraday)",
                                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra></extra>"
                            ))
                        intr_fig.update_layout(
                            title=f"{ticker}: Intraday – letzte 5 Sessions ({intraday_interval})",
                            xaxis_title="Zeit", yaxis_title="Preis",
                            height=420, margin=dict(t=50, b=30, l=40, r=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(intr_fig, use_container_width=True)

            # Equity curve
            eq = go.Figure()
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Open)",
                mode="lines", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"
            ))
            bh_curve = float(INIT_CAP_PER_TICKER) * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                line=dict(dash="dash", color="black")
            ))
            eq.update_layout(
                title=f"{ticker}: Net Equity vs. Buy & Hold",
                xaxis_title="Datum", yaxis_title="Equity (€)",
                height=380, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(eq, use_container_width=True)

            # Trades table
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                trades_df = pd.DataFrame(trades)
                if trades_df.empty:
                    st.info("Keine Trades für diesen Zeitraum / diese Parameter.")
                else:
                    td = trades_df.copy()
                    td["Ticker"] = ticker
                    td["Name"] = get_ticker_name(ticker)
                    if "Date" in td.columns:
                        td["Date"] = pd.to_datetime(td["Date"]).dt.strftime("%d.%m.%Y")
                    rename_map = {"Prob": "Signal Prob", "HoldDays": "Hold (days)", "Net P&L": "PnL", "kum P&L": "CumPnL"}
                    td = td.rename(columns={k: v for k, v in rename_map.items() if k in td.columns})
                    desired = ["Ticker","Name","Date","Typ","Price","Shares","Signal Prob","Hold (days)","PnL","CumPnL","Fees"]
                    show_cols = [c for c in desired if c in td.columns]
                    st.dataframe(td[show_cols], use_container_width=True)
                    st.download_button(
                        "Trades als CSV",
                        to_csv_eu(td[show_cols]),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key=f"dl_trades_{ticker}",
                    )

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")


# ─────────────────────────────────────────────────────────────
# Summary + Portfolio Analytics
# ─────────────────────────────────────────────────────────────
if not results:
    st.warning("Noch keine Ergebnisse verfügbar. Prüfe Ticker/Zeitraum.")
    st.stop()

summary_df = pd.DataFrame(results).set_index("Ticker")
summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / float(INIT_CAP_PER_TICKER)) * 100

total_net_pnl   = float(summary_df["Net P&L (€)"].sum())
total_fees      = float(summary_df["Fees (€)"].sum())
total_gross_pnl = total_net_pnl + total_fees
total_trades    = int(summary_df["Number of Trades"].sum())
total_capital   = float(INIT_CAP_PER_TICKER) * len(summary_df)
total_net_return_pct   = (total_net_pnl / total_capital * 100) if total_capital else np.nan
total_gross_return_pct = (total_gross_pnl / total_capital * 100) if total_capital else np.nan
bh_total_pct = float(summary_df["Buy & Hold Net (%)"].dropna().mean())

st.subheader("📊 Summary of all Tickers (Daily data, Next Open backtest)")
cols = st.columns(4)
cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
cols[3].metric("Total Number of Trades",   f"{int(total_trades)}")

cols_pct = st.columns(4)
cols_pct[0].metric("Strategy Net (%) – total",   f"{total_net_return_pct:.2f}")
cols_pct[1].metric("Strategy Gross (%) – total", f"{total_gross_return_pct:.2f}")
cols_pct[2].metric("Buy & Hold Net (%) – avg",   f"{bh_total_pct:.2f}")
cols_pct[3].metric("Avg CAGR (%)",               f"{summary_df['CAGR (%)'].dropna().mean():.2f}")

st.dataframe(summary_df, use_container_width=True)
st.download_button(
    "Summary als CSV herunterladen",
    to_csv_eu(summary_df.reset_index()),
    file_name="strategy_summary.csv", mime="text/csv"
)

# Portfolio (Equal-weight, Close-to-Close)
st.markdown("### 📈 Portfolio – Equal-Weight Performance (Close-to-Close)")

price_series = []
for tk, dfbt in all_dfs.items():
    if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
        s = dfbt["Close"].copy()
        s.name = tk
        try:
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        except Exception:
            pass
        s.index = pd.to_datetime(s.index).normalize()
        price_series.append(s)

if len(price_series) < 2:
    st.info("Portfolio-Analytics: Mindestens zwei Ticker mit Close-Daten nötig.")
    st.stop()

prices_port = pd.concat(price_series, axis=1, join="outer").sort_index()
rets = prices_port.pct_change()

valid = rets.notna().sum(axis=1) >= 2
rets2 = rets.loc[valid].copy()

if rets2.empty:
    st.info("Portfolio-Returns sind leer (zu wenig Overlap).")
    st.stop()

w_row = rets2.notna().astype(float)
w_row = w_row.div(w_row.sum(axis=1), axis=0)
port_ret = (rets2.fillna(0.0) * w_row).sum(axis=1).dropna()

ann_return = (1.0 + port_ret).prod() ** (252 / len(port_ret)) - 1.0
ann_vol = port_ret.std(ddof=0) * np.sqrt(252)
sharpe = (port_ret.mean() / (port_ret.std(ddof=0) + 1e-12)) * np.sqrt(252)

nav0 = float(INIT_CAP_PER_TICKER) * len(summary_df)
nav = nav0 * (1.0 + port_ret).cumprod()
dd = (nav / nav.cummax()) - 1.0
max_dd = float(dd.min())

c1p, c2p, c3p, c4p = st.columns(4)
c1p.metric("Ø Return p.a.", f"{ann_return*100:.2f}%")
c2p.metric("Vol p.a.", f"{ann_vol*100:.2f}%")
c3p.metric("Sharpe", f"{sharpe:.2f}")
c4p.metric("Max Drawdown", f"{max_dd*100:.2f}%")

fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(x=nav.index, y=nav.values, mode="lines", name="Portfolio NAV"))
fig_nav.update_layout(
    height=360,
    title="Portfolio NAV (Equal-Weight, Close-to-Close)",
    xaxis_title="Datum", yaxis_title="NAV (€)",
    margin=dict(t=45, b=30, l=40, r=20)
)
st.plotly_chart(fig_nav, use_container_width=True)

# Correlation
st.markdown("### 🔗 Portfolio-Korrelation (Close-Returns)")
freq_label = st.selectbox("Return-Frequenz", ["täglich", "wöchentlich", "monatlich"], index=0)
corr_method = st.selectbox("Korrelationsmethode", ["Pearson", "Spearman", "Kendall"], index=0)
min_obs = st.slider("Min. gemeinsame Zeitpunkte", min_value=10, max_value=300, value=20, step=5)
use_ffill = st.checkbox("Lücken per FFill schließen", value=True)

prices_corr = prices_port.copy()
if use_ffill:
    prices_corr = prices_corr.ffill()

if freq_label == "wöchentlich":
    prices_corr = prices_corr.resample("W").last()
elif freq_label == "monatlich":
    prices_corr = prices_corr.resample("M").last()

rets_corr = prices_corr.pct_change().dropna(how="all")
counts = rets_corr.notna().sum()
keep = counts[counts >= int(min_obs)].index.tolist()
rets_corr = rets_corr[keep]

if rets_corr.shape[1] < 2 or rets_corr.empty:
    st.warning("Zu wenig Overlap nach Filter. Zeitraum/FFill/min_obs prüfen.")
else:
    corr = rets_corr.corr(method=corr_method.lower())
    fig_corr = px.imshow(
        corr.round(2),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Korrelationsmatrix (Close-Returns)",
    )
    fig_corr.update_layout(height=520, margin=dict(t=55, b=30, l=40, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.download_button(
        "Korrelationsmatrix als CSV",
        to_csv_eu(corr.reset_index().rename(columns={"index": "Ticker"})),
        file_name="portfolio_correlation_matrix.csv",
        mime="text/csv",
    )

# Forecast (Bootstrap)
st.markdown(f"### 🔮 Portfolio-Forecast (Bootstrap, {int(FORECAST_DAYS)} Tage)")
try:
    daily_rets = prices_port[rets_corr.columns].pct_change().dropna(how="any")
    out = bootstrap_portfolio_forecast(
        rets=daily_rets,
        nav0=float(nav.iloc[-1]) if len(nav) else nav0,
        horizon_days=int(FORECAST_DAYS),
        sims=int(BOOT_SIMS),
        block=int(block_len),
        seed=42
    )

    cA, cB, cC = st.columns(3)
    cA.metric("Return 5% / 50% / 95%",
              f"{out['port_ret_q05']*100:.2f}% / {out['port_ret_q50']*100:.2f}% / {out['port_ret_q95']*100:.2f}%")
    cB.metric("NAV 5% / 50% / 95%",
              f"{out['nav_q05']:,.0f}€ / {out['nav_q50']:,.0f}€ / {out['nav_q95']:,.0f}€")
    cC.metric("Bootstrap Sims / Block", f"{int(BOOT_SIMS)} / {int(block_len)}")

    fig_fc = go.Figure(go.Histogram(x=out["port_rets"]*100, nbinsx=40, marker_line_width=0))
    fig_fc.add_vline(x=out["port_ret_q50"]*100, line_dash="dash", opacity=0.7)
    fig_fc.add_vline(x=out["port_ret_q05"]*100, line_dash="dot", opacity=0.7)
    fig_fc.add_vline(x=out["port_ret_q95"]*100, line_dash="dot", opacity=0.7)
    fig_fc.update_layout(
        title=f"Simulierte Portfolio-Returns (Bootstrap, {int(FORECAST_DAYS)} Tage)",
        xaxis_title="Return (%)",
        yaxis_title="Häufigkeit",
        height=360,
        showlegend=False,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

except Exception as e:
    st.info(f"Forecast nicht möglich: {e}")
