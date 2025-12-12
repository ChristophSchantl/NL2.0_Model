# ============================================================
# PROTEUS – OOS Signal Strategy (Per-Ticker Capital)
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Tuple, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="PROTEUS – OOS Signal Model", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")
pd.options.display.float_format = "{:,.4f}".format

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Parameter")

TICKERS = st.sidebar.text_input(
    "Ticker (comma separated)",
    value="VOW3.DE, REGN, LULU"
).upper().replace(" ", "").split(",")

START_DATE = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
END_DATE   = st.sidebar.date_input("End Date", pd.to_datetime(datetime.now().date()))

LOOKBACK = st.sidebar.slider("Lookback", 20, 120, 40, step=5)
HORIZON  = st.sidebar.slider("Horizon (days)", 2, 10, 5)
K_VOL    = st.sidebar.slider("Vol-Target k", 0.5, 2.5, 1.0, step=0.1)

Q_ENTRY  = st.sidebar.slider("Entry Quantile", 0.60, 0.95, 0.80, step=0.05)
Q_EXIT   = st.sidebar.slider("Exit Quantile", 0.10, 0.60, 0.40, step=0.05)

INIT_CAP = st.sidebar.number_input("Initial Capital (€)", 5_000, 100_000, 10_000, step=1_000)
POS_FRAC = st.sidebar.slider("Position Size", 0.25, 1.0, 1.0, step=0.25)

COMMISSION = st.sidebar.number_input("Commission (ad valorem)", 0.0, 0.01, 0.004, step=0.0005)
SLIPPAGE_BPS = st.sidebar.slider("Slippage (bps)", 0, 20, 5)

# ============================================================
# Data Loader (FIXED)
# ============================================================
@st.cache_data(ttl=600)
def load_prices(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(
        start=str(START_DATE),
        end=str(END_DATE + timedelta(days=1)),
        auto_adjust=True,
        actions=False
    )

    if df.empty:
        raise ValueError("No data")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)

    df = df.sort_index()
    df = df.dropna(subset=["Open","High","Low","Close"])
    return df

# ============================================================
# Feature Engineering
# ============================================================
def make_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    f = df.copy()

    ret = f["Close"].pct_change()
    f["ret_1d"]  = ret
    f["ret_5d"]  = f["Close"].pct_change(5)
    f["ret_20d"] = f["Close"].pct_change(20)

    f["vol_20d"] = ret.rolling(20).std()
    f["atr_pct"] = (f["High"] - f["Low"]).rolling(14).mean() / f["Close"]

    f["range_lb"] = (
        f["High"].rolling(lookback).max()
        - f["Low"].rolling(lookback).min()
    ) / f["Close"]

    f["ma_fast"] = f["Close"].rolling(10).mean()
    f["ma_slow"] = f["Close"].rolling(50).mean()
    f["trend"]   = f["ma_fast"] / f["ma_slow"] - 1

    f["dd_20d"] = f["Close"] / f["Close"].rolling(20).max() - 1

    return f.dropna()

# ============================================================
# Target (vol-adaptive)
# ============================================================
def make_target(df: pd.DataFrame, horizon: int, k: float) -> pd.Series:
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    sigma   = df["ret_1d"].rolling(20).std() * np.sqrt(horizon)
    return (fwd_ret > k * sigma).astype(int)

# ============================================================
# OOS Probabilities (Purged TSS)
# ============================================================
def oos_probabilities(
    df: pd.DataFrame,
    features: list,
    target: pd.Series,
    horizon: int
) -> pd.Series:

    X = df[features]
    y = target
    probs = pd.Series(index=df.index, dtype=float)

    tscv = TimeSeriesSplit(n_splits=5)

    for tr_idx, te_idx in tscv.split(X):
        tr_idx = tr_idx[tr_idx < te_idx[0] - horizon]
        if len(tr_idx) < 60:
            continue

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te       = X.iloc[te_idx]

        if y_tr.nunique() < 2:
            continue

        base = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X_tr, y_tr)

        probs.iloc[te_idx] = clf.predict_proba(X_te)[:, 1]

    return probs.ffill().fillna(0.5)

# ============================================================
# Backtest (Next Open, per-Ticker Capital)
# ============================================================
def backtest(
    df: pd.DataFrame,
    entry_thr: float,
    exit_thr: float
) -> Tuple[pd.DataFrame, list]:

    cash = INIT_CAP
    shares = 0.0
    in_pos = False
    trades = []

    equity = []

    for i in range(1, len(df)):
        prob_prev = df["SignalProb"].iloc[i-1]
        open_px = df["Open"].iloc[i]

        buy_px  = open_px * (1 + SLIPPAGE_BPS/10_000)
        sell_px = open_px * (1 - SLIPPAGE_BPS/10_000)

        # Entry
        if (not in_pos) and prob_prev >= entry_thr:
            invest = cash * POS_FRAC
            fee = invest * COMMISSION
            shares = (invest - fee) / buy_px
            cash -= invest
            in_pos = True
            trades.append({"Date": df.index[i], "Type": "Entry", "Price": buy_px})

        # Exit
        elif in_pos and prob_prev <= exit_thr:
            gross = shares * sell_px
            fee = gross * COMMISSION
            cash += gross - fee
            shares = 0.0
            in_pos = False
            trades.append({"Date": df.index[i], "Type": "Exit", "Price": sell_px})

        equity.append(cash + shares * df["Close"].iloc[i])

    df_bt = df.iloc[1:].copy()
    df_bt["Equity"] = equity
    return df_bt, trades

# ============================================================
# Main
# ============================================================
st.title("📈 PROTEUS – OOS Signal Strategy")

FEATURES = [
    "ret_1d","ret_5d","ret_20d",
    "vol_20d","atr_pct","range_lb",
    "trend","dd_20d"
]

for ticker in TICKERS:
    st.subheader(ticker)

    try:
        df = load_prices(ticker)
        feat = make_features(df, LOOKBACK)
        target = make_target(feat, HORIZON, K_VOL)

        feat["SignalProb"] = oos_probabilities(
            feat, FEATURES, target, HORIZON
        )

        entry_thr = feat["SignalProb"].quantile(Q_ENTRY)
        exit_thr  = feat["SignalProb"].quantile(Q_EXIT)

        df_bt, trades = backtest(feat, entry_thr, exit_thr)

        ret = df_bt["Equity"].iloc[-1] / INIT_CAP - 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Net (%)", f"{ret*100:.2f}")
        c2.metric("Trades", len(trades)//2)
        c3.metric("Entry / Exit", f"{entry_thr:.2f} / {exit_thr:.2f}")

        st.line_chart(df_bt[["Equity"]])

    except Exception as e:
        st.error(f"{ticker}: {e}")
