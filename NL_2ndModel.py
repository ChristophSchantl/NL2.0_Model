from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG + DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NEXUS Maison v3 — Signal Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOCAL_TZ = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 8
TRADING_DAYS = 252
pd.options.display.float_format = "{:,.4f}".format

GOLD = "#B69D5F"
GOLD_DEEP = "#9A8243"
GOLD_PALE = "#EDE4CC"
STONE = "#F5F1EB"
INK = "#1E1E1E"
INK_MID = "#6E6E6E"
INK_LIGHT = "#9C9C9C"
RISE = "#4D7C5B"
FALL = "#944848"
WHITE = "#FFFFFF"
DEPTH = "#5B6B8A"

BASE_FEATURE_COLS = [
    "Range",
    "SlopeHigh",
    "SlopeLow",
    "Ret_5d",
    "Ret_20d",
    "MA_ratio",
    "Volatility",
    "RSI",
    "Vol_ratio",
]

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Outfit:wght@300;400;500;600&display=swap');

.stApp {{ background: {STONE}; }}
[data-testid="stSidebar"] {{ background: {WHITE}; border-right: 1px solid #DDD8CE; }}
[data-testid="stSidebar"] * {{ font-family: 'Outfit', sans-serif; }}

h1, h2, h3 {{ font-family: 'Cormorant Garamond', Georgia, serif !important; color: {INK}; }}
p, span, div, label {{ font-family: 'Outfit', sans-serif; color: {INK}; }}

[data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid #DDD8CE;
    border-radius: 14px;
    padding: 16px 18px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 10px !important;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {INK_LIGHT} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 28px !important;
    color: {INK} !important;
    font-weight: 400;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid #DDD8CE;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    color: {INK_LIGHT};
    padding: 10px 22px;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD};
}}

.stButton > button, .stDownloadButton > button {{
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: linear-gradient(135deg, {GOLD}, #D4B96A);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
}}

[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

.gold-rule {{
    height: 1px;
    background: linear-gradient(90deg, {GOLD}, {GOLD_PALE}, transparent);
    margin: 12px 0 24px 0;
}}
.section-header {{
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    color: {GOLD};
    margin-bottom: 8px;
}}
.hero-name {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 36px;
    font-weight: 400;
    color: {INK};
    letter-spacing: -0.03em;
    line-height: 1.05;
}}
.hero-sub {{
    font-size: 12px;
    font-weight: 600;
    color: {GOLD};
    letter-spacing: 0.16em;
    text-transform: uppercase;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def fmt_pct(v: Optional[float], scale: float = 100.0, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v * scale:.{digits}f}%"


def fmt_num(v: Optional[float], digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.{digits}f}"


def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(
        index=False,
        sep=";",
        decimal=",",
        date_format="%d.%m.%Y",
        float_format=float_format,
    ).encode("utf-8-sig")


def normalize_tickers(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    for x in items or []:
        if isinstance(x, str):
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
            return normalize_tickers(df[cols_lower[key]].astype(str).tolist())
    return normalize_tickers(df.iloc[:, 0].astype(str).tolist())


def cagr_from_equity(equity: pd.Series) -> float:
    equity = pd.Series(equity).dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return np.nan
    total_years = max((len(equity) - 1) / TRADING_DAYS, 1 / TRADING_DAYS)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / total_years) - 1)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    eq = pd.Series(equity).astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def sharpe_from_returns(ret: pd.Series) -> float:
    r = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2 or np.isclose(r.std(ddof=0), 0.0):
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=0))


def rolling_slope_vec(series: pd.Series, window: int) -> pd.Series:
    vals = series.to_numpy(dtype=np.float64)
    if len(vals) < window:
        return pd.Series(np.full(len(vals), np.nan), index=series.index)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    weights = (x - xm) / denom
    wins = sliding_window_view(vals, window_shape=window)
    ym = wins.mean(axis=1, keepdims=True)
    slopes = ((wins - ym) * weights).sum(axis=1)
    out = np.full(len(vals), np.nan, dtype=np.float64)
    out[window - 1:] = slopes
    return pd.Series(out, index=series.index)


def composite_score(sharpe: float, winrate: float, cagr: float, max_dd: float, w_dd: float) -> float:
    if not (np.isfinite(sharpe) and np.isfinite(winrate) and np.isfinite(cagr) and np.isfinite(max_dd)):
        return float("-inf")
    wr_factor = winrate * 2.0
    cagr_bonus = 1.0 + max(cagr, -1.0)
    dd_penalty = w_dd * abs(max_dd)
    return float(sharpe * wr_factor * cagr_bonus - dd_penalty)


def vol_scaled_frac(realized_vol_ann: float, base_frac: float, target_vol_ann: float) -> float:
    """Single-position volatility scaling; this is NOT portfolio-level vol targeting."""
    if not np.isfinite(realized_vol_ann) or realized_vol_ann <= 0:
        return float(base_frac)
    scaled = base_frac * (target_vol_ann / realized_vol_ann)
    return float(np.clip(scaled, 0.10 * base_frac, 1.50 * base_frac))


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=180)
def get_price_data_tail_intraday(
    ticker: str,
    years: int = 3,
    use_tail: bool = True,
    interval: str = "5m",
    fallback_last_session: bool = False,
    exec_mode_key: str = "Next Open (backtest+live)",
    moc_cutoff_min_val: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)
    df = df.sort_index().drop_duplicates()
    meta = {"tail_is_intraday": False, "tail_ts": None}

    if not use_tail:
        return df.dropna(subset=["Open", "High", "Low", "Close"]), meta

    try:
        intr = tk.history(period="1d", interval=interval, auto_adjust=True, actions=False, prepost=False)
        if not intr.empty:
            if intr.index.tz is None:
                intr.index = intr.index.tz_localize("UTC")
            intr.index = intr.index.tz_convert(LOCAL_TZ).sort_index()
    except Exception:
        intr = pd.DataFrame()

    if exec_mode_key.startswith("Market-On-Close") and not intr.empty:
        cutoff_time = datetime.now(LOCAL_TZ) - timedelta(minutes=int(moc_cutoff_min_val))
        intr = intr.loc[:cutoff_time]

    if intr.empty and fallback_last_session:
        try:
            intr5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
            if not intr5.empty:
                if intr5.index.tz is None:
                    intr5.index = intr5.index.tz_localize("UTC")
                intr5.index = intr5.index.tz_convert(LOCAL_TZ).sort_index()
                last_session_date = intr5.index[-1].date()
                intr = intr5.loc[str(last_session_date)]
        except Exception:
            intr = pd.DataFrame()

    if not intr.empty:
        last_bar = intr.iloc[-1]
        day_key = pd.Timestamp(last_bar.name.date(), tz=LOCAL_TZ)
        df.loc[day_key] = {
            "Open": float(intr["Open"].iloc[0]),
            "High": float(intr["High"].max()),
            "Low": float(intr["Low"].min()),
            "Close": float(last_bar["Close"]),
            "Volume": float(intr["Volume"].sum()),
        }
        df = df.sort_index()
        meta = {"tail_is_intraday": True, "tail_ts": last_bar.name}

    return df.dropna(subset=["Open", "High", "Low", "Close"]), meta


def load_all_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_tail: bool,
    interval: str,
    fallback_last: bool,
    exec_key: str,
    moc_cutoff: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not tickers:
        return price_map, meta_map

    st.info(f"Kurse laden für {len(tickers)} Ticker …")
    prog = st.progress(0.0)
    total = len(tickers)
    done = 0

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as ex:
        futures = {
            ex.submit(
                get_price_data_tail_intraday,
                tk,
                3,
                use_tail,
                interval,
                fallback_last,
                exec_key,
                int(moc_cutoff),
            ): tk
            for tk in tickers
        }
        for fut in as_completed(futures):
            tk = futures[fut]
            try:
                df_full, meta = fut.result()
                price_map[tk] = df_full.loc[str(start):str(end)].copy()
                meta_map[tk] = meta
            except Exception as e:
                st.error(f"Fehler beim Laden von {tk}: {e}")
            finally:
                done += 1
                prog.progress(done / total)
    return price_map, meta_map


# ═══════════════════════════════════════════════════════════════
# FEATURES + MODEL
# ═══════════════════════════════════════════════════════════════

def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"] = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = rolling_slope_vec(feat["High"], lookback)
    feat["SlopeLow"] = rolling_slope_vec(feat["Low"], lookback)
    feat["Ret_5d"] = feat["Close"].pct_change(5)
    feat["Ret_20d"] = feat["Close"].pct_change(20)
    feat["MA_ratio"] = feat["Close"] / feat["Close"].rolling(20).mean()
    feat["Volatility"] = feat["Close"].pct_change().rolling(lookback).std()

    delta = feat["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat["RSI"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    if "Volume" in feat.columns and feat["Volume"].gt(0).any():
        vol_ma = feat["Volume"].rolling(20).mean().replace(0, np.nan)
        feat["Vol_ratio"] = feat["Volume"] / vol_ma
    else:
        feat["Vol_ratio"] = 1.0

    feat = feat.iloc[lookback - 1 :].copy()
    feat["FutureRet"] = feat["Close"].shift(-horizon) / feat["Close"] - 1.0
    return feat


@dataclass
class TrainedModelBundle:
    model: HistGradientBoostingClassifier
    calibrator: Optional[LogisticRegression]
    x_cols: List[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_p = self.model.predict_proba(X)[:, 1]
        if self.calibrator is None:
            return base_p
        x_cal = np.log(np.clip(base_p, 1e-6, 1 - 1e-6) / np.clip(1 - base_p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return self.calibrator.predict_proba(x_cal)[:, 1]


def build_hgb_model(model_params: Dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(model_params["learning_rate"]),
        max_depth=int(model_params["max_depth"]),
        max_iter=int(model_params["max_iter"]),
        min_samples_leaf=int(model_params.get("min_samples_leaf", 20)),
        l2_regularization=float(model_params.get("l2_regularization", 0.0)),
        random_state=42,
    )


def fit_model_with_calibration(
    X: np.ndarray,
    y: np.ndarray,
    model_params: Dict[str, Any],
    use_calibration: bool,
    calibration_frac: float,
) -> Optional[TrainedModelBundle]:
    if len(X) < 40 or len(np.unique(y)) < 2:
        return None

    if not use_calibration or len(X) < 80:
        model = build_hgb_model(model_params)
        model.fit(X, y)
        return TrainedModelBundle(model=model, calibrator=None, x_cols=BASE_FEATURE_COLS)

    split = int(len(X) * (1.0 - calibration_frac))
    split = max(30, min(split, len(X) - 20))
    X_fit, y_fit = X[:split], y[:split]
    X_calib, y_calib = X[split:], y[split:]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_calib)) < 2:
        model = build_hgb_model(model_params)
        model.fit(X, y)
        return TrainedModelBundle(model=model, calibrator=None, x_cols=BASE_FEATURE_COLS)

    model = build_hgb_model(model_params)
    model.fit(X_fit, y_fit)

    p_cal = model.predict_proba(X_calib)[:, 1]
    x_platt = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / np.clip(1 - p_cal, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(x_platt, y_calib)
    return TrainedModelBundle(model=model, calibrator=calibrator, x_cols=BASE_FEATURE_COLS)


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def make_features_and_backtest(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: Dict[str, Any],
    entry_prob: float,
    exit_prob: float,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
    walk_forward: bool = True,
    use_vol_sizing: bool = False,
    target_vol_annual: float = 0.15,
    base_pos_frac: float = 1.0,
    commission: float = 0.0,
    slippage_bps: float = 0.0,
    init_cap: float = 10_000.0,
    wf_stride: int = 5,
    use_calibration: bool = True,
    calibration_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:
    feat = make_features(df, lookback, horizon)
    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < 60:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")

    x_cols = [c for c in BASE_FEATURE_COLS if c in hist.columns]
    X_all = feat[x_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    future_ret = feat["FutureRet"].to_numpy(dtype=np.float32)
    probs = np.full(len(feat), np.nan, dtype=np.float32)

    if not walk_forward:
        train_mask = np.isfinite(future_ret)
        X_train = X_all[train_mask]
        y_train = (future_ret[train_mask] > threshold).astype(np.int8)
        bundle = fit_model_with_calibration(X_train, y_train, model_params, use_calibration, calibration_frac)
        if bundle is None:
            probs[:] = 0.5
        else:
            probs[:] = bundle.predict_proba(X_all).astype(np.float32)
    else:
        min_train = max(lookback + horizon + 20, 80)
        t = min_train
        while t < len(feat):
            train_end = t
            train_mask = np.isfinite(future_ret[:train_end])
            if train_mask.sum() >= min_train:
                X_train = X_all[:train_end][train_mask]
                y_train = (future_ret[:train_end][train_mask] > threshold).astype(np.int8)
                bundle = fit_model_with_calibration(X_train, y_train, model_params, use_calibration, calibration_frac)
                pred_end = min(t + int(wf_stride), len(feat))
                if bundle is None:
                    probs[t:pred_end] = 0.5
                else:
                    probs[t:pred_end] = bundle.predict_proba(X_all[t:pred_end]).astype(np.float32)
            t += int(wf_stride)
        probs[:min_train] = np.nan
        probs = pd.Series(probs, index=feat.index).ffill().fillna(0.5).to_numpy(dtype=np.float32)

    feat["SignalProb"] = probs

    bt = feat[["Open", "High", "Low", "Close", "SignalProb", "Volatility"]].copy()
    bt["MarketRet"] = bt["Close"].pct_change().fillna(0.0)
    bt["Target_5d"] = feat["FutureRet"]

    position = 0
    shares = 0.0
    cash = float(init_cap)
    last_exit_idx = -10_000
    hold_days = 0
    trades: List[dict] = []
    pending_entry = False
    pending_exit = False
    entry_px = np.nan
    entry_dt = None
    pos_frac_t = float(base_pos_frac)

    equity_gross = []
    equity_net = []
    pos_list = []
    act_list = []
    action_adj_list = []

    slip = float(slippage_bps) / 10_000.0

    for i in range(len(bt)):
        row = bt.iloc[i]
        open_px = float(row["Open"])
        close_px = float(row["Close"])
        prob = float(row["SignalProb"])
        realized_vol_ann = float(row["Volatility"] * np.sqrt(TRADING_DAYS)) if np.isfinite(row["Volatility"]) else np.nan

        action = "Hold"
        action_adj = "Hold"

        if pending_exit and position == 1:
            exec_px = open_px * (1.0 - slip)
            proceeds = shares * exec_px
            fee = proceeds * commission
            cash += proceeds - fee
            pnl_pct = (exec_px / entry_px - 1.0) if entry_px > 0 else np.nan
            trades.append(
                {
                    "Typ": "Exit",
                    "Date": bt.index[i],
                    "Price": exec_px,
                    "PnL_%": pnl_pct * 100.0 if np.isfinite(pnl_pct) else np.nan,
                    "Held_Days": hold_days,
                    "Probability": prob,
                }
            )
            shares = 0.0
            position = 0
            last_exit_idx = i
            hold_days = 0
            pending_exit = False
            action_adj = "Exit @ Next Open"

        if pending_entry and position == 0:
            if i - last_exit_idx > cooldown_days:
                pos_frac_t = float(base_pos_frac)
                if use_vol_sizing:
                    pos_frac_t = vol_scaled_frac(realized_vol_ann, float(base_pos_frac), float(target_vol_annual))
                exec_px = open_px * (1.0 + slip)
                alloc = cash * pos_frac_t
                fee = alloc * commission
                effective_alloc = max(alloc - fee, 0.0)
                shares = effective_alloc / exec_px if exec_px > 0 else 0.0
                cash -= alloc
                entry_px = exec_px
                entry_dt = bt.index[i]
                position = 1
                hold_days = 0
                trades.append(
                    {
                        "Typ": "Entry",
                        "Date": bt.index[i],
                        "Price": exec_px,
                        "PnL_%": np.nan,
                        "Held_Days": 0,
                        "Probability": prob,
                        "PosFrac": pos_frac_t,
                    }
                )
                action_adj = "Enter @ Next Open"
            pending_entry = False

        if position == 1:
            hold_days += 1
            if prob <= exit_prob and hold_days >= min_hold_days:
                pending_exit = True
                action = "Exit"
            else:
                action = "Hold Long"
        else:
            can_enter = (i - last_exit_idx) > cooldown_days
            if prob >= entry_prob and can_enter:
                pending_entry = True
                action = "Enter"
            else:
                action = "Hold Cash"

        gross_eq = cash + shares * close_px
        net_eq = gross_eq
        equity_gross.append(gross_eq)
        equity_net.append(net_eq)
        pos_list.append(position)
        act_list.append(action)
        action_adj_list.append(action_adj)

    bt["Position"] = pos_list
    bt["Action"] = act_list
    bt["Action_adj"] = action_adj_list
    bt["Equity_Gross"] = np.array(equity_gross, dtype=np.float64)
    bt["Equity_Net"] = np.array(equity_net, dtype=np.float64)
    bt["StrategyRet"] = bt["Equity_Net"].pct_change().fillna(0.0)

    trade_cols = ["Typ", "Date", "Price", "PnL_%", "Held_Days", "Probability", "PosFrac"]
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=trade_cols)
    else:
        for col in trade_cols:
            if col not in trades_df.columns:
                trades_df[col] = np.nan
        trades_df = trades_df[trade_cols]

    bh_ret = bt["Close"].iloc[-1] / bt["Close"].iloc[0] - 1.0
    net_ret = bt["Equity_Net"].iloc[-1] / bt["Equity_Net"].iloc[0] - 1.0
    max_dd = max_drawdown_from_equity(bt["Equity_Net"])
    cagr = cagr_from_equity(bt["Equity_Net"])
    sharpe = sharpe_from_returns(bt["StrategyRet"])

    closed_trades = trades_df.loc[trades_df["Typ"].eq("Exit")].copy()
    winrate = float((closed_trades["PnL_%"] > 0).mean()) if len(closed_trades) else np.nan

    summary = {
        "Net (%)": net_ret * 100.0,
        "CAGR (%)": cagr * 100.0 if np.isfinite(cagr) else np.nan,
        "Sharpe": sharpe,
        "Max DD (%)": max_dd * 100.0 if np.isfinite(max_dd) else np.nan,
        "Buy & Hold (%)": bh_ret * 100.0,
        "Closed Trades": int(len(closed_trades)),
        "Win Rate (%)": winrate * 100.0 if np.isfinite(winrate) else np.nan,
        "Last Signal Probability": float(bt["SignalProb"].iloc[-1]),
        "Next Action": bt["Action"].iloc[-1],
    }
    return feat, bt, trades, summary


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER — TWO STAGE RANDOM SEARCH
# ═══════════════════════════════════════════════════════════════

def sample_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "lookback": rng.choice([20, 25, 30, 35, 40, 50, 60]),
        "horizon": rng.choice([3, 4, 5, 7, 10]),
        "threshold": round(rng.uniform(0.01, 0.08), 3),
        "entry_prob": round(rng.uniform(0.55, 0.72), 2),
        "exit_prob": round(rng.uniform(0.35, 0.54), 2),
        "max_iter": rng.choice([80, 100, 120, 150, 180, 220]),
        "learning_rate": round(rng.uniform(0.03, 0.15), 3),
        "max_depth": rng.choice([2, 3, 4, 5, 6]),
        "min_samples_leaf": rng.choice([10, 15, 20, 25, 30]),
        "l2_regularization": round(rng.uniform(0.0, 1.5), 2),
        "wf_stride": rng.choice([3, 5, 7, 10]),
    }


def jitter_params(base: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    p = dict(base)
    p["lookback"] = int(np.clip(base["lookback"] + rng.choice([-10, -5, 0, 5, 10]), 15, 80))
    p["horizon"] = int(np.clip(base["horizon"] + rng.choice([-2, -1, 0, 1, 2]), 2, 12))
    p["threshold"] = round(float(np.clip(base["threshold"] + rng.uniform(-0.01, 0.01), 0.0, 0.10)), 3)
    p["entry_prob"] = round(float(np.clip(base["entry_prob"] + rng.uniform(-0.04, 0.04), 0.50, 0.80)), 2)
    p["exit_prob"] = round(float(np.clip(base["exit_prob"] + rng.uniform(-0.04, 0.04), 0.20, 0.65)), 2)
    if p["exit_prob"] >= p["entry_prob"]:
        p["exit_prob"] = round(max(0.20, p["entry_prob"] - 0.05), 2)
    p["max_iter"] = int(np.clip(base["max_iter"] + rng.choice([-40, -20, 0, 20, 40]), 60, 260))
    p["learning_rate"] = round(float(np.clip(base["learning_rate"] + rng.uniform(-0.03, 0.03), 0.02, 0.20)), 3)
    p["max_depth"] = int(np.clip(base["max_depth"] + rng.choice([-1, 0, 1]), 2, 7))
    p["min_samples_leaf"] = int(np.clip(base["min_samples_leaf"] + rng.choice([-10, -5, 0, 5, 10]), 5, 40))
    p["l2_regularization"] = round(float(np.clip(base["l2_regularization"] + rng.uniform(-0.4, 0.4), 0.0, 2.5)), 2)
    p["wf_stride"] = int(np.clip(base["wf_stride"] + rng.choice([-2, 0, 2]), 2, 12))
    return p


def evaluate_param_set(
    params: Dict[str, Any],
    price_map: Dict[str, pd.DataFrame],
    tickers: List[str],
    base_settings: Dict[str, Any],
    w_dd: float,
    stage_name: str,
) -> Dict[str, Any]:
    scores: List[float] = []
    diagnostics: List[Dict[str, Any]] = []

    for tk in tickers:
        df = price_map.get(tk)
        if df is None or len(df) < 150:
            continue

        split_idx = len(df) // 2
        halves = [df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()]
        for part in halves:
            if len(part) < max(params["lookback"] + params["horizon"] + 80, 120):
                continue
            try:
                _, bt, trades, summary = make_features_and_backtest(
                    df=part,
                    lookback=params["lookback"],
                    horizon=params["horizon"],
                    threshold=params["threshold"],
                    model_params={
                        "max_iter": params["max_iter"],
                        "learning_rate": params["learning_rate"],
                        "max_depth": params["max_depth"],
                        "min_samples_leaf": params["min_samples_leaf"],
                        "l2_regularization": params["l2_regularization"],
                    },
                    entry_prob=params["entry_prob"],
                    exit_prob=params["exit_prob"],
                    min_hold_days=base_settings["min_hold_days"],
                    cooldown_days=base_settings["cooldown_days"],
                    walk_forward=True,
                    use_vol_sizing=base_settings["use_vol_sizing"],
                    target_vol_annual=base_settings["target_vol_annual"],
                    base_pos_frac=base_settings["pos_frac"],
                    commission=base_settings["commission"],
                    slippage_bps=base_settings["slippage_bps"],
                    init_cap=base_settings["init_cap"],
                    wf_stride=params["wf_stride"],
                    use_calibration=base_settings["use_calibration"],
                    calibration_frac=base_settings["calibration_frac"],
                )
                metric = composite_score(
                    sharpe=summary["Sharpe"],
                    winrate=(summary["Win Rate (%)"] / 100.0) if np.isfinite(summary["Win Rate (%)"]) else np.nan,
                    cagr=(summary["CAGR (%)"] / 100.0) if np.isfinite(summary["CAGR (%)"]) else np.nan,
                    max_dd=(summary["Max DD (%)"] / 100.0) if np.isfinite(summary["Max DD (%)"]) else np.nan,
                    w_dd=w_dd,
                )
                scores.append(metric)
                diagnostics.append(
                    {
                        "Ticker": tk,
                        "Stage": stage_name,
                        "Score": metric,
                        "Sharpe": summary["Sharpe"],
                        "Win Rate (%)": summary["Win Rate (%)"],
                        "CAGR (%)": summary["CAGR (%)"],
                        "Max DD (%)": summary["Max DD (%)"],
                        "Closed Trades": summary["Closed Trades"],
                    }
                )
            except Exception:
                continue

    overall = float(np.median(scores)) if scores else float("-inf")
    return {"params": params, "score": overall, "details": diagnostics}


def run_two_stage_optimizer(
    price_map: Dict[str, pd.DataFrame],
    tickers: List[str],
    coarse_trials: int,
    fine_trials: int,
    top_k: int,
    seed: int,
    base_settings: Dict[str, Any],
    w_dd: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = random.Random(seed)
    eligible = [tk for tk in tickers if tk in price_map and len(price_map[tk]) >= 160]
    if not eligible:
        return pd.DataFrame(), {}

    stage1_tickers = eligible[: max(2, min(len(eligible), max(3, len(eligible) // 2)))]
    stage2_tickers = eligible

    results: List[Dict[str, Any]] = []
    for _ in range(int(coarse_trials)):
        params = sample_params(rng)
        results.append(evaluate_param_set(params, price_map, stage1_tickers, base_settings, w_dd, "coarse"))

    coarse_df = pd.DataFrame(
        [{**r["params"], "score": r["score"], "stage": "coarse"} for r in results]
    ).sort_values("score", ascending=False)
    top_params = coarse_df.head(int(top_k)).to_dict("records")

    fine_results: List[Dict[str, Any]] = []
    if top_params:
        for _ in range(int(fine_trials)):
            parent = rng.choice(top_params)
            params = jitter_params(parent, rng)
            fine_results.append(evaluate_param_set(params, price_map, stage2_tickers, base_settings, w_dd, "fine"))

    fine_df = pd.DataFrame(
        [{**r["params"], "score": r["score"], "stage": "fine"} for r in fine_results]
    )
    out = pd.concat([coarse_df, fine_df], ignore_index=True) if not fine_df.empty else coarse_df.copy()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    best = out.iloc[0].to_dict() if not out.empty else {}
    return out, best


# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    font_family="Outfit, sans-serif",
    paper_bgcolor=STONE,
    plot_bgcolor=STONE,
    margin=dict(l=0, r=0, t=35, b=0),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(color=INK_LIGHT, size=10)),
    yaxis=dict(showgrid=True, gridcolor="#DDD8CE", gridwidth=1, zeroline=False, showline=False, tickfont=dict(color=INK_LIGHT, size=10)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11, color=INK_MID)),
)


def equity_chart(bt: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Equity_Net"], mode="lines", name="Strategy Equity", line=dict(color=GOLD, width=2.5)))
    bh = bt["Close"] / bt["Close"].iloc[0] * bt["Equity_Net"].iloc[0]
    fig.add_trace(go.Scatter(x=bt.index, y=bh, mode="lines", name="Buy & Hold", line=dict(color=DEPTH, width=1.8, dash="dot")))
    fig.update_layout(title=f"{ticker} — Equity Curve", **PLOT_LAYOUT)
    return fig


def prob_chart(bt: pd.DataFrame, entry_prob: float, exit_prob: float, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["SignalProb"], mode="lines", name="Signal Probability", line=dict(color=GOLD_DEEP, width=2)))
    fig.add_hline(y=entry_prob, line_dash="dash", line_color=RISE, annotation_text="Entry")
    fig.add_hline(y=exit_prob, line_dash="dash", line_color=FALL, annotation_text="Exit")
    fig.update_layout(title=f"{ticker} — Calibrated Signal Probability", yaxis_range=[0, 1], **PLOT_LAYOUT)
    return fig


def summary_bar(summary_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(summary_df.sort_values("Net (%)"), x="Net (%)", y="Ticker", orientation="h")
    fig.update_traces(marker_color=GOLD, hovertemplate="%{y}: %{x:.2f}%<extra></extra>")
    fig.update_layout(title="Net Return by Ticker", **PLOT_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown('<div class="hero-name">NEXUS Maison v3</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Faster Walk-Forward · HistGradientBoosting · Probability Calibration</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Strategy Setup")

    ticker_source = st.sidebar.selectbox("Ticker Source", ["Manual", "CSV Upload"], index=0)
    tickers: List[str] = []
    if ticker_source == "Manual":
        tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value="REGN, LULU, VOW3.DE, REI, DDL")
        tickers = normalize_tickers([t for t in tickers_input.split(",") if t.strip()])
    else:
        uploads = st.sidebar.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)
        collected: List[str] = []
        if uploads:
            for up in uploads:
                try:
                    collected += parse_ticker_csv(up)
                except Exception as e:
                    st.sidebar.error(f"CSV read error in {up.name}: {e}")
        extras = st.sidebar.text_input("Additional tickers", value="")
        tickers = normalize_tickers(collected + [t for t in extras.split(",") if t.strip()])

    if not tickers:
        tickers = ["REGN", "LULU", "VOW3.DE"]

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

    lookback = int(st.sidebar.number_input("Lookback", min_value=10, max_value=252, value=35, step=5))
    horizon = int(st.sidebar.number_input("Horizon", min_value=1, max_value=15, value=5, step=1))
    threshold = float(st.sidebar.number_input("Target Threshold", min_value=0.0, max_value=0.15, value=0.046, step=0.005, format="%.3f"))
    entry_prob = float(st.sidebar.slider("Entry Threshold", 0.0, 1.0, 0.62, step=0.01))
    exit_prob = float(st.sidebar.slider("Exit Threshold", 0.0, 1.0, 0.48, step=0.01))
    if exit_prob >= entry_prob:
        st.sidebar.error("Exit Threshold must be below Entry Threshold.")
        st.stop()

    min_hold_days = int(st.sidebar.number_input("Minimum Hold Days", 0, 252, 5, step=1))
    cooldown_days = int(st.sidebar.number_input("Cooldown Days", 0, 252, 0, step=1))
    commission = float(st.sidebar.number_input("Commission", 0.0, 0.02, 0.004, step=0.0001, format="%.4f"))
    slippage_bps = float(st.sidebar.number_input("Slippage (bp)", 0, 50, 5, step=1))
    pos_frac = float(st.sidebar.slider("Position Size", 0.1, 1.0, 1.0, step=0.1))
    init_cap = float(st.sidebar.number_input("Initial Capital (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f"))

    st.sidebar.markdown("**Execution / Data**")
    use_live = st.sidebar.checkbox("Use intraday tail for latest day", value=True)
    intraday_interval = st.sidebar.selectbox("Intraday Interval", ["1m", "2m", "5m", "15m"], index=2)
    fallback_last_session = st.sidebar.checkbox("Fallback to last session", value=False)
    exec_mode = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
    moc_cutoff_min = int(st.sidebar.number_input("MOC cutoff (minutes before close)", 5, 60, 15, step=5))

    st.sidebar.markdown("**Model — HistGradientBoosting**")
    walk_forward = st.sidebar.checkbox("Walk-Forward", value=True)
    max_iter = int(st.sidebar.number_input("max_iter", 40, 400, 120, step=10))
    learning_rate = float(st.sidebar.number_input("learning_rate", 0.01, 0.30, 0.08, step=0.01, format="%.2f"))
    max_depth = int(st.sidebar.number_input("max_depth", 2, 10, 4, step=1))
    min_samples_leaf = int(st.sidebar.number_input("min_samples_leaf", 5, 50, 20, step=5))
    l2_regularization = float(st.sidebar.number_input("l2_regularization", 0.0, 5.0, 0.3, step=0.1))
    wf_stride = int(st.sidebar.number_input("Walk-Forward Stride", 1, 30, 5, step=1))

    st.sidebar.markdown("**Probability Calibration**")
    use_calibration = st.sidebar.checkbox("Enable calibration", value=True)
    calibration_frac = float(st.sidebar.slider("Calibration split", 0.10, 0.35, 0.20, step=0.05))

    st.sidebar.markdown("**Position Sizing**")
    use_vol_sizing = st.sidebar.checkbox("Volatility-scaled single-position sizing", value=False)
    target_vol_annual = float(
        st.sidebar.number_input(
            "Target annualized volatility (%)",
            min_value=5.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="This scales each individual position by its own realized volatility. It is not portfolio-level volatility targeting.",
        )
        / 100.0
    )

    st.sidebar.markdown("**Optimizer**")
    enable_optimizer = st.sidebar.checkbox("Run two-stage optimizer", value=False)
    coarse_trials = int(st.sidebar.number_input("Coarse trials", 5, 300, 40, step=5))
    fine_trials = int(st.sidebar.number_input("Fine trials", 5, 300, 20, step=5))
    top_k = int(st.sidebar.number_input("Top-k seeds", 1, 20, 5, step=1))
    w_dd = float(st.sidebar.number_input("Drawdown weight", 0.0, 10.0, 1.5, step=0.1))
    opt_seed = int(st.sidebar.number_input("Optimizer seed", 0, 10000, 42, step=1))

    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()

    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "lookback": lookback,
        "horizon": horizon,
        "threshold": threshold,
        "entry_prob": entry_prob,
        "exit_prob": exit_prob,
        "min_hold_days": min_hold_days,
        "cooldown_days": cooldown_days,
        "commission": commission,
        "slippage_bps": slippage_bps,
        "pos_frac": pos_frac,
        "init_cap": init_cap,
        "use_live": use_live,
        "intraday_interval": intraday_interval,
        "fallback_last_session": fallback_last_session,
        "exec_mode": exec_mode,
        "moc_cutoff_min": moc_cutoff_min,
        "walk_forward": walk_forward,
        "model_params": {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        },
        "wf_stride": wf_stride,
        "use_calibration": use_calibration,
        "calibration_frac": calibration_frac,
        "use_vol_sizing": use_vol_sizing,
        "target_vol_annual": target_vol_annual,
        "enable_optimizer": enable_optimizer,
        "coarse_trials": coarse_trials,
        "fine_trials": fine_trials,
        "top_k": top_k,
        "w_dd": w_dd,
        "opt_seed": opt_seed,
    }


def style_live_board(df: pd.DataFrame, prob_col: str, entry_threshold: float):
    def row_color(row):
        act = str(row.get("Action_adj", row.get("Action", ""))).lower()
        if "enter" in act:
            return ["background-color: #E6F4EA"] * len(row)
        if "exit" in act:
            return ["background-color: #FDECEC"] * len(row)
        if float(row.get(prob_col, 0.0)) >= entry_threshold:
            return ["background-color: #F6F0DE"] * len(row)
        return [""] * len(row)

    fmt = {
        prob_col: "{:.4f}",
        "Close": "{:.2f}",
        "Target_5d": "{:.2%}",
    }
    cols = [c for c in df.columns if c in fmt]
    return df.style.format({k: v for k, v in fmt.items() if k in cols}).apply(row_color, axis=1)


def main() -> None:
    render_header()
    cfg = render_sidebar()

    price_map, meta_map = load_all_prices(
        tickers=cfg["tickers"],
        start=str(cfg["start_date"]),
        end=str(cfg["end_date"]),
        use_tail=cfg["use_live"],
        interval=cfg["intraday_interval"],
        fallback_last=cfg["fallback_last_session"],
        exec_key=cfg["exec_mode"],
        moc_cutoff=cfg["moc_cutoff_min"],
    )

    if not price_map:
        st.warning("No valid price data loaded.")
        return

    if cfg["enable_optimizer"]:
        with st.expander("Optimizer Results", expanded=False):
            st.caption("Two-stage random search with coarse screening and fine local refinement. Trade count is intentionally excluded from the score.")
            opt_df, best = run_two_stage_optimizer(
                price_map=price_map,
                tickers=cfg["tickers"],
                coarse_trials=cfg["coarse_trials"],
                fine_trials=cfg["fine_trials"],
                top_k=cfg["top_k"],
                seed=cfg["opt_seed"],
                base_settings={
                    "min_hold_days": cfg["min_hold_days"],
                    "cooldown_days": cfg["cooldown_days"],
                    "use_vol_sizing": cfg["use_vol_sizing"],
                    "target_vol_annual": cfg["target_vol_annual"],
                    "pos_frac": cfg["pos_frac"],
                    "commission": cfg["commission"],
                    "slippage_bps": cfg["slippage_bps"],
                    "init_cap": cfg["init_cap"],
                    "use_calibration": cfg["use_calibration"],
                    "calibration_frac": cfg["calibration_frac"],
                },
                w_dd=cfg["w_dd"],
            )
            if not opt_df.empty:
                st.dataframe(opt_df.head(20), use_container_width=True)
                st.download_button(
                    "Download optimizer CSV",
                    to_csv_eu(opt_df),
                    file_name="nexus_optimizer_results.csv",
                    mime="text/csv",
                )
                if best:
                    st.markdown("**Best parameter set applied below:**")
                    st.json(best)
                    cfg["lookback"] = int(best["lookback"])
                    cfg["horizon"] = int(best["horizon"])
                    cfg["threshold"] = float(best["threshold"])
                    cfg["entry_prob"] = float(best["entry_prob"])
                    cfg["exit_prob"] = float(best["exit_prob"])
                    cfg["wf_stride"] = int(best["wf_stride"])
                    cfg["model_params"] = {
                        "max_iter": int(best["max_iter"]),
                        "learning_rate": float(best["learning_rate"]),
                        "max_depth": int(best["max_depth"]),
                        "min_samples_leaf": int(best["min_samples_leaf"]),
                        "l2_regularization": float(best["l2_regularization"]),
                    }

    summaries = []
    live_rows = []
    results: Dict[str, Dict[str, Any]] = {}

    for tk in cfg["tickers"]:
        df = price_map.get(tk)
        if df is None or len(df) < 120:
            continue
        try:
            feat, bt, trades, summary = make_features_and_backtest(
                df=df,
                lookback=cfg["lookback"],
                horizon=cfg["horizon"],
                threshold=cfg["threshold"],
                model_params=cfg["model_params"],
                entry_prob=cfg["entry_prob"],
                exit_prob=cfg["exit_prob"],
                min_hold_days=cfg["min_hold_days"],
                cooldown_days=cfg["cooldown_days"],
                walk_forward=cfg["walk_forward"],
                use_vol_sizing=cfg["use_vol_sizing"],
                target_vol_annual=cfg["target_vol_annual"],
                base_pos_frac=cfg["pos_frac"],
                commission=cfg["commission"],
                slippage_bps=cfg["slippage_bps"],
                init_cap=cfg["init_cap"],
                wf_stride=cfg["wf_stride"],
                use_calibration=cfg["use_calibration"],
                calibration_frac=cfg["calibration_frac"],
            )
            summaries.append({"Ticker": tk, **summary})
            live_rows.append(
                {
                    "Ticker": tk,
                    "Close": float(bt["Close"].iloc[-1]),
                    "SignalProb": float(bt["SignalProb"].iloc[-1]),
                    "Action": bt["Action"].iloc[-1],
                    "Action_adj": bt["Action_adj"].iloc[-1],
                    "Target_5d": float(bt["Target_5d"].iloc[-1]) if np.isfinite(bt["Target_5d"].iloc[-1]) else np.nan,
                }
            )
            results[tk] = {"features": feat, "bt": bt, "trades": pd.DataFrame(trades), "summary": summary, "meta": meta_map.get(tk, {})}
        except Exception as e:
            st.error(f"{tk}: {e}")

    if not summaries:
        st.warning("No ticker could be backtested successfully.")
        return

    summary_df = pd.DataFrame(summaries).sort_values("Net (%)", ascending=False).reset_index(drop=True)
    live_df = pd.DataFrame(live_rows).sort_values("SignalProb", ascending=False).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers", len(summary_df))
    c2.metric("Median Net", fmt_num(summary_df["Net (%)"].median()) + "%")
    c3.metric("Median Sharpe", fmt_num(summary_df["Sharpe"].median()))
    c4.metric("Median Max DD", fmt_num(summary_df["Max DD (%)"].median()) + "%")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Board", "Ticker Detail"])

    with tab1:
        st.markdown('<div class="section-header">Portfolio Summary</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True)
        st.plotly_chart(summary_bar(summary_df), use_container_width=True)
        st.download_button(
            "Download summary CSV",
            to_csv_eu(summary_df),
            file_name="nexus_summary.csv",
            mime="text/csv",
        )

    with tab2:
        st.markdown('<div class="section-header">Latest Signals</div>', unsafe_allow_html=True)
        st.dataframe(style_live_board(live_df, "SignalProb", cfg["entry_prob"]), use_container_width=True)

    with tab3:
        selected = st.selectbox("Ticker", options=list(results.keys()))
        res = results[selected]
        bt = res["bt"]
        trades_df = res["trades"]
        summary = res["summary"]
        meta = res["meta"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net (%)", fmt_num(summary["Net (%)"]) + "%")
        k2.metric("Sharpe", fmt_num(summary["Sharpe"]))
        k3.metric("Buy & Hold (%)", fmt_num(summary["Buy & Hold (%)"]) + "%")
        k4.metric("Closed Trades", str(summary["Closed Trades"]))

        if meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
            st.caption(f"Last datapoint: {bt.index[-1].strftime('%Y-%m-%d %H:%M %Z')} · intraday tail through {meta['tail_ts'].strftime('%H:%M %Z')}")
        else:
            st.caption(f"Last datapoint: {bt.index[-1].strftime('%Y-%m-%d %H:%M %Z')}")

        st.plotly_chart(equity_chart(bt, selected), use_container_width=True)
        st.plotly_chart(prob_chart(bt, cfg["entry_prob"], cfg["exit_prob"], selected), use_container_width=True)
        st.dataframe(trades_df, use_container_width=True)
        if not trades_df.empty:
            st.download_button(
                "Download trades CSV",
                to_csv_eu(trades_df),
                file_name=f"{selected}_trades.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
