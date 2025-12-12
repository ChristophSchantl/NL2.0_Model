# streamlit_app.py
# -*- coding: utf-8 -*-
"""
SHI – STOCK CHECK / Signal-basierte Strategie (OOS-robust, vol-adaptiver Target, kalibrierte Probas)
- Pro Ticker separates Konto (wie gewünscht)
- Out-of-sample SignalProb via Purged TimeSeries CV + Embargo (kein Look-Ahead)
- Vol-adaptiver Target: FutureRet > k * sigma * sqrt(horizon)
- Entry/Exit robust: Quantile der OOS-Probas (oder optional fixe Schwellen)
- Random-Search Optimizer auf OOS-Backtest-Score
"""

# ─────────────────────────────────────────────────────────────
# Imports & Global Config
# ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

from math import sqrt
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# ─────────────────────────────────────────────────────────────
# Streamlit config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHI – STOCK CHECK / PROTEUS (OOS)", layout="wide")

LOCAL_TZ = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 4  # yfinance ist rate-limit sensitiv
pd.options.display.float_format = "{:,.4f}".format


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(index=False, sep=";", decimal=",", date_format="%d.%m.%Y",
                     float_format=float_format).encode("utf-8-sig")


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


def show_styled_or_plain(df: pd.DataFrame, styler):
    try:
        html = getattr(styler, "to_html", None)
        if callable(html):
            st.markdown(html(), unsafe_allow_html=True)
        else:
            raise AttributeError("Styler ohne to_html()")
    except Exception as e:
        st.warning(f"Styled-Tabelle nicht renderbar, fallback auf DataFrame. ({e})")
        st.dataframe(df, use_container_width=True)


def last_timestamp_info(df: pd.DataFrame, meta: Optional[dict] = None):
    ts = df.index[-1]
    msg = f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}"
    if meta and meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
        msg += f" (intraday bis {meta['tail_ts'].strftime('%H:%M %Z')})"
    st.caption(msg)


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


# ─────────────────────────────────────────────────────────────
# Data Loading (Daily + optional Intraday Tail Aggregation)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)  # 5 min cache (live)
def get_price_data_tail_intraday(
    ticker: str,
    years: int = 3,
    use_tail: bool = True,
    interval: str = "5m",
    fallback_last_session: bool = False,
    exec_mode_key: str = "Next Open (backtest+live)",
    moc_cutoff_min_val: int = 15,
) -> Tuple[pd.DataFrame, dict]:

    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{years}y", interval="1d", auto_adjust=True, actions=False)

    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(LOCAL_TZ)
    df = df.sort_index().drop_duplicates()
    meta = {"tail_is_intraday": False, "tail_ts": None}

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    if not use_tail:
        return df, meta

    # Intraday holen
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

    # MOC cutoff
    if exec_mode_key.startswith("Market-On-Close") and not intraday.empty:
        now_local = datetime.now(LOCAL_TZ)
        cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min_val))
        intraday = intraday.loc[:cutoff_time]

    # Fallback: letzte Session
    if intraday.empty and fallback_last_session:
        try:
            intr5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
            if not intr5.empty:
                if intr5.index.tz is None:
                    intr5.index = intr5.index.tz_localize("UTC")
                intr5.index = intr5.index.tz_convert(LOCAL_TZ).sort_index()
                last_session = intr5.index[-1].normalize()
                intraday = intr5.loc[str(last_session.date())]
        except Exception:
            pass

    # Tail-aggregation: überschreibe Tagesbar robust
    if not intraday.empty:
        last_bar_ts = intraday.index[-1]
        day_key = last_bar_ts.normalize()

        daily_row = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(intraday["Close"].iloc[-1]),
            "Volume": float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan,
        }

        # Wenn es bereits eine Daily-Bar für diesen Tag gibt → update, sonst append
        if day_key in df.index.normalize():
            # finde Index-Position des Tages (tz-aware), der normalize entspricht
            idx_match = df.index.normalize() == day_key
            if idx_match.any():
                real_idx = df.index[idx_match][0]
                for k, v in daily_row.items():
                    df.loc[real_idx, k] = v
        else:
            df.loc[pd.Timestamp(day_key, tz=LOCAL_TZ)] = daily_row
            df = df.sort_index()

        meta["tail_is_intraday"] = True
        meta["tail_ts"] = last_bar_ts

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=300)
def get_intraday_last_n_sessions(ticker: str, sessions: int = 5, days_buffer: int = 10, interval: str = "5m") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ).sort_index()
    unique_dates = pd.Index(intr.index.normalize().unique())
    keep_dates = set(unique_dates[-sessions:])
    return intr.loc[intr.index.normalize().isin(keep_dates)].copy()


def load_all_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_tail: bool,
    interval: str,
    fallback_last: bool,
    exec_key: str,
    moc_cutoff: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map: Dict[str, dict] = {}
    if not tickers:
        return price_map, meta_map

    st.info(f"Kurse laden für {len(tickers)} Ticker … (parallel, max_workers={min(MAX_WORKERS, len(tickers))})")
    prog = st.progress(0.0)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        futures = {
            ex.submit(get_price_data_tail_intraday, tk, 3, use_tail, interval, fallback_last, exec_key, int(moc_cutoff)): tk
            for tk in tickers
        }
        done = 0
        for fut in futures:
            tk = futures[fut]
            try:
                df_full, meta = fut.result()
                df_use = df_full.loc[str(start):str(end)].copy()
                if not df_use.empty:
                    price_map[tk] = df_use
                    meta_map[tk] = meta
            except Exception as e:
                st.error(f"Fehler beim Laden von {tk}: {e}")
            finally:
                done += 1
                prog.progress(done / len(tickers))

    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Feature Engineering & Target (vol-adaptiv)
# ─────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    f = df.copy()

    ret = f["Close"].pct_change()
    f["ret_1d"]  = ret
    f["ret_5d"]  = f["Close"].pct_change(5)
    f["ret_20d"] = f["Close"].pct_change(20)

    f["vol_20d"] = ret.rolling(20).std()

    tr = (f["High"] - f["Low"]).abs()
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct"] = f["atr_14"] / f["Close"]

    f["range_lb"] = (
        f["High"].rolling(lookback).max()
        - f["Low"].rolling(lookback).min()
    ) / f["Close"]

    f["ma_fast"] = f["Close"].rolling(10).mean()
    f["ma_slow"] = f["Close"].rolling(50).mean()
    f["trend"]   = (f["ma_fast"] / f["ma_slow"]) - 1.0

    f["dd_20d"] = f["Close"] / f["Close"].rolling(20).max() - 1.0

    f = f.dropna()
    return f


def make_target_vol_adaptive(feat: pd.DataFrame, horizon: int, k_vol: float) -> pd.Series:
    fwd_ret = feat["Close"].shift(-horizon) / feat["Close"] - 1.0
    sigma = feat["ret_1d"].rolling(20).std() * np.sqrt(horizon)
    y = (fwd_ret > (k_vol * sigma)).astype(int)
    return y


# ─────────────────────────────────────────────────────────────
# Purged TimeSeries CV (Embargo) + Calibrated Probas
# ─────────────────────────────────────────────────────────────
def oos_signal_probabilities_purged(
    feat: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    horizon: int,
    model_params: dict,
    n_splits: int = 5,
    min_train: int = 120,
    calibrate: bool = True,
    calib_method: str = "isotonic",  # "sigmoid" oder "isotonic"
) -> pd.Series:
    """
    Liefert OOS-Probas für jede Zeitperiode via Walk-Forward Splits.
    Purge: Training darf nicht in Test überlappen (Embargo = horizon Bars).
    """

    X = feat[feature_cols].copy()
    y = y.copy()

    # gemeinsame gültige Zeilen (Label kann am Ende NaN sein wegen shift(-horizon))
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    probs = pd.Series(index=X.index, dtype=float)

    n = len(X)
    if n < (min_train + 40):
        return probs.reindex(feat.index).ffill().fillna(0.5)

    # Splitpunkte deterministisch
    split_points = np.linspace(min_train, n - 1, num=n_splits + 1, dtype=int)[1:]
    prev = 0
    for sp in split_points:
        test_start = prev
        test_end = sp
        prev = sp

        if test_end - test_start < 10:
            continue

        # Train-Ende ist vor Test-Start mit Embargo
        train_end = max(0, test_start - horizon)
        if train_end < min_train:
            continue

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[test_start:test_end]

        if y_tr.nunique() < 2:
            continue

        base = GradientBoostingClassifier(**model_params)

        if calibrate:
            clf = CalibratedClassifierCV(base, method=calib_method, cv=3)
        else:
            clf = base

        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        probs.iloc[test_start:test_end] = p

    # Zurück auf feat-index, forward-fill, fallback 0.5
    probs = probs.reindex(feat.index).ffill().fillna(0.5)
    return probs


def compute_entry_exit_thresholds_from_probs(
    prob_oos: pd.Series,
    mode: str,
    entry_fixed: float,
    exit_fixed: float,
    q_entry: float,
    q_exit: float,
) -> Tuple[float, float]:
    if mode == "Fixe Schwellen":
        return float(entry_fixed), float(exit_fixed)
    # Quantile (robust)
    entry_thr = float(prob_oos.quantile(q_entry))
    exit_thr  = float(prob_oos.quantile(q_exit))
    # Safety: exit < entry
    if exit_thr >= entry_thr:
        exit_thr = max(0.0, entry_thr - 0.05)
    return entry_thr, exit_thr


# ─────────────────────────────────────────────────────────────
# Backtest (Next Open) – separates Konto pro Ticker
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

    # Execution: SignalProb[t-1] → Trade at Open[t]
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

            # ENTRY
            can_enter = (not in_pos) and (prob_prev > entry_thr) and cool_ok
            if can_enter:
                invest_net = cash_net * float(pos_frac)
                fee_entry = invest_net * float(commission)
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)

                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9:
                    shares = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry

                    cash_gross -= cost_basis_gross
                    cash_net   -= cost_basis_net
                    in_pos = True
                    last_entry_idx = i

                    trades.append({
                        "Date": date_exec, "Typ": "Entry", "Price": round(slip_buy, 4),
                        "Shares": round(shares, 6), "Gross P&L": 0.0,
                        "Fees": round(fee_entry, 2), "Net P&L": 0.0,
                        "kum P&L": round(cum_pl_net, 2), "Prob": round(prob_prev, 4),
                        "HoldDays": np.nan
                    })

            # EXIT
            elif in_pos and (prob_prev < exit_thr):
                held_bars = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_days) > 0 and held_bars < int(min_hold_days):
                    pass
                else:
                    gross_value = shares * slip_sell
                    fee_exit = gross_value * float(commission)

                    pnl_gross = gross_value - cost_basis_gross
                    pnl_net   = (gross_value - fee_exit) - cost_basis_net

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

    out = df.copy()
    out["Equity_Gross"] = equity_gross
    out["Equity_Net"]   = equity_net
    return out, trades


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────
def _cagr_from_path(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    years = len(values) / 252.0
    if years <= 0:
        return np.nan
    return (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1


def _sortino(rets: pd.Series) -> float:
    if rets.empty:
        return np.nan
    mean = rets.mean() * 252
    downside = rets[rets < 0]
    dd = downside.std() * np.sqrt(252) if len(downside) else np.nan
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

    vol_ann = rets.std() * sqrt(252) * 100
    sharpe = (rets.mean() * sqrt(252)) / (rets.std() + 1e-12)

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
    }


def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        current_entry = None
        for ev in tr:
            if ev["Typ"] == "Entry":
                current_entry = ev
            elif ev["Typ"] == "Exit" and current_entry is not None:
                entry_date = pd.to_datetime(current_entry["Date"])
                exit_date = pd.to_datetime(ev["Date"])
                hold_days = (exit_date - entry_date).days
                shares = float(current_entry.get("Shares", 0.0))
                entry_p = float(current_entry.get("Price", np.nan))
                exit_p = float(ev.get("Price", np.nan))
                fee_e = float(current_entry.get("Fees", 0.0))
                fee_x = float(ev.get("Fees", 0.0))
                pnl_net = float(ev.get("Net P&L", 0.0))
                cost_net = shares * entry_p + fee_e
                ret_pct = (pnl_net / cost_net * 100.0) if cost_net else np.nan
                rows.append({
                    "Ticker": tk, "Name": name,
                    "Entry Date": entry_date, "Exit Date": exit_date,
                    "Hold (days)": hold_days,
                    "Entry Prob": current_entry.get("Prob", np.nan),
                    "Exit Prob": ev.get("Prob", np.nan),
                    "Shares": round(shares, 6),
                    "Entry Price": round(entry_p, 4),
                    "Exit Price": round(exit_p, 4),
                    "PnL Net (€)": round(pnl_net, 2),
                    "Fees (€)": round(fee_e + fee_x, 2),
                    "Return (%)": round(ret_pct, 2),
                })
                current_entry = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# UI – Sidebar
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
    extra_csv = st.sidebar.text_input("Weitere Ticker hinzufügen (Komma-getrennt)", value="", key="extra_csv")
    extras = _normalize_tickers([t for t in extra_csv.split(",") if t.strip()]) if extra_csv else []
    tickers_final = _normalize_tickers(base + extras)

    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker: {len(tickers_final)}")
        if st.sidebar.checkbox("Zufällig mischen (seed=42)", value=False):
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

# Dates
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
END_DATE   = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

# Costs / sizing (pro ticker)
st.sidebar.markdown("**Execution / Kosten**")
COMMISSION   = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", 0.0, 0.02, 0.004,
                                       step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", 0, 50, 5, step=1)
POS_FRAC     = st.sidebar.slider("Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1)
INIT_CAP     = st.sidebar.number_input("Initial Capital pro Ticker (€)", min_value=1000.0,
                                       value=10_000.0, step=1000.0, format="%.2f")

MIN_HOLD_DAYS = st.sidebar.number_input("Mindesthaltedauer (Handelstage)", 0, 252, 5, step=1)
COOLDOWN_DAYS = st.sidebar.number_input("Cooling Phase nach Exit (Handelstage)", 0, 252, 0, step=1)

# Live intraday tail
st.sidebar.markdown("**Live / Intraday**")
use_live = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall", ["1m", "2m", "5m", "15m"], index=2)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session nutzen (wenn heute leer)", value=False)
exec_mode = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", 5, 60, 15, step=5)
intraday_chart_type = st.sidebar.selectbox("Intraday-Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

# Model / OOS engine
st.sidebar.markdown("**Modell / OOS Engine**")
LOOKBACK = st.sidebar.number_input("Lookback (Tage)", 10, 252, 60, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)", 1, 15, 5)

K_VOL = st.sidebar.slider("Target k (vol-adaptiv): fwd_ret > k·σ·√h", 0.3, 3.0, 1.0, step=0.05)

n_splits = st.sidebar.slider("OOS Splits (Walk-Forward)", 3, 10, 5, step=1)
min_train = st.sidebar.number_input("Min. Train Bars", 60, 500, 160, step=10)
calibrate = st.sidebar.checkbox("Probas kalibrieren", value=True)
calib_method = st.sidebar.selectbox("Kalibrierung", ["isotonic", "sigmoid"], index=0)

# Entry/Exit threshold mode
st.sidebar.markdown("**Entry/Exit Logik**")
thr_mode = st.sidebar.selectbox("Threshold-Modus", ["Quantile (robust)", "Fixe Schwellen"], index=0)
ENTRY_FIXED = st.sidebar.slider("Entry Threshold (fix)", 0.0, 1.0, 0.65, step=0.01)
EXIT_FIXED  = st.sidebar.slider("Exit Threshold (fix)",  0.0, 1.0, 0.45, step=0.01)
Q_ENTRY = st.sidebar.slider("Entry Quantil", 0.50, 0.99, 0.80, step=0.01)
Q_EXIT  = st.sidebar.slider("Exit Quantil",  0.01, 0.80, 0.40, step=0.01)

if thr_mode == "Fixe Schwellen" and EXIT_FIXED >= ENTRY_FIXED:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()
if thr_mode == "Quantile (robust)" and Q_EXIT >= Q_ENTRY:
    st.sidebar.error("Exit-Quantil muss unter Entry-Quantil liegen.")
    st.stop()

# GBC params
st.sidebar.markdown("**GradientBoostingClassifier**")
n_estimators  = st.sidebar.number_input("n_estimators",  50, 800, 200, step=50)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 0.5, 0.05, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     1, 6, 3, step=1)
subsample     = st.sidebar.slider("subsample", 0.5, 1.0, 1.0, step=0.05)

MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    subsample=float(subsample),
    random_state=42
)

# Cache management
c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear()
    st.rerun()
if c2.button("↻ Rerun"):
    st.rerun()


# ─────────────────────────────────────────────────────────────
# Optimizer (OOS robust)
# ─────────────────────────────────────────────────────────────
st.subheader("🧭 Parameter-Optimierung (OOS-robust)")
with st.expander("Optimizer (Random Search, Purged OOS + Score)", expanded=False):
    n_trials = st.number_input("Trials", 10, 1500, 120, step=10)
    seed = st.number_input("Seed", 0, 100_000, 42)

    # Score weights
    lambda_dd = st.number_input("Penalty λ_DD (Drawdown)", 0.0, 5.0, 0.8, step=0.1)
    lambda_tr = st.number_input("Penalty λ_TR (Trades/Jahr)", 0.0, 1.0, 0.02, step=0.005)

    min_trades_req = st.number_input("Min. Trades gesamt (Filter)", 0, 10_000, 5, step=1)

    # Ranges
    lb_lo, lb_hi = st.slider("Lookback Range", 10, 252, (30, 120), step=5)
    hz_lo, hz_hi = st.slider("Horizon Range", 1, 15, (3, 8), step=1)
    k_lo,  k_hi  = st.slider("k_vol Range", 0.3, 3.0, (0.6, 1.6), step=0.05)

    qe_lo, qe_hi = st.slider("Entry Quantil Range", 0.50, 0.99, (0.70, 0.90), step=0.01)
    qx_lo, qx_hi = st.slider("Exit Quantil Range", 0.01, 0.80, (0.20, 0.55), step=0.01)

    @st.cache_data(show_spinner=False)
    def _get_prices_for_optimizer(
        tickers: tuple, start: str, end: str, use_tail: bool, interval: str,
        fallback_last: bool, exec_key: str, moc_cutoff: int
    ):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fallback_last, exec_key, moc_cutoff)[0]

    def _run_one(df: pd.DataFrame, lb: int, hz: int, k: float, qe: float, qx: float) -> Tuple[dict, int]:
        feat = make_features(df, lb)
        y = make_target_vol_adaptive(feat, hz, k)

        feature_cols = ["ret_1d","ret_5d","ret_20d","vol_20d","atr_pct","range_lb","trend","dd_20d"]
        feat["SignalProb"] = oos_signal_probabilities_purged(
            feat, feature_cols, y, hz, MODEL_PARAMS,
            n_splits=int(n_splits),
            min_train=int(min_train),
            calibrate=bool(calibrate),
            calib_method=str(calib_method)
        )

        entry_thr, exit_thr = compute_entry_exit_thresholds_from_probs(
            feat["SignalProb"], "Quantile (robust)", ENTRY_FIXED, EXIT_FIXED, qe, qx
        )

        bt, trades = backtest_next_open(
            feat, entry_thr, exit_thr,
            COMMISSION, SLIPPAGE_BPS, INIT_CAP, POS_FRAC,
            min_hold_days=int(MIN_HOLD_DAYS),
            cooldown_days=int(COOLDOWN_DAYS),
        )
        mets = compute_performance(bt, trades, INIT_CAP)
        return mets, int(mets["Number of Trades"])

    if st.button("🔎 Suche starten", type="primary", use_container_width=True):
        import random
        rng = random.Random(int(seed))

        price_map = _get_prices_for_optimizer(
            tuple(TICKERS),
            str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
        )

        feasible_tickers = [tk for tk, df in price_map.items() if df is not None and len(df) >= 220]
        if not feasible_tickers:
            st.warning("Keine ausreichenden Preisdaten für Optimierung.")
        else:
            rows = []
            best = None
            prog = st.progress(0.0)

            for t in range(int(n_trials)):
                lb = rng.randrange(lb_lo, lb_hi + 1, 5)
                hz = rng.randrange(hz_lo, hz_hi + 1, 1)
                k  = rng.uniform(k_lo, k_hi)
                qe = rng.uniform(qe_lo, qe_hi)
                qx = rng.uniform(qx_lo, qx_hi)

                if qx >= qe:
                    prog.progress((t+1)/n_trials)
                    continue

                sharps = []
                maxdds = []
                trades_total = 0
                feasible = 0

                # Walk-Forward-light: split je ticker in zwei Hälften (wie bei dir),
                # aber OOS-probas sind bereits purged.
                for tk in feasible_tickers:
                    df = price_map.get(tk)
                    if df is None or len(df) < max(220, lb + hz + 80):
                        continue
                    mid = len(df) // 2
                    for sub in (df.iloc[:mid], df.iloc[mid:]):
                        if len(sub) < max(220, lb + hz + 80):
                            continue
                        feasible += 1
                        try:
                            mets, trn = _run_one(sub, lb, hz, k, qe, qx)
                            if np.isfinite(mets["Sharpe-Ratio"]):
                                sharps.append(float(mets["Sharpe-Ratio"]))
                            if np.isfinite(mets["Max Drawdown (%)"]):
                                maxdds.append(float(mets["Max Drawdown (%)"]))
                            trades_total += int(trn)
                        except Exception:
                            pass

                if feasible == 0:
                    prog.progress((t+1)/n_trials)
                    continue

                sharpe_med = float(np.nanmedian(sharps)) if sharps else np.nan
                mdd_med = float(np.nanmedian(maxdds)) if maxdds else np.nan

                # Trades/Jahr approx (pro ticker-half ~ length/252)
                denom = max(1, feasible)
                trades_avg = trades_total / denom

                if trades_total < int(min_trades_req) or not np.isfinite(sharpe_med):
                    prog.progress((t+1)/n_trials)
                    continue

                # Score: Sharpe − λ_DD*|DD| − λ_TR*trades_avg
                score = sharpe_med - float(lambda_dd) * abs(mdd_med)/100.0 - float(lambda_tr) * float(trades_avg)

                rec = dict(
                    trial=t, score=score,
                    sharpe_med=sharpe_med,
                    mdd_med=mdd_med,
                    trades=trades_total,
                    lookback=lb, horizon=hz, k_vol=k, q_entry=qe, q_exit=qx
                )
                rows.append(rec)
                if (best is None) or (score > best["score"]):
                    best = rec

                prog.progress((t+1)/n_trials)

            if not rows:
                st.warning("Keine gültigen Kandidaten gefunden.")
            else:
                df_res = pd.DataFrame(rows).sort_values("score", ascending=False)
                st.success(
                    f"Beste Parameter: Score={best['score']:.3f} | Sharpe_med={best['sharpe_med']:.2f} "
                    f"| MDD_med={best['mdd_med']:.2f}% | Trades={best['trades']}"
                )
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Lookback", int(best["lookback"]))
                c2.metric("Horizon", int(best["horizon"]))
                c3.metric("k_vol", f"{best['k_vol']:.2f}")
                c4.metric("Entry Quantil", f"{best['q_entry']:.2f}")
                c5.metric("Exit Quantil", f"{best['q_exit']:.2f}")

                st.caption("Top-Ergebnisse (Score = Sharpe_med − λ_DD·|DD| − λ_TR·Trades_avg)")
                st.dataframe(df_res.head(30), use_container_width=True)

                st.download_button(
                    "Optimierergebnisse als CSV",
                    to_csv_eu(df_res),
                    file_name="param_search_results.csv", mime="text/csv"
                )


# ─────────────────────────────────────────────────────────────
# Main – Pipeline
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 34px;'>📈 PROTEUS – OOS Signal Engine</h1>", unsafe_allow_html=True)
st.caption("Kerneigenschaft: keine In-Sample-Leakage (Purged OOS), vol-adaptives Target, robuste Entry/Exit.")

price_map, meta_map = load_all_prices(
    TICKERS,
    str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
)

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs: Dict[str, pd.DataFrame] = {}
all_feat: Dict[str, pd.DataFrame] = {}

live_forecasts_run: List[dict] = []


def decide_action(p: float, entry_thr: float, exit_thr: float) -> str:
    if p > entry_thr:
        return "Enter / Add"
    if p < exit_thr:
        return "Exit / Reduce"
    return "Hold / No Trade"


# Features used
FEATURE_COLS = ["ret_1d","ret_5d","ret_20d","vol_20d","atr_pct","range_lb","trend","dd_20d"]

for ticker in TICKERS:
    if ticker not in price_map:
        continue

    df = price_map[ticker]
    meta = meta_map.get(ticker, {})

    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker} — {get_ticker_name(ticker)}")
        try:
            last_timestamp_info(df, meta)

            feat = make_features(df, int(LOOKBACK))
            y = make_target_vol_adaptive(feat, int(HORIZON), float(K_VOL))

            feat["SignalProb"] = oos_signal_probabilities_purged(
                feat, FEATURE_COLS, y,
                horizon=int(HORIZON),
                model_params=MODEL_PARAMS,
                n_splits=int(n_splits),
                min_train=int(min_train),
                calibrate=bool(calibrate),
                calib_method=str(calib_method)
            )

            entry_thr, exit_thr = compute_entry_exit_thresholds_from_probs(
                feat["SignalProb"],
                "Fixe Schwellen" if thr_mode == "Fixe Schwellen" else "Quantile (robust)",
                ENTRY_FIXED, EXIT_FIXED,
                Q_ENTRY, Q_EXIT
            )

            df_bt, trades = backtest_next_open(
                feat,
                entry_thr, exit_thr,
                COMMISSION, SLIPPAGE_BPS,
                INIT_CAP, POS_FRAC,
                min_hold_days=int(MIN_HOLD_DAYS),
                cooldown_days=int(COOLDOWN_DAYS)
            )

            metrics = compute_performance(df_bt, trades, INIT_CAP)
            metrics["Ticker"] = ticker
            metrics["EntryThr"] = float(entry_thr)
            metrics["ExitThr"]  = float(exit_thr)

            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat

            # Live row
            live_ts = pd.Timestamp(feat.index[-1])
            live_prob = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1])
            tail_info = "intraday" if meta.get("tail_is_intraday") else "daily"

            live_forecasts_run.append({
                "AsOf": live_ts.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "Name": get_ticker_name(ticker),
                f"P(signal, {HORIZON}d)": round(live_prob, 4),
                "EntryThr": round(float(entry_thr), 4),
                "ExitThr":  round(float(exit_thr), 4),
                "Action": decide_action(live_prob, float(entry_thr), float(exit_thr)),
                "Close": round(live_close, 4),
                "Target_approx": round(live_close * (1 + float(K_VOL) * float(feat["vol_20d"].iloc[-1]) * np.sqrt(int(HORIZON))), 4)
                                 if np.isfinite(float(feat["vol_20d"].iloc[-1])) else np.nan,
                "Bar": tail_info,
            })

            # KPIs
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)", f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe", f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Sortino", f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5.metric("Max DD (%)", f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Trades", f"{int(metrics['Number of Trades'])}")

            st.caption(
                f"Entry/Exit: {thr_mode} | EntryThr={entry_thr:.4f} | ExitThr={exit_thr:.4f} "
                f"| Target: fwd_ret > k·σ·√h (k={K_VOL:.2f})"
            )

            # Charts: Price colored by prob (OOS)
            chart_cols = st.columns(2)

            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], mode="lines",
                                           name="Close", line=dict(color="rgba(0,0,0,0.35)", width=1)))

            sp = df_plot["SignalProb"].astype(float)
            norm = (sp - sp.min()) / (sp.max() - sp.min() + 1e-9)
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i:i+2]
                seg_y = df_plot["Close"].iloc[i:i+2]
                color_seg = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, float(norm.iloc[i]))[0]
                price_fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False,
                                               line=dict(color=color_seg, width=2), hoverinfo="skip"))

            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"] == "Entry"]
                exits = trades_df[trades_df["Typ"] == "Exit"]
                price_fig.add_trace(go.Scatter(x=entries["Date"], y=entries["Price"], mode="markers", name="Entry",
                                               marker_symbol="triangle-up", marker=dict(size=12, color="green")))
                price_fig.add_trace(go.Scatter(x=exits["Date"], y=exits["Price"], mode="markers", name="Exit",
                                               marker_symbol="triangle-down", marker=dict(size=12, color="red")))

            price_fig.update_layout(
                title=f"{ticker}: Preis + OOS SignalProb (Daily)",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            # Intraday last 5 sessions
            intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
            with chart_cols[1]:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar.")
                else:
                    intr_fig = go.Figure()
                    if intraday_chart_type == "Candlestick (OHLC)":
                        intr_fig.add_trace(go.Candlestick(
                            x=intra.index, open=intra["Open"], high=intra["High"],
                            low=intra["Low"], close=intra["Close"], name="OHLC"
                        ))
                    else:
                        intr_fig.add_trace(go.Scatter(x=intra.index, y=intra["Close"], mode="lines", name="Close"))

                    intr_fig.update_layout(
                        title=f"{ticker}: Intraday – letzte 5 Handelstage ({intraday_interval})",
                        height=420, margin=dict(t=50, b=30, l=40, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(intr_fig, use_container_width=True)

            # Equity
            eq = go.Figure()
            eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity",
                                    mode="lines"))
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(x=df_bt.index, y=bh_curve, name="Buy & Hold",
                                    mode="lines", line=dict(dash="dash", color="black")))
            eq.update_layout(title=f"{ticker}: Net Equity vs. Buy & Hold",
                             height=400, margin=dict(t=50, b=30, l=40, r=20),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(eq, use_container_width=True)

            # Trades table + export
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                if trades_df.empty:
                    st.info("Keine Trades vorhanden.")
                else:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"] = ticker
                    df_tr["Name"] = get_ticker_name(ticker)
                    df_tr["Date"] = pd.to_datetime(df_tr["Date"])
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%d.%m.%Y")
                    df_tr["CumPnL"] = (
                        df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"]
                        .cumsum().ffill().fillna(0.0)
                    )
                    df_tr = df_tr.rename(columns={"Net P&L":"PnL","Prob":"Signal Prob","HoldDays":"Hold (days)"})

                    disp_cols = ["Ticker","Name","DateStr","Typ","Price","Shares","Signal Prob","Hold (days)","PnL","CumPnL","Fees"]
                    styled = df_tr[disp_cols].rename(columns={"DateStr":"Date"}).style.format({
                        "Price":"{:.2f}","Shares":"{:.6f}","Signal Prob":"{:.4f}",
                        "PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}"
                    })
                    show_styled_or_plain(df_tr[disp_cols].rename(columns={"DateStr":"Date"}), styled)

                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=to_csv_eu(
                            df_tr[["Ticker","Name","Date","Typ","Price","Shares","Signal Prob","Hold (days)","PnL","CumPnL","Fees"]],
                            float_format="%.6f"
                        ),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key=f"dl_trades_{ticker}",
                    )

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")


# ─────────────────────────────────────────────────────────────
# Live Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
          .drop_duplicates(subset=["Ticker"], keep="last")
          .sort_values(["AsOf", "Ticker"])
          .reset_index(drop=True)
    )

    st.markdown("### 🟣 Live–Forecast Board (OOS-Proba, pro Ticker)")
    st.dataframe(live_df, use_container_width=True)

    st.download_button(
        "Live-Forecasts als CSV",
        to_csv_eu(live_df),
        file_name=f"live_forecasts_today_{HORIZON}d.csv", mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Round-Trips / Correlation
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / float(INIT_CAP)) * 100

    total_net_pnl   = summary_df["Net P&L (€)"].sum()
    total_fees      = summary_df["Fees (€)"].sum()
    total_gross_pnl = total_net_pnl + total_fees
    total_trades    = summary_df["Number of Trades"].sum()

    total_capital   = float(INIT_CAP) * len(summary_df)
    total_net_return_pct = (total_net_pnl / total_capital) * 100 if total_capital else np.nan
    total_gross_return_pct = (total_gross_pnl / total_capital) * 100 if total_capital else np.nan
    bh_total_pct = float(summary_df["Buy & Hold Net (%)"].dropna().mean()) if "Buy & Hold Net (%)" in summary_df.columns else float("nan")

    st.subheader("📊 Summary of all Tickers (Next Open Backtest, OOS-Proba)")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)", f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Trades", f"{int(total_trades)}")

    cols_pct = st.columns(4)
    cols_pct[0].metric("Strategy Net (%) – total", f"{total_net_return_pct:.2f}")
    cols_pct[1].metric("Strategy Gross (%) – total", f"{total_gross_return_pct:.2f}")
    cols_pct[2].metric("Buy & Hold Net (%) – avg", f"{bh_total_pct:.2f}")
    cols_pct[3].metric("Ø Sharpe", f"{summary_df['Sharpe-Ratio'].dropna().mean():.2f}" if "Sharpe-Ratio" in summary_df else "–")

    styled = (
        summary_df.style
        .format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}",
            "Buy & Hold Net (%)":"{:.2f}","Volatility (%)":"{:.2f}",
            "Sharpe-Ratio":"{:.2f}","Sortino-Ratio":"{:.2f}",
            "Max Drawdown (%)":"{:.2f}","Calmar-Ratio":"{:.2f}",
            "Fees (€)":"{:.2f}","Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}",
            "CAGR (%)":"{:.2f}","Winrate (%)":"{:.2f}",
            "EntryThr":"{:.4f}","ExitThr":"{:.4f}",
        })
        .set_caption("Performance pro Ticker (OOS SignalProb, Next-Open Execution)")
    )
    show_styled_or_plain(summary_df, styled)

    st.download_button(
        "Summary als CSV herunterladen",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary.csv", mime="text/csv"
    )

    # Open Positions (pro ticker)
    st.subheader("📋 Open Positions (Next Open Backtest)")
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            entry_ts = pd.to_datetime(last_entry["Date"])

            prob = float(all_feat[ticker]["SignalProb"].iloc[-1])
            last_close = float(all_dfs[ticker]["Close"].iloc[-1])
            upnl = (last_close - float(last_entry["Price"])) * float(last_entry["Shares"])

            open_positions.append({
                "Ticker": ticker, "Name": get_ticker_name(ticker),
                "Entry Date": entry_ts, "Entry Price": round(float(last_entry["Price"]), 2),
                "Current Prob.": round(prob, 4),
                "Unrealized PnL (€)": round(upnl, 2),
            })

    if open_positions:
        open_df = pd.DataFrame(open_positions).sort_values("Entry Date", ascending=False)
        open_disp = open_df.copy()
        open_disp["Entry Date"] = open_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(open_disp, use_container_width=True)
        st.download_button("Offene Positionen als CSV", to_csv_eu(open_df), file_name="open_positions.csv", mime="text/csv")
    else:
        st.success("Keine offenen Positionen.")

    # Round Trips
    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips)")
        st.dataframe(rt_df.sort_values("Exit Date", ascending=False), use_container_width=True)
        st.download_button("Round-Trips als CSV", to_csv_eu(rt_df), file_name="round_trips.csv", mime="text/csv")

    # Correlation (Close returns)
    st.subheader("🔗 Portfolio-Korrelation (Close-Returns)")
    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.2, 1.0])
    with c1:
        corr_freq = st.selectbox("Return-Frequenz", ["täglich", "wöchentlich", "monatlich"], index=0)
    with c2:
        corr_method = st.selectbox("Korrelationsmethode", ["Pearson", "Spearman", "Kendall"], index=0)
    with c3:
        min_obs = st.slider("Min. gemeinsame Zeitpunkte", 3, 60, 20, step=1)
    with c4:
        use_ffill = st.checkbox("Lücken per FFill schließen", value=True)

    price_series = []
    for tk, dfbt in all_dfs.items():
        if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
            s = dfbt["Close"].copy()
            s.name = tk
            price_series.append(s)

    if len(price_series) < 2:
        st.info("Mindestens zwei Ticker mit Daten nötig.")
    else:
        prices = pd.concat(price_series, axis=1, join="outer").sort_index()
        if use_ffill:
            prices = prices.ffill()

        if corr_freq == "wöchentlich":
            prices = prices.resample("W-FRI").last()
        elif corr_freq == "monatlich":
            prices = prices.resample("M").last()

        rets = prices.pct_change().dropna(how="all")
        enough = [c for c in rets.columns if rets[c].count() >= min_obs]
        rets = rets[enough]
        common_rows = rets.dropna(how="any")

        if rets.shape[1] < 2 or len(common_rows) < min_obs:
            st.info("Zu wenige Datenüberschneidungen für eine Korrelationsmatrix.")
        else:
            corr = rets.corr(method=corr_method.lower(), min_periods=min_obs)
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_layout(height=560, margin=dict(t=40, l=40, r=30, b=40),
                                   coloraxis_colorbar=dict(title="ρ"))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(f"Basis: {len(common_rows)} gemeinsame Zeitpunkte · Frequenz: {corr_freq} · Methode: {corr_method}")

else:
    st.warning("Noch keine Ergebnisse verfügbar. Prüfe Ticker-Eingaben und Datenabdeckung.")
