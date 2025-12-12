# streamlit_app.py
# ─────────────────────────────────────────────────────────────
# PROTEUS – Signal-basierte Strategie (Full Version)
# (A) pro Ticker separates Konto + Fix: Phase-Style ("Open" blau)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Imports & Global Config
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
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SHI – STOCK CHECK / PROTEUS", layout="wide")
LOCAL_TZ = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 6  # yfinance rate-limit sensibel: ggf. 2-4
pd.options.display.float_format = "{:,.4f}".format


# ─────────────────────────────────────────────────────────────
# Helpers (CSV zuerst wegen früher Nutzung)
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
# Sidebar – Global Controls
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

# Core Backtest-Parameter
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
END_DATE   = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))

LOOKBACK = st.sidebar.number_input("Lookback (Tage)", 10, 252, 35, step=5)
HORIZON  = st.sidebar.number_input("Horizon (Tage)", 1, 10, 5)
THRESH   = st.sidebar.number_input("Threshold für Target (fix)", 0.0, 0.1, 0.046, step=0.005, format="%.3f")

ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", 0.0, 1.0, 0.62, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  0.0, 1.0, 0.48, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()

MIN_HOLD_DAYS = st.sidebar.number_input(
    "Mindesthaltedauer (Handelstage)", 0, 252, 5, step=1,
    help="Sperrt Exits, bis die Position mindestens so viele Handelstage gehalten wurde."
)
COOLDOWN_DAYS = st.sidebar.number_input(
    "Cooling Phase nach Exit (Handelstage)", 0, 252, 0, step=1,
    help="Verhindert Neueinstiege für X Handelstage nach einem Exit (pro Ticker)."
)

COMMISSION   = st.sidebar.number_input("Commission (ad valorem, z.B. 0.001=10bp)", 0.0, 0.02, 0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS = st.sidebar.number_input("Slippage (bp je Ausführung)", 0, 50, 5, step=1)
POS_FRAC     = st.sidebar.slider("Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1)

# (A) Pro Ticker separates Konto:
INIT_CAP_PER_TICKER = st.sidebar.number_input("Initial Capital pro Ticker (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# Intraday
use_live = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox("Intraday-Intervall (Tail & 5-Tage-Chart)", ["1m", "2m", "5m", "15m"], index=2)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session verwenden (wenn heute leer)", value=False)
exec_mode = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
moc_cutoff_min = st.sidebar.number_input("MOC Cutoff (Minuten vor Close, nur live)", 5, 60, 15, step=5)
intraday_chart_type = st.sidebar.selectbox("Intraday-Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

# Modellparameter
st.sidebar.markdown("**Modellparameter**")
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 1.0, 0.1, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth",     1, 10, 3, step=1)
MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42
)

# Walk-forward/OOS
st.sidebar.markdown("**OOS / Walk-Forward**")
use_walk_forward = st.sidebar.checkbox("Walk-Forward (OOS) Probas verwenden", value=False, help="Langsamer, aber OOS-robuster.")
wf_min_train = st.sidebar.number_input("WF min_train Bars", 40, 500, 120, step=10)

# Optionsdaten
st.sidebar.markdown("**Optionsdaten (Einzelaktie)**")
use_chain_live = st.sidebar.checkbox("Live-Optionskette je Aktie nutzen (PCR/VOI)", value=True)
atm_band_pct   = st.sidebar.slider("ATM-Band (±%)", 1, 15, 5, step=1) / 100.0
max_days_to_exp= st.sidebar.slider("Max. Restlaufzeit (Tage)", 7, 45, 21, step=1)
n_expiries     = st.sidebar.slider("Nächste n Verfälle", 1, 4, 2, step=1)

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
def show_styled_or_plain(df: pd.DataFrame, styler):
    """
    Robustes Rendering:
    - bevorzugt HTML-Styler (damit Farben zuverlässig sind)
    - Fallback nur wenn Styler wirklich scheitert
    """
    try:
        st.markdown(styler.to_html(), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Styled-Tabelle nicht renderbar, fallback auf DataFrame. ({e})")
        st.dataframe(df, use_container_width=True)


def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0


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


def style_live_board(df: pd.DataFrame, prob_col: str, entry_threshold: float):
    def _row_color(row):
        act = str(row.get("Action_adj", row.get("Action",""))).lower()
        if "enter" in act: return ["background-color: #D7F3F7"] * len(row)
        if "exit"  in act: return ["background-color: #FFE8E8"] * len(row)
        try:
            if float(row.get(prob_col, np.nan)) >= float(entry_threshold):
                return ["background-color: #E6F7FF"] * len(row)
        except Exception:
            pass
        return ["background-color: #F7F7F7"] * len(row)

    fmt = {prob_col: "{:.4f}"}
    if "P_adj" in df.columns: fmt["P_adj"] = "{:.4f}"
    if "Close" in df.columns: fmt["Close"] = "{:.2f}"
    if "Target_5d" in df.columns: fmt["Target_5d"] = "{:.2f}"
    for c in ["PCR_oi","PCR_vol","VOI_call","VOI_put"]:
        if c in df.columns: fmt[c] = "{:.4f}"

    sty = df.style.format(fmt).apply(_row_color, axis=1)
    subset_cols = [c for c in ["Action","Action_adj"] if c in df.columns]
    if subset_cols:
        sty = sty.set_properties(subset=subset_cols, **{"font-weight": "600"})
    return sty


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=180)
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
        now_local = datetime.now(LOCAL_TZ)
        cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min_val))
        intraday = intraday.loc[:cutoff_time]

    if intraday.empty and fallback_last_session:
        try:
            intraday5 = tk.history(period="5d", interval=interval, auto_adjust=True, actions=False, prepost=False)
            if not intraday5.empty:
                if intraday5.index.tz is None:
                    intraday5.index = intraday5.index.tz_localize("UTC")
                intraday5.index = intraday5.index.tz_convert(LOCAL_TZ)
                intraday5 = intraday5.sort_index()
                last_session_date = intraday5.index[-1].date()
                intraday = intraday5.loc[str(last_session_date)]
        except Exception:
            pass

    if not intraday.empty:
        last_bar = intraday.iloc[-1]
        day_key = pd.Timestamp(last_bar.name.date(), tz=LOCAL_TZ)

        daily_row = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(last_bar["Close"]),
            "Volume": float(intraday["Volume"].sum()) if "Volume" in intraday.columns else np.nan,
        }

        if day_key in df.index:
            for k, v in daily_row.items():
                df.loc[day_key, k] = v
        else:
            df.loc[day_key] = daily_row

        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"] = last_bar.name

    df.dropna(subset=["High", "Low", "Close", "Open"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=180)
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


def load_all_prices(tickers: List[str], start: str, end: str,
                    use_tail: bool, interval: str, fallback_last: bool,
                    exec_key: str, moc_cutoff: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map: Dict[str, dict] = {}
    if not tickers:
        return price_map, meta_map

    st.info(f"Kurse laden für {len(tickers)} Ticker … (parallel)")
    prog = st.progress(0.0)

    futures = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        for tk in tickers:
            futures.append(ex.submit(
                get_price_data_tail_intraday, tk, 3, use_tail, interval, fallback_last, exec_key, int(moc_cutoff)
            ))

        done = 0
        for tk, fut in zip(tickers, futures):
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
# Optionsketten-Aggregation (PCR/VOI)
# ─────────────────────────────────────────────────────────────
def _atm_strike(ref_px: float, strikes: np.ndarray) -> float:
    if not np.isfinite(ref_px) or strikes.size == 0:
        return np.nan
    return float(strikes[np.argmin(np.abs(strikes - ref_px))])


def _band_mask(strikes: pd.Series, atm: float, band: float) -> pd.Series:
    if not np.isfinite(atm):
        return pd.Series([False]*len(strikes), index=strikes.index)
    lo, hi = atm*(1-band), atm*(1+band)
    return strikes.between(lo, hi)


@st.cache_data(show_spinner=False, ttl=180)
def get_equity_chain_aggregates_for_today(ticker: str,
                                          ref_price: float,
                                          atm_band: float,
                                          n_exps: int,
                                          max_days: int) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options or []
    except Exception:
        exps = []
    if not exps:
        return pd.DataFrame()

    today = pd.Timestamp.today(tz=LOCAL_TZ).normalize()
    exps_filt = []
    for e in exps:
        try:
            d = pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize()
            if (d - today).days <= max_days:
                exps_filt.append((d, e))
        except Exception:
            pass
    exps_filt.sort(key=lambda x: x[0])
    exps_use = [e for _, e in exps_filt[:max(1, n_exps)]]
    if not exps_use:
        return pd.DataFrame()

    rows = []
    for e in exps_use:
        try:
            ch = tk.option_chain(e)
            calls, puts = ch.calls.copy(), ch.puts.copy()
        except Exception:
            continue

        for df in (calls, puts):
            for c in ["volume","openInterest","impliedVolatility","strike"]:
                if c not in df.columns:
                    df[c] = np.nan

        strikes = np.sort(pd.concat([calls["strike"], puts["strike"]]).dropna().unique())
        atm = _atm_strike(ref_price, strikes)

        mC = calls[_band_mask(calls["strike"], atm, atm_band)]
        mP = puts [_band_mask(puts ["strike"], atm, atm_band)]

        vC = float(np.nansum(mC["volume"])); vP = float(np.nansum(mP["volume"]))
        oiC= float(np.nansum(mC["openInterest"])); oiP= float(np.nansum(mP["openInterest"]))
        ivC= float(np.nanmean(mC["impliedVolatility"])) if len(mC) else np.nan
        ivP= float(np.nanmean(mP["impliedVolatility"])) if len(mP) else np.nan

        rows.append({"exp": e, "vol_c": vC, "vol_p": vP, "oi_c": oiC, "oi_p": oiP,
                     "voi_c": vC/max(oiC,1.0), "voi_p": vP/max(oiP,1.0),
                     "iv_c": ivC, "iv_p": ivP})

    if not rows:
        return pd.DataFrame()

    agg = pd.DataFrame(rows).agg({"vol_c":"sum","vol_p":"sum","oi_c":"sum","oi_p":"sum",
                                  "voi_c":"mean","voi_p":"mean","iv_c":"mean","iv_p":"mean"})
    out = pd.DataFrame([{
        "PCR_vol": float(agg["vol_p"]/max(agg["vol_c"],1.0)),
        "PCR_oi":  float(agg["oi_p"] /max(agg["oi_c"],1.0)),
        "VOI_call": float(agg["voi_c"]),
        "VOI_put":  float(agg["voi_p"]),
        "IV_skew_p_minus_c": float(agg["iv_p"] - agg["iv_c"]),
        "VOL_tot":  float(agg["vol_c"] + agg["vol_p"]),
        "OI_tot":   float(agg["oi_c"]  + agg["oi_p"]),
    }])
    out.index = [pd.Timestamp.today(tz=LOCAL_TZ).normalize()]
    return out


# ─────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, lookback: int, horizon: int, exog: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    feat = df.copy()
    feat["Range"]     = feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)
    feat = feat.iloc[lookback-1:].copy()
    if exog is not None and not exog.empty:
        feat = feat.join(exog, how="left").ffill()
    feat["FutureRet"] = feat["Close"].shift(-horizon) / feat["Close"] - 1
    return feat


# ─────────────────────────────────────────────────────────────
# Model Training + Backtest (A: pro Ticker separates Konto)
# ─────────────────────────────────────────────────────────────
def make_features_and_train(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: dict,
    entry_prob: float,
    exit_prob: float,
    init_capital: float,
    pos_frac: float,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
    exog_df: Optional[pd.DataFrame] = None,
    walk_forward: bool = False,
    wf_min_train: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:

    feat = make_features(df, lookback, horizon, exog=exog_df)
    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < 30:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")

    X_cols = ["Range", "SlopeHigh", "SlopeLow"]
    opt_cols = ["PCR_vol", "PCR_oi", "VOI_call", "VOI_put",
                "IV_skew_p_minus_c", "VOL_tot", "OI_tot"]
    X_cols += [c for c in opt_cols if c in feat.columns]

    hist["Target"] = (hist["FutureRet"] > threshold).astype(int)

    def make_pipe():
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(**model_params)),
        ])

    if hist["Target"].nunique() < 2:
        feat["SignalProb"] = 0.5
    else:
        if not walk_forward:
            pipe = make_pipe()
            pipe.fit(hist[X_cols].values, hist["Target"].values)
            feat["SignalProb"] = pipe.predict_proba(feat[X_cols].values)[:, 1]
        else:
            probs = np.full(len(feat), np.nan, dtype=float)
            min_train = max(int(wf_min_train), lookback + horizon + 10)

            for t in range(min_train, len(feat)):
                train = feat.iloc[:t].dropna(subset=["FutureRet"]).copy()
                if len(train) < min_train:
                    continue
                train["Target"] = (train["FutureRet"] > threshold).astype(int)
                if train["Target"].nunique() < 2:
                    continue

                pipe = make_pipe()
                pipe.fit(train[X_cols].values, train["Target"].values)
                probs[t] = pipe.predict_proba(feat[X_cols].iloc[[t]].values)[0, 1]

            feat["SignalProb"] = pd.Series(probs, index=feat.index).ffill().fillna(0.5)

    feat_bt = feat.iloc[:-1].copy()
    df_bt, trades = backtest_next_open(
        feat_bt,
        entry_prob, exit_prob,
        COMMISSION, SLIPPAGE_BPS,
        init_capital, pos_frac,
        min_hold_days=int(min_hold_days),
        cooldown_days=int(cooldown_days),
    )
    metrics = compute_performance(df_bt, trades, init_capital)
    return feat, df_bt, trades, metrics


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
                invest_net   = cash_net * float(pos_frac)
                fee_entry    = invest_net * float(commission)
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
# Performance-Kennzahlen
# ─────────────────────────────────────────────────────────────
def _cagr_from_path(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    years = len(values) / 252.0
    return (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else np.nan


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
        "InitCap (€)": float(init_cap),
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
                exit_date  = pd.to_datetime(ev["Date"])
                hold_days  = (exit_date - entry_date).days
                shares     = float(current_entry.get("Shares", 0.0))
                entry_p    = float(current_entry.get("Price", np.nan))
                exit_p     = float(ev.get("Price", np.nan))
                fee_e      = float(current_entry.get("Fees", 0.0))
                fee_x      = float(ev.get("Fees", 0.0))
                pnl_net    = float(ev.get("Net P&L", 0.0))
                cost_net   = shares * entry_p + fee_e
                ret_pct    = (pnl_net / cost_net * 100.0) if cost_net else np.nan
                rows.append({
                    "Ticker": tk, "Name": name,
                    "Entry Date": entry_date, "Exit Date": exit_date,
                    "Hold (days)": hold_days,
                    "Entry Prob": current_entry.get("Prob", np.nan),
                    "Exit Prob":  ev.get("Prob", np.nan),
                    "Shares": round(shares, 4),
                    "Entry Price": round(entry_p, 4), "Exit Price": round(exit_p, 4),
                    "PnL Net (€)": round(pnl_net, 2), "Fees (€)": round(fee_e + fee_x, 2),
                    "Return (%)": round(ret_pct, 2),
                })
                current_entry = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 🧭 Parameter-Optimierung (wie im Original)
# ─────────────────────────────────────────────────────────────
st.subheader("🧭 Parameter-Optimierung")
with st.expander("Optimizer (Random Search mit Walk-Forward-Light)", expanded=False):
    n_trials = st.number_input("Trials", 10, 1000, 80, step=10)
    seed = st.number_input("Seed", 0, 10_000, 42)
    lambda_trades = st.number_input("Penalty λ pro Trade", 0.0, 1.0, 0.02, step=0.005)
    min_trades_req = st.number_input("Min. Trades gesamt (Filter)", 0, 10000, 5, step=1)

    lb_lo, lb_hi = st.slider("Lookback", 10, 252, (30, 120), step=5)
    hz_lo, hz_hi = st.slider("Horizon", 1, 10, (3, 8))
    thr_lo, thr_hi = st.slider("Threshold Target", 0.0, 0.10, (0.035, 0.10), step=0.005, format="%.3f")
    en_lo, en_hi = st.slider("Entry Prob Range", 0.0, 1.0, (0.55, 0.85), step=0.01)
    ex_lo, ex_hi = st.slider("Exit Prob Range", 0.0, 1.0, (0.30, 0.60), step=0.01)

    @st.cache_data(show_spinner=False)
    def _get_prices_for_optimizer(tickers: tuple, start: str, end: str, use_tail: bool, interval: str,
                                  fallback_last: bool, exec_key: str, moc_cutoff: int):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fallback_last, exec_key, moc_cutoff)[0]

    def _sample_params(rng):
        return dict(
            lookback = rng.randrange(lb_lo, lb_hi+1, 5),
            horizon  = rng.randrange(hz_lo, hz_hi+1, 1),
            thresh   = rng.uniform(thr_lo, thr_hi),
            entry    = rng.uniform(en_lo, en_hi),
            exit     = rng.uniform(ex_lo, ex_hi),
        )

    if st.button("🔎 Suche starten", type="primary", use_container_width=True):
        import random
        rng = random.Random(int(seed))
        price_map_opt = _get_prices_for_optimizer(
            tuple(TICKERS), str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
        )

        rows, best = [], None
        prog = st.progress(0.0)

        feasible_tickers = [tk for tk, df in price_map_opt.items() if df is not None and len(df) >= 80]
        if not feasible_tickers:
            st.warning("Keine ausreichenden Preisdaten für Optimierung.")
        else:
            for t in range(int(n_trials)):
                p = _sample_params(rng)
                if p["exit"] >= p["entry"]:
                    prog.progress((t+1)/n_trials)
                    continue

                sharps, trades_total, feasible = [], 0, 0

                for tk in feasible_tickers:
                    df0 = price_map_opt.get(tk)
                    if df0 is None or len(df0) < max(60, p["lookback"] + p["horizon"] + 5):
                        continue
                    feasible += 1

                    mid = len(df0) // 2
                    for sub in (df0.iloc[:mid], df0.iloc[mid:]):
                        if len(sub) < max(60, p["lookback"] + p["horizon"] + 5):
                            continue
                        try:
                            _, df_bt, trades, mets = make_features_and_train(
                                sub, p["lookback"], p["horizon"], p["thresh"],
                                MODEL_PARAMS, p["entry"], p["exit"],
                                init_capital=float(INIT_CAP_PER_TICKER),
                                pos_frac=float(POS_FRAC),
                                min_hold_days=int(MIN_HOLD_DAYS),
                                cooldown_days=int(COOLDOWN_DAYS),
                                walk_forward=True,
                                wf_min_train=int(wf_min_train),
                            )
                            sharps.append(mets["Sharpe-Ratio"])
                            trades_total += int(mets["Number of Trades"])
                        except Exception:
                            pass

                if feasible == 0:
                    prog.progress((t+1)/n_trials)
                    continue

                sharpe_avg = float(np.nanmedian(sharps)) if len(sharps) else float("nan")
                denom = max(1, feasible * 2)
                trades_avg = trades_total / denom

                if trades_total < int(min_trades_req) or not np.isfinite(sharpe_avg):
                    prog.progress((t+1)/n_trials)
                    continue

                score = sharpe_avg - float(lambda_trades) * float(trades_avg)

                rec = dict(trial=t, score=score, sharpe_avg=sharpe_avg,
                           trades=trades_total, **p)
                rows.append(rec)
                if (best is None) or (score > best["score"]):
                    best = rec

                prog.progress((t+1)/n_trials)

            if not rows:
                st.warning("Keine gültigen Kandidaten gefunden.")
            else:
                df_res = pd.DataFrame(rows).sort_values("score", ascending=False)

                mask = pd.to_numeric(df_res["trades"], errors="coerce").fillna(0).astype(int) >= int(min_trades_req)
                df_show = df_res.loc[mask].copy()
                if df_show.empty:
                    df_show = df_res.copy()

                st.success(f"Beste Parameter: Score={best['score']:.3f} | Sharpe={best['sharpe_avg']:.2f} | Trades={best['trades']}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Lookback", int(best["lookback"]))
                c2.metric("Horizon",  int(best["horizon"]))
                c3.metric("Target Thresh", f"{best['thresh']:.3f}")
                c4.metric("Entry Prob",    f"{best['entry']:.2f}")
                c5.metric("Exit Prob",     f"{best['exit']:.2f}")

                st.caption("Top-Ergebnisse (Score = Sharpe − λ·Trades/(Ticker·Hälften))")
                st.dataframe(df_show.head(25), use_container_width=True)
                st.download_button("Optimierergebnisse als CSV",
                                   to_csv_eu(df_res),
                                   file_name="param_search_results.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────
# Haupt – Pipeline
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size: 36px;'>📈 PROTEUS - AI Modell</h1>", unsafe_allow_html=True)

results = []
all_trades: Dict[str, List[dict]] = {}
all_dfs:   Dict[str, pd.DataFrame] = {}
all_feat:  Dict[str, pd.DataFrame] = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min)
)

options_live: Dict[str, pd.DataFrame] = {}
if use_chain_live:
    st.info("Optionsketten je Aktie einlesen …")
    prog_opt = st.progress(0.0)
    tks = list(price_map.keys())
    for i, tk in enumerate(tks):
        try:
            df = price_map[tk]
            if df is None or df.empty:
                continue
            ref = float(df["Close"].iloc[-1])
            ch = get_equity_chain_aggregates_for_today(tk, ref, atm_band_pct, int(n_expiries), int(max_days_to_exp))
            if not ch.empty:
                options_live[tk] = ch
        except Exception:
            pass
        finally:
            prog_opt.progress((i+1)/max(1,len(tks)))

live_forecasts_run: List[dict] = []

for ticker in TICKERS:
    if ticker not in price_map:
        continue
    df = price_map[ticker]
    meta = meta_map.get(ticker, {})
    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker} — {get_ticker_name(ticker)}")
        try:
            last_timestamp_info(df, meta)

            exog_tk = None
            if use_chain_live and ticker in options_live and not options_live[ticker].empty:
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
            all_dfs[ticker] = df_bt
            all_feat[ticker] = feat

            def _decide_action_local(p: float, entry_thr: float, exit_thr: float) -> str:
                if p > entry_thr:  return "Enter / Add"
                if p < exit_thr:   return "Exit / Reduce"
                return "Hold / No Trade"

            live_ts    = pd.Timestamp(feat.index[-1])
            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            tail_info  = "intraday" if meta.get("tail_is_intraday") else "daily"

            row = {
                "AsOf": live_ts.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "Name": get_ticker_name(ticker),
                f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob, 4),
                "Action": _decide_action_local(live_prob, float(ENTRY_PROB), float(EXIT_PROB)),
                "Close": round(live_close, 4),
                "Bar": tail_info,
            }

            if use_chain_live and exog_tk is not None:
                vals = exog_tk.iloc[-1]
                for col in ["PCR_vol","PCR_oi","VOI_call","VOI_put","IV_skew_p_minus_c","VOL_tot","OI_tot"]:
                    if col in vals and pd.notna(vals[col]):
                        row[col] = round(float(vals[col]), 4)

            live_forecasts_run.append(row)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)", f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",      f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Sortino",              f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5.metric("Max DD (%)",           f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Trades (Round-Trips)", f"{int(metrics['Number of Trades'])}")

            st.caption(
                f"Entry/Exit: {'Walk-Forward' if use_walk_forward else 'In-Sample'} "
                f"| Target: FutureRet > {THRESH:.3f} (in {HORIZON}d)"
            )

            chart_cols = st.columns(2)

            df_plot = feat.copy()
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>"
            ))
            signal_probs = df_plot["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (signal_probs.max() - signal_probs.min() + 1e-9)
            for i in range(len(df_plot) - 1):
                seg_x = df_plot.index[i:i+2]
                seg_y = df_plot["Close"].iloc[i:i+2]
                color_seg = px.colors.sample_colorscale(px.colors.diverging.RdYlGn, float(norm.iloc[i]))[0]
                price_fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", showlegend=False,
                                               line=dict(color=color_seg, width=2), hoverinfo="skip"))
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"]=="Entry"]; exits = trades_df[trades_df["Typ"]=="Exit"]
                price_fig.add_trace(go.Scatter(
                    x=entries["Date"], y=entries["Price"], mode="markers", name="Entry",
                    marker_symbol="triangle-up", marker=dict(size=12, color="green"),
                    hovertemplate="Entry<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>"
                ))
                price_fig.add_trace(go.Scatter(
                    x=exits["Date"], y=exits["Price"], mode="markers", name="Exit",
                    marker_symbol="triangle-down", marker=dict(size=12, color="red"),
                    hovertemplate="Exit<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>"
                ))
            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit (Daily)",
                xaxis_title="Datum", yaxis_title="Preis",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
            with chart_cols[1]:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar (Ticker/Intervall/Zeitraum).")
                else:
                    intr_fig = go.Figure()
                    if intraday_chart_type == "Candlestick (OHLC)":
                        intr_fig.add_trace(go.Candlestick(
                            x=intra.index, open=intra["Open"], high=intra["High"],
                            low=intra["Low"],  close=intra["Close"],
                            name="OHLC (intraday)",
                            increasing_line_width=1, decreasing_line_width=1
                        ))
                    else:
                        intr_fig.add_trace(go.Scatter(
                            x=intra.index, y=intra["Close"], mode="lines", name="Close (intraday)",
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra></extra>"
                        ))
                    intr_fig.update_layout(
                        title=f"{ticker}: Intraday – letzte 5 Handelstage ({intraday_interval})",
                        xaxis_title="Zeit", yaxis_title="Preis",
                        height=420, margin=dict(t=50, b=30, l=40, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(intr_fig, use_container_width=True)

            eq = go.Figure()
            eq.add_trace(go.Scatter(x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net Equity (Next Open)",
                        mode="lines", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>"))
            bh_curve = INIT_CAP_PER_TICKER * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(x=df_bt.index, y=bh_curve, name="Buy & Hold", mode="lines",
                                    line=dict(dash="dash", color="black")))
            eq.update_layout(title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold", xaxis_title="Datum", yaxis_title="Equity (€)",
                             height=400, margin=dict(t=50, b=30, l=40, r=20),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(eq, use_container_width=True)

            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                trades_df = pd.DataFrame(trades)
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"] = ticker
                    df_tr["Name"] = get_ticker_name(ticker)
                    df_tr["Date"] = pd.to_datetime(df_tr["Date"])
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%d.%m.%Y")
                    df_tr["CumPnL"] = (
                        df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"]
                        .cumsum().ffill().fillna(0)
                    )
                    df_tr = df_tr.rename(columns={"Net P&L":"PnL","Prob":"Signal Prob","HoldDays":"Hold (days)"})

                    disp_cols = ["Ticker","Name","DateStr","Typ","Price","Shares",
                                 "Signal Prob","Hold (days)","PnL","CumPnL","Fees"]
                    styled_tr = (
                        df_tr[disp_cols].rename(columns={"DateStr":"Date"}).style.format({
                            "Price":"{:.2f}","Shares":"{:.4f}","Signal Prob":"{:.4f}",
                            "PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}"
                        })
                    )
                    show_styled_or_plain(df_tr[disp_cols].rename(columns={"DateStr":"Date"}), styled_tr)

                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=to_csv_eu(
                            df_tr[["Ticker","Name","Date","Typ","Price","Shares",
                                   "Signal Prob","Hold (days)","PnL","CumPnL","Fees"]],
                            float_format="%.4f"
                        ),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key=f"dl_trades_{ticker}",
                    )
                else:
                    st.info("Keine Trades vorhanden.")

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")


# ─────────────────────────────────────────────────────────────
# 🔮 Live-Forecast Board
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
          .drop_duplicates(subset=["Ticker"], keep="last")
          .sort_values(["AsOf", "Ticker"])
          .reset_index(drop=True)
    )

    live_df["Target_5d"] = (pd.to_numeric(live_df["Close"], errors="coerce") * (1.0 + float(THRESH))).round(2)

    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"
    if prob_col not in live_df.columns:
        cand = [c for c in live_df.columns if c.startswith("P(") and c.endswith("d)")]
        if cand:
            prob_col = cand[0]

    st.markdown(f"### 🟣 Live–Forecast Board – {HORIZON}-Tage Prognose (heute)")
    styled_live = style_live_board(live_df, prob_col, ENTRY_PROB)
    show_styled_or_plain(live_df, styled_live)

    st.download_button(
        "Live-Forecasts als CSV",
        to_csv_eu(live_df),
        file_name=f"live_forecasts_today_{HORIZON}d.csv", mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Round-Trips / Histogramme / Korrelation
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")

    # ✅ FIX 1: Phase normalisieren (damit Styling sicher trifft)
    if "Phase" in summary_df.columns:
        summary_df["Phase"] = (
            summary_df["Phase"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": ""})
            .map(lambda x: "Open" if x == "open" else ("Flat" if x == "flat" else x.capitalize()))
        )

    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / float(INIT_CAP_PER_TICKER)) * 100

    total_net_pnl   = float(summary_df["Net P&L (€)"].sum())
    total_fees      = float(summary_df["Fees (€)"].sum())
    total_gross_pnl = total_net_pnl + total_fees
    total_trades    = int(summary_df["Number of Trades"].sum())
    total_capital   = float(INIT_CAP_PER_TICKER) * len(summary_df)

    total_net_return_pct   = (total_net_pnl / total_capital * 100) if total_capital else np.nan
    total_gross_return_pct = (total_gross_pnl / total_capital * 100) if total_capital else np.nan
    bh_total_pct = float(summary_df["Buy & Hold Net (%)"].dropna().mean()) if "Buy & Hold Net (%)" in summary_df.columns else float("nan")

    st.subheader("📊 Summary of all Tickers (Next Open Backtest) — per Ticker Konto")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)",  f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)", f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)", f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades",   f"{int(total_trades)}")

    cols_pct = st.columns(4)
    cols_pct[0].metric("Strategy Net (%) – total",   f"{total_net_return_pct:.2f}")
    cols_pct[1].metric("Strategy Gross (%) – total", f"{total_gross_return_pct:.2f}")
    cols_pct[2].metric("Buy & Hold Net (%) – total", f"{bh_total_pct:.2f}")
    cols_pct[3].metric("Durchschn. CAGR (%)", f"{summary_df['CAGR (%)'].dropna().mean():.2f}" if "CAGR (%)" in summary_df else "–")

    # ✅ FIX 2: Phase Styling (Open blau)
    def phase_style(val):
        v = str(val).strip().lower()
        if v == "open":
            return "background-color:#d0ebff; color:#1f77b4; font-weight:800;"
        if v == "flat":
            return "background-color:#f0f0f0; color:#666;"
        return ""

    styled_sum = (
        summary_df.style
        .format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}",
            "Buy & Hold Net (%)":"{:.2f}","Volatility (%)":"{:.2f}",
            "Sharpe-Ratio":"{:.2f}","Sortino-Ratio":"{:.2f}",
            "Max Drawdown (%)":"{:.2f}","Calmar-Ratio":"{:.2f}",
            "Fees (€)":"{:.2f}","Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}",
            "CAGR (%)":"{:.2f}","Winrate (%)":"{:.2f}",
            "InitCap (€)":"{:.2f}"
        })
        .set_caption("Strategy-Performance per Ticker (Next Open Execution)")
    )

    if "Phase" in summary_df.columns:
        styled_sum = styled_sum.applymap(phase_style, subset=["Phase"])

    # ✅ FIX 3: immer HTML rendern (kein st.dataframe-Fallback → Farben bleiben)
    show_styled_or_plain(summary_df, styled_sum)

    st.download_button(
        "Summary als CSV herunterladen",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary.csv", mime="text/csv"
    )

    # Open Positions
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
                "Entry Date": entry_ts,
                "Entry Price": round(float(last_entry["Price"]), 2),
                "Current Prob.": round(prob, 4),
                "Unrealized PnL (€)": round(upnl, 2),
            })

    if open_positions:
        open_df = pd.DataFrame(open_positions).sort_values("Entry Date", ascending=False)
        open_df_display = open_df.copy()
        open_df_display["Entry Date"] = open_df_display["Entry Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(open_df_display, use_container_width=True)
        st.download_button("Offene Positionen als CSV", to_csv_eu(open_df), file_name="open_positions.csv", mime="text/csv")
    else:
        st.success("Keine offenen Positionen.")

    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips)")
        st.dataframe(rt_df.sort_values("Exit Date", ascending=False), use_container_width=True)
        st.download_button("Round-Trips als CSV", to_csv_eu(rt_df), file_name="round_trips.csv", mime="text/csv")

        st.markdown("### 📊 Verteilung der Round-Trip-Ergebnisse")
        bins = st.slider("Anzahl Bins", 10, 100, 30, step=5, key="rt_bins")

        ret = pd.to_numeric(rt_df.get("Return (%)"), errors="coerce").dropna()
        pnl = pd.to_numeric(rt_df.get("PnL Net (€)"), errors="coerce").dropna()

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if ret.empty:
                st.info("Keine Return-Werte vorhanden.")
            else:
                fig_ret = go.Figure(go.Histogram(x=ret, nbinsx=bins, marker_line_width=0))
                fig_ret.add_vline(x=0, line_dash="dash", opacity=0.5)
                fig_ret.update_layout(title="Histogramm: Return (%)", height=360, showlegend=False)
                st.plotly_chart(fig_ret, use_container_width=True)
        with col_h2:
            if pnl.empty:
                st.info("Keine PnL-Werte vorhanden.")
            else:
                fig_pnl = go.Figure(go.Histogram(x=pnl, nbinsx=bins, marker_line_width=0))
                fig_pnl.add_vline(x=0, line_dash="dash", opacity=0.5)
                fig_pnl.update_layout(title="Histogramm: PnL Net (€)", height=360, showlegend=False)
                st.plotly_chart(fig_pnl, use_container_width=True)

        # Korrelation
        st.markdown("### 🔗 Portfolio-Korrelation (Close-Returns)")
        price_series = []
        for tk, dfbt in all_dfs.items():
            if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
                s = dfbt["Close"].copy()
                s.name = tk
                price_series.append(s)
        
        if len(price_series) < 2:
            st.info("Mindestens zwei Ticker mit Daten nötig.")
        else:
            prices = pd.concat(price_series, axis=1, join="outer").sort_index().ffill()
            rets = prices.pct_change().dropna(how="all")
            corr = rets.corr(method="pearson")
        
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1
            )
        
            # ✅ Labels NICHT automatisch ausdünnen
            fig_corr.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(corr.columns))),
                ticktext=list(corr.columns),
                tickangle=45,
                automargin=True
            )
            fig_corr.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(corr.index))),
                ticktext=list(corr.index),
                automargin=True
            )
        
            fig_corr.update_layout(
                height=650,
                margin=dict(t=40, l=80, r=30, b=120),
                coloraxis_colorbar=dict(title="ρ")
            )
        
            st.plotly_chart(fig_corr, use_container_width=True)


else:
    st.warning("Noch keine Ergebnisse verfügbar. Prüfe Ticker-Eingaben und Datenabdeckung.")
