# streamlit_app.py  –  NEXUS v2  (alle Fixes & Verbesserungen)
#
# ── Changelog gegenüber v1 ───────────────────────────────────────────────────
#  FIX 1  Options-Features werden jetzt dynamisch in X_cols aufgenommen
#  FIX 2  Walk-Forward als Standard (kein Lookahead-Bias mehr)
#  FIX 3  Deprecated APIs: .ffill(), Styler.map(), st.query_params
#  FIX 4  Cooldown Off-by-One korrigiert  (>= → >)
#  FIX 5  Doppelte trades_df-Definition entfernt
#  NEW 1  Erweiterte Features: RSI, Ret_5d/20d, MA_ratio, Volatility, Vol_ratio
#  NEW 2  Ensemble-Modell: GBM + RandomForest + LogisticRegression
#  NEW 3  Volatilitäts-skalierte Positionsgröße (Kelly-Light)
#  NEW 4  Feature-Importance-Chart (aus GBM-Teilmodell)
# ─────────────────────────────────────────────────────────────────────────────

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

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS v2 – Signal-basierte Strategie", layout="wide"
)
LOCAL_TZ   = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 8
pd.options.display.float_format = "{:,.4f}".format

# Feste Feature-Listen (werden dynamisch gefiltert)
BASE_FEATURE_COLS = [
    "Range", "SlopeHigh", "SlopeLow",
    "Ret_5d", "Ret_20d", "MA_ratio",
    "Volatility", "RSI", "Vol_ratio",
]
OPTIONS_FEATURE_COLS = [
    "PCR_vol", "PCR_oi", "VOI_call", "VOI_put", "IV_skew_p_minus_c",
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def to_csv_eu(df: pd.DataFrame, float_format: Optional[str] = None) -> bytes:
    return df.to_csv(
        index=False, sep=";", decimal=",",
        date_format="%d.%m.%Y", float_format=float_format,
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

ticker_source = st.sidebar.selectbox(
    "Ticker-Quelle", ["Manuell (Textfeld)", "CSV-Upload"], index=0
)

tickers_final: List[str] = []
if ticker_source == "Manuell (Textfeld)":
    tickers_input = st.sidebar.text_input(
        "Tickers (Komma-getrennt)", value="REGN, LULU, VOW3.DE, REI, DDL"
    )
    tickers_final = _normalize_tickers(
        [t for t in tickers_input.split(",") if t.strip()]
    )
else:
    st.sidebar.caption(
        "Lade eine oder mehrere CSVs mit Spalte **ticker** (oder erste Spalte)."
    )
    uploads = st.sidebar.file_uploader(
        "CSV-Dateien", type=["csv"], accept_multiple_files=True
    )
    collected = []
    if uploads:
        for up in uploads:
            try:
                collected += parse_ticker_csv(up)
            except Exception as e:
                st.sidebar.error(f"Fehler beim Lesen von '{up.name}': {e}")
    base = _normalize_tickers(collected)
    extra_csv = st.sidebar.text_input(
        "Weitere Ticker manuell hinzufügen (Komma-getrennt)",
        value="", key="extra_csv", help="Beispiel: AAPL, TSLA, BABA",
    )
    extras = (
        _normalize_tickers([t for t in extra_csv.split(",") if t.strip()])
        if extra_csv else []
    )
    tickers_final = _normalize_tickers(base + extras)
    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker: {len(tickers_final)}")
        if st.sidebar.checkbox(
            "Zufällig mischen", value=False, help="Reihenfolge zufällig (seed=42)"
        ):
            import random
            random.seed(42)
            random.shuffle(tickers_final)
        max_n = st.sidebar.number_input(
            "Max. Anzahl (0 = alle)", min_value=0,
            max_value=len(tickers_final), value=0, step=10,
        )
        if max_n and max_n < len(tickers_final):
            tickers_final = tickers_final[: int(max_n)]
        tickers_final = st.sidebar.multiselect(
            "Auswahl verfeinern", options=tickers_final, default=tickers_final
        )

if not tickers_final:
    tickers_final = _normalize_tickers(["REGN", "VOW3.DE", "LULU", "REI", "DDL"])

st.sidebar.download_button(
    "Kombinierte Ticker als CSV",
    to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
    file_name="tickers_combined.csv",
    mime="text/csv",
)
TICKERS = tickers_final

# Core Backtest-Parameter
START_DATE = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2025-01-01")
)
END_DATE = st.sidebar.date_input(
    "End Date", value=pd.to_datetime(datetime.now(LOCAL_TZ).date())
)
LOOKBACK   = st.sidebar.number_input("Lookback (Tage)", 10, 252, 35, step=5)
HORIZON    = st.sidebar.number_input("Horizon (Tage)", 1, 10, 5)
THRESH     = st.sidebar.number_input(
    "Threshold für Target", 0.0, 0.1, 0.046, step=0.005, format="%.3f"
)
ENTRY_PROB = st.sidebar.slider("Entry Threshold (P(Signal))", 0.0, 1.0, 0.62, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold (P(Signal))",  0.0, 1.0, 0.48, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()

MIN_HOLD_DAYS = st.sidebar.number_input(
    "Mindesthaltedauer (Handelstage)", 0, 252, 5, step=1,
    help="Sperrt Exits, bis die Position mindestens so viele Handelstage gehalten wurde.",
)
COOLDOWN_DAYS = st.sidebar.number_input(
    "Cooling Phase nach Exit (Handelstage)", 0, 252, 0, step=1,
    help="Verhindert Neueinstiege für X Handelstage nach einem Exit (pro Ticker).",
)

COMMISSION   = st.sidebar.number_input(
    "Commission (ad valorem, z.B. 0.001=10bp)",
    0.0, 0.02, 0.004, step=0.0001, format="%.4f",
)
SLIPPAGE_BPS = st.sidebar.number_input(
    "Slippage (bp je Ausführung)", 0, 50, 5, step=1
)
POS_FRAC = st.sidebar.slider(
    "Positionsgröße (% des Kapitals)", 0.1, 1.0, 1.0, step=0.1
)
INIT_CAP = st.sidebar.number_input(
    "Initial Capital  (€)", min_value=1000.0, value=10_000.0,
    step=1000.0, format="%.2f",
)

# Intraday
use_live          = st.sidebar.checkbox("Letzten Tag intraday aggregieren (falls verfügbar)", value=True)
intraday_interval = st.sidebar.selectbox(
    "Intraday-Intervall (Tail & 5-Tage-Chart)", ["1m", "2m", "5m", "15m"], index=2
)
fallback_last_session = st.sidebar.checkbox(
    "Fallback: letzte Session verwenden (wenn heute leer)", value=False
)
exec_mode = st.sidebar.selectbox(
    "Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"]
)
moc_cutoff_min = st.sidebar.number_input(
    "MOC Cutoff (Minuten vor Close, nur live)", 5, 60, 15, step=5
)
intraday_chart_type = st.sidebar.selectbox(
    "Intraday-Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0
)

# ── NEU: Modell-Optionen ──────────────────────────────────────
st.sidebar.markdown("**Modellparameter**")
USE_WALK_FORWARD = st.sidebar.checkbox(
    "Walk-Forward (kein Lookahead-Bias)", value=True,
    help="Empfohlen: Das Modell wird immer nur auf vergangenen Daten trainiert.",
)
USE_ENSEMBLE = st.sidebar.checkbox(
    "Ensemble-Modell (GBM + RF + LR)", value=True,
    help="Robuster, aber im Walk-Forward-Modus deutlich langsamer.",
)
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 100, step=10)
learning_rate = st.sidebar.number_input(
    "learning_rate", 0.01, 1.0, 0.1, step=0.01, format="%.2f"
)
max_depth = st.sidebar.number_input("max_depth", 1, 10, 3, step=1)
MODEL_PARAMS = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42,
)

# ── NEU: Volatilitäts-Positionsgröße ─────────────────────────
st.sidebar.markdown("**Positionsgröße**")
USE_VOL_SIZING = st.sidebar.checkbox(
    "Volatilitäts-Positionsgröße (Kelly-Light)", value=False,
    help=(
        "Skaliert die Positionsgröße so, dass die annualisierte Portfolio-Volatilität "
        "annähernd der Ziel-Vola entspricht. Reduziert automatisch das Exposure "
        "bei turbulenten Märkten."
    ),
)
TARGET_VOL_ANNUAL = st.sidebar.number_input(
    "Ziel-Volatilität p.a. (%)", 5.0, 50.0, 15.0, step=1.0,
    help="Nur aktiv wenn 'Volatilitäts-Positionsgröße' aktiviert.",
) / 100.0

# Optionsdaten
st.sidebar.markdown("**Optionsdaten (Einzelaktie)**")
use_chain_live = st.sidebar.checkbox(
    "Live-Optionskette je Aktie nutzen (PCR/VOI)", value=True
)
atm_band_pct    = st.sidebar.slider("ATM-Band (±%)", 1, 15, 5, step=1) / 100.0
max_days_to_exp = st.sidebar.slider("Max. Restlaufzeit (Tage)", 7, 45, 21, step=1)
n_expiries      = st.sidebar.slider("Nächste n Verfälle", 1, 4, 2, step=1)

# Housekeeping
c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear()
    st.rerun()
if c2.button("📥 Summary CSV"):
    # FIX 3: st.experimental_set_query_params → st.query_params
    st.query_params["download"] = "summary"


# ─────────────────────────────────────────────────────────────
# Misc Helpers
# ─────────────────────────────────────────────────────────────
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


def slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0] if len(arr) >= 2 else 0.0


def last_timestamp_info(df: pd.DataFrame, meta: Optional[dict] = None):
    ts  = df.index[-1]
    msg = f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}"
    if meta and meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
        msg += f" (intraday bis {meta['tail_ts'].strftime('%H:%M %Z')})"
    st.caption(msg)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def get_ticker_name(ticker: str) -> str:
    try:
        tk   = yf.Ticker(ticker)
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
        act = str(row.get("Action_adj", row.get("Action", ""))).lower()
        if "enter" in act:
            return ["background-color: #D7F3F7"] * len(row)
        if "exit" in act:
            return ["background-color: #FFE8E8"] * len(row)
        try:
            if float(row.get(prob_col, np.nan)) >= float(entry_threshold):
                return ["background-color: #E6F7FF"] * len(row)
        except Exception:
            pass
        return ["background-color: #F7F7F7"] * len(row)

    fmt = {prob_col: "{:.4f}"}
    if "P_adj"    in df.columns: fmt["P_adj"]    = "{:.4f}"
    if "Close"    in df.columns: fmt["Close"]    = "{:.2f}"
    if "Target_5d" in df.columns: fmt["Target_5d"] = "{:.2f}"
    for c in ["PCR_oi", "PCR_vol", "VOI_call", "VOI_put"]:
        if c in df.columns: fmt[c] = "{:.4f}"

    sty = df.style.format(fmt).apply(_row_color, axis=1)
    subset_cols = [c for c in ["Action", "Action_adj"] if c in df.columns]
    if subset_cols:
        sty = sty.set_properties(subset=subset_cols, **{"font-weight": "600"})
    return sty


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def get_price_data_tail_intraday(
    ticker: str,
    years: int = 2,
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
        intraday = tk.history(
            period="1d", interval=interval,
            auto_adjust=True, actions=False, prepost=False,
        )
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
        now_local   = datetime.now(LOCAL_TZ)
        cutoff_time = now_local - timedelta(minutes=int(moc_cutoff_min_val))
        intraday    = intraday.loc[:cutoff_time]

    if intraday.empty and fallback_last_session:
        try:
            intraday5 = tk.history(
                period="5d", interval=interval,
                auto_adjust=True, actions=False, prepost=False,
            )
            if not intraday5.empty:
                if intraday5.index.tz is None:
                    intraday5.index = intraday5.index.tz_localize("UTC")
                intraday5.index = intraday5.index.tz_convert(LOCAL_TZ)
                last_session_date = intraday5.index[-1].date()
                intraday = intraday5.loc[str(last_session_date)]
        except Exception:
            pass

    if not intraday.empty:
        last_bar  = intraday.iloc[-1]
        day_key   = pd.Timestamp(last_bar.name.date(), tz=LOCAL_TZ)
        daily_row = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(last_bar["Close"]),
            "Volume": float(intraday["Volume"].sum()),
        }
        df.loc[day_key] = daily_row
        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"]          = last_bar.name

    df.dropna(subset=["High", "Low", "Close", "Open"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=120)
def get_intraday_last_n_sessions(
    ticker: str, sessions: int = 5, days_buffer: int = 10, interval: str = "5m"
) -> pd.DataFrame:
    tk   = yf.Ticker(ticker)
    intr = tk.history(
        period=f"{days_buffer}d", interval=interval,
        auto_adjust=True, actions=False, prepost=False,
    )
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr        = intr.sort_index()
    unique_dates = pd.Index(intr.index.normalize().unique())
    keep_dates   = set(unique_dates[-sessions:])
    mask         = intr.index.normalize().isin(keep_dates)
    return intr.loc[mask].copy()


def load_all_prices(
    tickers: List[str], start: str, end: str,
    use_tail: bool, interval: str, fallback_last: bool,
    exec_key: str, moc_cutoff: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map:  Dict[str, dict] = {}
    if not tickers:
        return price_map, meta_map
    st.info(f"Kurse laden für {len(tickers)} Ticker … (parallel)")
    prog = st.progress(0.0)

    futures = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        for tk in tickers:
            futures.append(
                ex.submit(
                    get_price_data_tail_intraday,
                    tk, 3, use_tail, interval, fallback_last, exec_key, int(moc_cutoff),
                )
            )
        done = 0
        for tk, fut in zip(tickers, futures):
            try:
                df_full, meta = fut.result()
                df_use        = df_full.loc[str(start):str(end)].copy()
                price_map[tk] = df_use
                meta_map[tk]  = meta
            except Exception as e:
                st.error(f"Fehler beim Laden von {tk}: {e}")
            finally:
                done += 1
                prog.progress(done / len(tickers))
    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Optionsketten-Aggregation
# ─────────────────────────────────────────────────────────────
def _atm_strike(ref_px: float, strikes: np.ndarray) -> float:
    if not np.isfinite(ref_px) or strikes.size == 0:
        return np.nan
    return float(strikes[np.argmin(np.abs(strikes - ref_px))])


def _band_mask(strikes: pd.Series, atm: float, band: float) -> pd.Series:
    if not np.isfinite(atm):
        return strikes == False  # noqa: E712
    lo, hi = atm * (1 - band), atm * (1 + band)
    return strikes.between(lo, hi)


@st.cache_data(show_spinner=False, ttl=180)
def get_equity_chain_aggregates_for_today(
    ticker: str,
    ref_price: float,
    atm_band: float,
    n_exps: int,
    max_days: int,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options or []
    except Exception:
        exps = []
    if not exps:
        return pd.DataFrame()

    today     = pd.Timestamp.today(tz=LOCAL_TZ).normalize()
    exps_filt = []
    for e in exps:
        try:
            d = pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize()
            if (d - today).days <= max_days:
                exps_filt.append((d, e))
        except Exception:
            pass
    exps_filt.sort(key=lambda x: x[0])
    exps_use = [e for _, e in exps_filt[: max(1, n_exps)]]
    if not exps_use:
        return pd.DataFrame()

    rows = []
    for e in exps_use:
        try:
            ch            = tk.option_chain(e)
            calls, puts   = ch.calls.copy(), ch.puts.copy()
        except Exception:
            continue
        for df in (calls, puts):
            for c in ["volume", "openInterest", "impliedVolatility", "strike"]:
                if c not in df.columns:
                    df[c] = np.nan

        strikes = np.sort(
            pd.concat([calls["strike"], puts["strike"]]).dropna().unique()
        )
        atm = _atm_strike(ref_price, strikes)
        mC  = calls[_band_mask(calls["strike"], atm, atm_band)]
        mP  = puts [_band_mask(puts ["strike"], atm, atm_band)]

        vC  = float(np.nansum(mC["volume"]))
        vP  = float(np.nansum(mP["volume"]))
        oiC = float(np.nansum(mC["openInterest"]))
        oiP = float(np.nansum(mP["openInterest"]))
        ivC = float(np.nanmean(mC["impliedVolatility"])) if len(mC) else np.nan
        ivP = float(np.nanmean(mP["impliedVolatility"])) if len(mP) else np.nan

        rows.append({
            "exp": e,
            "vol_c": vC, "vol_p": vP, "oi_c": oiC, "oi_p": oiP,
            "voi_c": vC / max(oiC, 1.0), "voi_p": vP / max(oiP, 1.0),
            "iv_c": ivC, "iv_p": ivP,
        })

    if not rows:
        return pd.DataFrame()
    agg = pd.DataFrame(rows).agg({
        "vol_c": "sum", "vol_p": "sum", "oi_c": "sum", "oi_p": "sum",
        "voi_c": "mean", "voi_p": "mean", "iv_c": "mean", "iv_p": "mean",
    })
    out = pd.DataFrame([{
        "PCR_vol":          agg["vol_p"] / max(agg["vol_c"], 1.0),
        "PCR_oi":           agg["oi_p"]  / max(agg["oi_c"],  1.0),
        "VOI_call":         float(agg["voi_c"]),
        "VOI_put":          float(agg["voi_p"]),
        "IV_skew_p_minus_c": float(agg["iv_p"] - agg["iv_c"]),
        "VOL_tot":          float(agg["vol_c"] + agg["vol_p"]),
        "OI_tot":           float(agg["oi_c"]  + agg["oi_p"]),
    }])
    out.index = [pd.Timestamp.today(tz=LOCAL_TZ).normalize()]
    return out


# ─────────────────────────────────────────────────────────────
# NEW 1 – Erweiterte Feature-Berechnung
# ─────────────────────────────────────────────────────────────
def make_features(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    exog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    feat = df.copy()

    # ── Ursprüngliche Features ──────────────────────────────
    feat["Range"]     = (
        feat["High"].rolling(lookback).max() - feat["Low"].rolling(lookback).min()
    )
    feat["SlopeHigh"] = feat["High"].rolling(lookback).apply(slope, raw=True)
    feat["SlopeLow"]  = feat["Low"].rolling(lookback).apply(slope, raw=True)

    # ── NEU: Momentum ────────────────────────────────────────
    feat["Ret_5d"]   = feat["Close"].pct_change(5)
    feat["Ret_20d"]  = feat["Close"].pct_change(20)
    feat["MA_ratio"] = feat["Close"] / (feat["Close"].rolling(20).mean() + 1e-9)

    # ── NEU: Volatilität (realisiert, annualisiert) ──────────
    feat["Volatility"] = feat["Close"].pct_change().rolling(lookback).std()

    # ── NEU: RSI (14 Perioden) ───────────────────────────────
    delta = feat["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    feat["RSI"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    # ── NEU: Volumen-Ratio ───────────────────────────────────
    if "Volume" in feat.columns and feat["Volume"].gt(0).any():
        vol_ma = feat["Volume"].rolling(20).mean().replace(0, np.nan)
        feat["Vol_ratio"] = feat["Volume"] / vol_ma
    else:
        feat["Vol_ratio"] = 1.0

    feat = feat.iloc[lookback - 1:].copy()

    # ── Exogene Daten (Options) ──────────────────────────────
    if exog is not None and not exog.empty:
        feat = feat.join(exog, how="left")
        # FIX 3: .fillna(method="ffill") → .ffill()
        feat = feat.ffill()

    feat["FutureRet"] = feat["Close"].shift(-horizon) / feat["Close"] - 1
    return feat


# ─────────────────────────────────────────────────────────────
# NEW 2 – Ensemble-Modell
# ─────────────────────────────────────────────────────────────
def build_ensemble(model_params: dict, use_ensemble: bool):
    """
    Gibt ein VotingClassifier-Ensemble (GBM + RF + LR) zurück.
    Falls use_ensemble=False wird nur der GBM verwendet.
    """
    gbm = GradientBoostingClassifier(**model_params)
    if not use_ensemble:
        return gbm

    rf  = RandomForestClassifier(
        n_estimators=max(50, model_params.get("n_estimators", 100) // 2),
        max_depth=model_params.get("max_depth", 4),
        random_state=42,
        n_jobs=-1,
    )
    lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)

    return VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1],   # GBM erhält höchste Gewichtung
    )


def extract_feature_importance(
    model, x_cols: List[str]
) -> Optional[pd.Series]:
    """
    Extrahiert Feature-Importances aus dem Modell.
    Bei VotingClassifier wird der GBM-Teilschätzer genutzt.
    """
    estimator = model
    if hasattr(model, "named_estimators_"):
        estimator = model.named_estimators_.get("gbm", model)
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=x_cols)
    if hasattr(estimator, "coef_"):
        return pd.Series(np.abs(estimator.coef_[0]), index=x_cols)
    return None


# ─────────────────────────────────────────────────────────────
# Features, Training & Backtest (Walk-Forward)
# ─────────────────────────────────────────────────────────────
def make_features_and_train(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    threshold: float,
    model_params: dict,
    entry_prob: float,
    exit_prob: float,
    min_hold_days: int = 0,
    cooldown_days: int = 0,
    exog_df: Optional[pd.DataFrame] = None,
    walk_forward: bool = True,         # FIX 2: jetzt True als Standard
    use_ensemble: bool = True,         # NEW 2
    use_vol_sizing: bool = False,      # NEW 3
    target_vol_annual: float = 0.15,   # NEW 3
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:

    feat = make_features(df, lookback, horizon, exog=exog_df)

    hist = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
    if len(hist) < 30:
        raise ValueError("Zu wenige Datenpunkte nach Preprocessing für das Modell.")

    # FIX 1: Dynamische X_cols – Options-Features einbeziehen wenn vorhanden
    x_cols = BASE_FEATURE_COLS + [
        c for c in OPTIONS_FEATURE_COLS
        if c in hist.columns and hist[c].notna().sum() >= 10
    ]
    # Nur vorhandene Spalten verwenden
    x_cols = [c for c in x_cols if c in hist.columns]

    last_fi: Optional[pd.Series] = None  # Feature-Importance des letzten Modells

    if not walk_forward:
        # Volltraining – KEIN Walk-Forward (Lookahead-Bias möglich!)
        hist["Target"] = (hist["FutureRet"] > threshold).astype(int)
        if hist["Target"].nunique() < 2:
            feat["SignalProb"] = 0.5
        else:
            X_train = hist[x_cols].fillna(0).values
            scaler  = StandardScaler().fit(X_train)
            model   = build_ensemble(model_params, use_ensemble)
            model.fit(scaler.transform(X_train), hist["Target"].values)
            feat["SignalProb"] = model.predict_proba(
                scaler.transform(feat[x_cols].fillna(0).values)
            )[:, 1]
            last_fi = extract_feature_importance(model, x_cols)
    else:
        # FIX 2: Walk-Forward – kein Lookahead-Bias
        probs    = np.full(len(feat), np.nan, dtype=float)
        min_train = max(lookback + horizon + 5, 40)

        for t in range(min_train, len(feat)):
            train = feat.iloc[:t].dropna(subset=["FutureRet"]).copy()
            if len(train) < min_train:
                continue
            train["Target"] = (train["FutureRet"] > threshold).astype(int)
            if train["Target"].nunique() < 2:
                continue
            X_train = train[x_cols].fillna(0).values
            scaler  = StandardScaler().fit(X_train)
            model   = build_ensemble(model_params, use_ensemble)
            model.fit(scaler.transform(X_train), train["Target"].values)
            probs[t] = model.predict_proba(
                scaler.transform(feat[x_cols].iloc[[t]].fillna(0).values)
            )[0, 1]

        feat["SignalProb"] = (
            pd.Series(probs, index=feat.index).ffill().fillna(0.5)
        )
        # Feature-Importance des letzten trainierten Modells
        last_fi = extract_feature_importance(model, x_cols)

    feat_bt = feat.iloc[:-1].copy()
    df_bt, trades = backtest_next_open(
        feat_bt, entry_prob, exit_prob,
        COMMISSION, SLIPPAGE_BPS, INIT_CAP, POS_FRAC,
        min_hold_days=int(min_hold_days),
        cooldown_days=int(cooldown_days),
        use_vol_sizing=use_vol_sizing,
        target_vol_annual=target_vol_annual,
    )
    metrics = compute_performance(df_bt, trades, INIT_CAP)
    # Feature-Importance in metrics-Dict einbetten
    metrics["_feature_importance"] = last_fi
    metrics["_x_cols"]             = x_cols
    return feat, df_bt, trades, metrics


# ─────────────────────────────────────────────────────────────
# Backtest  (NEW 3: Volatilitäts-Positionsgröße, FIX 4: Cooldown)
# ─────────────────────────────────────────────────────────────
def _vol_scaled_frac(
    vol_daily: float,
    pos_frac: float,
    target_vol_annual: float,
) -> float:
    """
    Skaliert die Positionsgröße (Kelly-Light):
    Zielt darauf ab, dass jede Position ~target_vol_annual Vola beisteuert.
    """
    if not np.isfinite(vol_daily) or vol_daily <= 1e-8:
        return pos_frac * 0.5
    vol_annual = vol_daily * sqrt(252)
    raw_frac   = target_vol_annual / max(vol_annual, 1e-6)
    return min(raw_frac * pos_frac, pos_frac)


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
    use_vol_sizing: bool = False,
    target_vol_annual: float = 0.15,
) -> Tuple[pd.DataFrame, List[dict]]:
    df = df.copy()
    n  = len(df)
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
            slip_buy   = open_today * (1 + slippage_bps / 10_000.0)
            slip_sell  = open_today * (1 - slippage_bps / 10_000.0)
            prob_prev  = float(df["SignalProb"].iloc[i - 1])
            date_exec  = df.index[i]

            # FIX 4: Cooldown Off-by-One → > statt >=
            cool_ok = True
            if (not in_pos) and cooldown_days > 0 and last_exit_idx is not None:
                bars_since_exit = i - last_exit_idx
                cool_ok = bars_since_exit > int(cooldown_days)

            # ── ENTRY ──────────────────────────────────────────
            can_enter = (not in_pos) and (prob_prev > entry_thr) and cool_ok
            if can_enter:
                # NEW 3: Positionsgröße bestimmen
                if use_vol_sizing and "Volatility" in df.columns:
                    vol_d      = float(df["Volatility"].iloc[i - 1])
                    eff_frac   = _vol_scaled_frac(vol_d, pos_frac, target_vol_annual)
                else:
                    eff_frac   = pos_frac

                invest_net    = cash_net * eff_frac
                fee_entry     = invest_net * commission
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)

                if (
                    target_shares > 0
                    and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9
                ):
                    shares           = target_shares
                    cost_basis_gross = shares * slip_buy
                    cost_basis_net   = shares * slip_buy + fee_entry
                    cash_gross      -= cost_basis_gross
                    cash_net        -= cost_basis_net
                    in_pos           = True
                    last_entry_idx   = i
                    trades.append({
                        "Date":     date_exec,
                        "Typ":      "Entry",
                        "Price":    round(slip_buy, 4),
                        "Shares":   round(shares, 4),
                        "Gross P&L": 0.0,
                        "Fees":     round(fee_entry, 2),
                        "Net P&L":  0.0,
                        "kum P&L":  round(cum_pl_net, 2),
                        "Prob":     round(prob_prev, 4),
                        "HoldDays": np.nan,
                        "PosFrac":  round(eff_frac, 4),
                    })

            # ── EXIT ───────────────────────────────────────────
            elif in_pos and prob_prev < exit_thr:
                held_bars = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_days) > 0 and held_bars < int(min_hold_days):
                    pass   # Mindesthaltedauer noch nicht erreicht
                else:
                    gross_value = shares * slip_sell
                    fee_exit    = gross_value * commission
                    pnl_gross   = gross_value - cost_basis_gross
                    pnl_net     = (gross_value - fee_exit) - cost_basis_net

                    cash_gross += gross_value
                    cash_net   += (gross_value - fee_exit)

                    in_pos           = False
                    shares           = 0.0
                    cost_basis_gross = 0.0
                    cost_basis_net   = 0.0
                    cum_pl_net      += pnl_net

                    trades.append({
                        "Date":      date_exec,
                        "Typ":       "Exit",
                        "Price":     round(slip_sell, 4),
                        "Shares":    0.0,
                        "Gross P&L": round(pnl_gross, 2),
                        "Fees":      round(fee_exit, 2),
                        "Net P&L":   round(pnl_net, 2),
                        "kum P&L":   round(cum_pl_net, 2),
                        "Prob":      round(prob_prev, 4),
                        "HoldDays":  int(held_bars),
                        "PosFrac":   np.nan,
                    })
                    last_exit_idx  = i
                    last_entry_idx = None

        close_today = float(df["Close"].iloc[i])
        equity_gross.append(cash_gross + (shares * close_today if in_pos else 0.0))
        equity_net.append(cash_net   + (shares * close_today if in_pos else 0.0))

    df_bt                = df.copy()
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
    if years <= 0:
        return np.nan
    v0, v1 = float(values.iloc[0]), float(values.iloc[-1])
    if v0 <= 0 or not np.isfinite(v0) or not np.isfinite(v1):
        return np.nan
    return (v1 / v0) ** (1.0 / years) - 1.0


def _sortino(rets: pd.Series) -> float:
    if rets.empty:
        return np.nan
    mean     = rets.mean() * 252
    downside = rets[rets < 0]
    dd       = downside.std() * sqrt(252) if len(downside) else np.nan
    return mean / dd if dd and np.isfinite(dd) and dd > 0 else np.nan


def _winrate_roundtrips(trades: List[dict]) -> float:
    if not trades:
        return np.nan
    pnl   = []
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


def compute_performance(
    df_bt: pd.DataFrame, trades: List[dict], init_cap: float
) -> dict:
    net_ret  = (df_bt["Equity_Net"].iloc[-1] / init_cap - 1) * 100
    rets     = df_bt["Equity_Net"].pct_change().dropna()
    vol_ann  = rets.std() * sqrt(252) * 100
    sharpe   = (rets.mean() * sqrt(252)) / (rets.std() + 1e-12)
    dd       = (
        (df_bt["Equity_Net"] - df_bt["Equity_Net"].cummax())
        / df_bt["Equity_Net"].cummax()
    )
    max_dd   = dd.min() * 100
    calmar   = (net_ret / 100) / abs(max_dd / 100) if max_dd < 0 else np.nan
    gross_ret = (df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100
    bh_ret   = (df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100
    fees     = sum(t["Fees"] for t in trades)
    phase    = "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat"
    completed = sum(1 for t in trades if t["Typ"] == "Exit")
    net_eur  = df_bt["Equity_Net"].iloc[-1] - init_cap
    cagr     = _cagr_from_path(df_bt["Equity_Net"])
    sortino  = _sortino(rets)
    winrate  = _winrate_roundtrips(trades)

    return {
        "Strategy Net (%)":   round(net_ret, 2),
        "Strategy Gross (%)": round(gross_ret, 2),
        "Buy & Hold Net (%)": round(bh_ret, 2),
        "Volatility (%)":     round(vol_ann, 2),
        "Sharpe-Ratio":       round(sharpe, 2),
        "Sortino-Ratio":      round(sortino, 2) if np.isfinite(sortino) else np.nan,
        "Max Drawdown (%)":   round(max_dd, 2),
        "Calmar-Ratio":       round(calmar, 2) if np.isfinite(calmar) else np.nan,
        "Fees (€)":           round(fees, 2),
        "Phase":              phase,
        "Number of Trades":   completed,
        "Net P&L (€)":        round(net_eur, 2),
        "CAGR (%)":           round(100 * (cagr if np.isfinite(cagr) else np.nan), 2),
        "Winrate (%)":        round(100 * (winrate if np.isfinite(winrate) else np.nan), 2),
    }


def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name          = get_ticker_name(tk)
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
                    "Ticker":       tk,
                    "Name":         name,
                    "Entry Date":   entry_date,
                    "Exit Date":    exit_date,
                    "Hold (days)":  hold_days,
                    "Entry Prob":   current_entry.get("Prob", np.nan),
                    "Exit Prob":    ev.get("Prob", np.nan),
                    "PosFrac":      current_entry.get("PosFrac", np.nan),
                    "Shares":       round(shares, 4),
                    "Entry Price":  round(entry_p, 4),
                    "Exit Price":   round(exit_p, 4),
                    "PnL Net (€)":  round(pnl_net, 2),
                    "Fees (€)":     round(fee_e + fee_x, 2),
                    "Return (%)":   round(ret_pct, 2),
                })
                current_entry = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# NEW 4 – Feature-Importance-Chart
# ─────────────────────────────────────────────────────────────
def show_feature_importance(fi: Optional[pd.Series], x_cols: List[str]) -> None:
    if fi is None or fi.empty:
        st.caption("Feature-Importance nicht verfügbar (LR-Fallback oder zu wenig Daten).")
        return
    fi_sorted = fi.sort_values(ascending=True)
    colors    = [
        "#EF4444" if c in OPTIONS_FEATURE_COLS else "#6366F1"
        for c in fi_sorted.index
    ]
    fig = go.Figure(
        go.Bar(
            x=fi_sorted.values,
            y=fi_sorted.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Feature Importance (GBM-Teilmodell) – Rot = Options-Feature",
        xaxis_title="Importance",
        height=max(300, len(fi_sorted) * 28),
        margin=dict(t=45, l=160, r=20, b=35),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 🧭 Parameter-Optimierung
# ─────────────────────────────────────────────────────────────
# Score-Funktion (trade-count-unabhängig):
#   Score = Sharpe × Winrate × (1 + CAGR_norm) − w_dd × |MaxDD|
#   Alle Komponenten werden als Median über Walk-Forward-Hälften
#   und alle Ticker aggregiert → robust gegen Einzelausreißer.
# ─────────────────────────────────────────────────────────────

def _composite_score(
    sharpe:   float,
    winrate:  float,   # 0–1
    cagr:     float,   # z.B. 0.12 für 12%
    max_dd:   float,   # negativ, z.B. -0.18 für 18% DD
    w_dd:     float,   # Gewicht für DD-Abzug (Sidebar)
) -> float:
    """
    Kombinierter Score ohne jegliche Trade-Count-Bedingung.

    Logik:
      - Sharpe:  Kern-Metrik (risikoadjustierte Rendite)
      - Winrate: Multiplikator – filtert Glücks-Strategien
                 (1 großer Win, viele Verluste)
      - CAGR:   Bonus für tatsächliches Kapitalwachstum,
                normiert auf [-1, +∞) damit 0% CAGR neutral ist
      - MaxDD:  Direkter Abzug – hohe Drawdowns werden
                unabhängig vom Sharpe bestraft
    """
    if not (np.isfinite(sharpe) and np.isfinite(winrate)
            and np.isfinite(cagr) and np.isfinite(max_dd)):
        return float("-inf")
    # winrate als Faktor: 0.5 = neutral, >0.5 = Bonus, <0.5 = Malus
    wr_factor   = winrate * 2.0          # 0→0, 0.5→1.0, 1→2.0
    cagr_bonus  = 1.0 + max(cagr, -1.0) # nie unter 0
    dd_penalty  = w_dd * abs(max_dd)     # max_dd ist negativ
    return sharpe * wr_factor * cagr_bonus - dd_penalty


st.subheader("🧭 Parameter-Optimierung")
with st.expander("Optimizer (Random Search · Composite Score · Walk-Forward)", expanded=False):

    st.markdown(
        "**Score = Sharpe × (2 × Winrate) × (1 + CAGR) − w_DD × |MaxDD|**  \n"
        "Keine Trade-Anzahl-Bedingung – die Strategie wird rein nach "
        "Qualität der Ergebnisse bewertet.",
        help=(
            "Sharpe: Kern-Metrik\n"
            "Winrate: Multiplikator (>50% = Bonus)\n"
            "CAGR: Wachstumsbonus\n"
            "MaxDD: Drawdown-Abzug (Gewicht w_DD einstellbar)"
        ),
    )

    # ── Score-Gewichte ────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        w_dd_penalty = st.number_input(
            "Drawdown-Gewicht w_DD",
            min_value=0.0, max_value=10.0, value=1.5, step=0.1, format="%.1f",
            help=(
                "Wie stark wird ein hoher Max-Drawdown bestraft?\n"
                "0 = ignorieren · 1 = linear · 2 = doppelt gewichtet"
            ),
        )
    with sc2:
        n_trials = st.number_input("Trials", 10, 2000, 150, step=10)

    seed = st.number_input("Seed (Reproduzierbarkeit)", 0, 10_000, 42)

    # ── Suchraum ──────────────────────────────────────────────
    st.markdown("**Suchraum**")
    lb_lo,  lb_hi  = st.slider("Lookback",         10, 252, (20, 120), step=5)
    hz_lo,  hz_hi  = st.slider("Horizon",          1,  10,  (3,  8))
    thr_lo, thr_hi = st.slider(
        "Threshold Target", 0.0, 0.10, (0.030, 0.10), step=0.005, format="%.3f"
    )
    en_lo, en_hi = st.slider("Entry Prob Range", 0.0, 1.0, (0.50, 0.90), step=0.01)
    ex_lo, ex_hi = st.slider("Exit Prob Range",  0.0, 1.0, (0.25, 0.65), step=0.01)

    @st.cache_data(show_spinner=False)
    def _get_prices_for_optimizer(
        tickers: tuple, start: str, end: str,
        use_tail: bool, interval: str,
        fallback_last: bool, exec_key: str, moc_cutoff: int,
    ):
        return load_all_prices(
            list(tickers), start, end, use_tail, interval,
            fallback_last, exec_key, moc_cutoff,
        )[0]

    def _sample_params(rng):
        return dict(
            lookback=rng.randrange(lb_lo, lb_hi + 1, 5),
            horizon =rng.randrange(hz_lo, hz_hi + 1, 1),
            thresh  =rng.uniform(thr_lo, thr_hi),
            entry   =rng.uniform(en_lo,  en_hi),
            exit    =rng.uniform(ex_lo,  ex_hi),
        )

    if st.button("🔎 Suche starten", type="primary", use_container_width=True):
        import random
        rng = random.Random(int(seed))
        price_map_opt = _get_prices_for_optimizer(
            tuple(TICKERS), str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session,
            exec_mode, int(moc_cutoff_min),
        )
        rows_opt, best_opt = [], None
        prog_opt  = st.progress(0.0)
        status_tx = st.empty()

        feasible_tickers = [
            tk for tk, df in price_map_opt.items()
            if df is not None and len(df) >= 80
        ]
        if not feasible_tickers:
            st.warning("Keine ausreichenden Preisdaten für Optimierung.")
        else:
            for t in range(int(n_trials)):
                p = _sample_params(rng)

                # Ungültige Kombination überspringen
                if p["exit"] >= p["entry"]:
                    prog_opt.progress((t + 1) / n_trials)
                    continue

                # Metriken über alle Ticker und beide Hälften sammeln
                sharpes, winrates, cagrs, dds = [], [], [], []
                trades_total = 0
                feasible     = 0

                for tk in feasible_tickers:
                    df = price_map_opt.get(tk)
                    min_len = max(60, p["lookback"] + p["horizon"] + 5)
                    if df is None or len(df) < min_len:
                        continue
                    feasible += 1
                    mid = len(df) // 2

                    for sub in (df.iloc[:mid], df.iloc[mid:]):
                        if len(sub) < min_len:
                            continue
                        try:
                            _, df_bt, trades_sub, mets = make_features_and_train(
                                sub, p["lookback"], p["horizon"], p["thresh"],
                                MODEL_PARAMS, p["entry"], p["exit"],
                                min_hold_days=int(MIN_HOLD_DAYS),
                                cooldown_days=int(COOLDOWN_DAYS),
                                walk_forward=True,
                                use_ensemble=False,   # Geschwindigkeit im Optimizer
                                use_vol_sizing=USE_VOL_SIZING,
                                target_vol_annual=TARGET_VOL_ANNUAL,
                            )
                            # Alle Score-Komponenten sammeln
                            sharpes.append(mets["Sharpe-Ratio"])
                            wr = mets.get("Winrate (%)", np.nan)
                            winrates.append(wr / 100.0 if np.isfinite(wr) else np.nan)
                            cg = mets.get("CAGR (%)", np.nan)
                            cagrs.append(cg / 100.0 if np.isfinite(cg) else np.nan)
                            dds.append(mets.get("Max Drawdown (%)", np.nan) / 100.0)
                            trades_total += int(mets["Number of Trades"])
                        except Exception:
                            pass

                if feasible == 0 or not sharpes:
                    prog_opt.progress((t + 1) / n_trials)
                    continue

                # Median je Komponente → robust gegen Ausreißer
                sh_med  = float(np.nanmedian(sharpes))
                wr_med  = float(np.nanmedian(winrates))
                cg_med  = float(np.nanmedian(cagrs))
                dd_med  = float(np.nanmedian(dds))

                # Composite Score (trade-count-unabhängig)
                score = _composite_score(
                    sharpe  = sh_med,
                    winrate = wr_med,
                    cagr    = cg_med,
                    max_dd  = dd_med,
                    w_dd    = float(w_dd_penalty),
                )

                if not np.isfinite(score):
                    prog_opt.progress((t + 1) / n_trials)
                    continue

                rec = dict(
                    trial       = t,
                    score       = round(score, 4),
                    sharpe_med  = round(sh_med, 3),
                    winrate_med = round(wr_med * 100, 1),
                    cagr_med    = round(cg_med * 100, 2),
                    maxdd_med   = round(dd_med * 100, 2),
                    trades      = trades_total,
                    **p,
                )
                rows_opt.append(rec)
                if best_opt is None or score > best_opt["score"]:
                    best_opt = rec

                status_tx.caption(
                    f"Trial {t+1}/{int(n_trials)} | "
                    f"Bester Score bisher: {best_opt['score']:.3f}"
                )
                prog_opt.progress((t + 1) / n_trials)

            status_tx.empty()

            if not rows_opt:
                st.warning("Keine gültigen Kandidaten gefunden.")
            else:
                df_res = (
                    pd.DataFrame(rows_opt)
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )

                # ── Ergebnis-Banner ───────────────────────────
                st.success(
                    f"✅  Beste Parameter gefunden — "
                    f"Score **{best_opt['score']:.3f}** | "
                    f"Sharpe **{best_opt['sharpe_med']:.2f}** | "
                    f"Winrate **{best_opt['winrate_med']:.1f}%** | "
                    f"CAGR **{best_opt['cagr_med']:.1f}%** | "
                    f"MaxDD **{best_opt['maxdd_med']:.1f}%** | "
                    f"Trades **{best_opt['trades']}**"
                )

                # ── Parameter-Kacheln ─────────────────────────
                st.markdown("**▶ Optimale Parameter – direkt übernehmen:**")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Lookback",      int(best_opt["lookback"]))
                c2.metric("Horizon",       int(best_opt["horizon"]))
                c3.metric("Threshold",     f"{best_opt['thresh']:.3f}")
                c4.metric("Entry Prob",    f"{best_opt['entry']:.2f}")
                c5.metric("Exit Prob",     f"{best_opt['exit']:.2f}")

                # ── Score-Komponenten Visualisierung ──────────
                st.markdown("### 📈 Score-Komponenten (Top 25)")
                top25 = df_res.head(25).copy()

                fig_sc = go.Figure()
                fig_sc.add_trace(go.Bar(
                    name="Sharpe (Median)",
                    x=top25["trial"].astype(str),
                    y=top25["sharpe_med"],
                    marker_color="#6366F1",
                ))
                fig_sc.add_trace(go.Bar(
                    name="Winrate % (Median)",
                    x=top25["trial"].astype(str),
                    y=top25["winrate_med"],
                    marker_color="#10B981",
                    visible="legendonly",
                ))
                fig_sc.add_trace(go.Bar(
                    name="CAGR % (Median)",
                    x=top25["trial"].astype(str),
                    y=top25["cagr_med"],
                    marker_color="#F59E0B",
                    visible="legendonly",
                ))
                fig_sc.add_trace(go.Bar(
                    name="MaxDD % (Median, negativ gut)",
                    x=top25["trial"].astype(str),
                    y=top25["maxdd_med"],
                    marker_color="#EF4444",
                    visible="legendonly",
                ))
                fig_sc.add_trace(go.Scatter(
                    name="Composite Score",
                    x=top25["trial"].astype(str),
                    y=top25["score"],
                    mode="lines+markers",
                    line=dict(color="black", width=2),
                    marker=dict(size=7),
                ))
                fig_sc.update_layout(
                    barmode="group",
                    height=380,
                    margin=dict(t=30, b=40, l=40, r=20),
                    legend=dict(orientation="h", y=-0.25),
                    xaxis_title="Trial-Nr.",
                    yaxis_title="Wert",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                # ── Scatter: Sharpe vs Winrate ─────────────────
                st.markdown("### 🔵 Sharpe vs. Winrate (alle Trials)")
                fig_sc2 = px.scatter(
                    df_res,
                    x="sharpe_med", y="winrate_med",
                    color="score",
                    size=df_res["score"].clip(lower=0) + 0.01,
                    hover_data=["trial", "lookback", "horizon",
                                "thresh", "entry", "exit",
                                "cagr_med", "maxdd_med", "trades"],
                    color_continuous_scale="RdYlGn",
                    labels={
                        "sharpe_med":  "Sharpe (Median)",
                        "winrate_med": "Winrate % (Median)",
                        "score":       "Score",
                    },
                    title="Sharpe vs. Winrate – Farbe = Composite Score",
                )
                fig_sc2.update_layout(
                    height=420, margin=dict(t=45, b=40, l=40, r=20)
                )
                st.plotly_chart(fig_sc2, use_container_width=True)

                # ── Ergebnis-Tabelle ──────────────────────────
                st.caption(
                    "Score = Sharpe × (2·Winrate) × (1+CAGR) − w_DD·|MaxDD| "
                    f"(w_DD={w_dd_penalty:.1f}) · Keine Trade-Anzahl-Bedingung"
                )
                display_cols = [
                    "trial", "score", "sharpe_med", "winrate_med",
                    "cagr_med", "maxdd_med", "trades",
                    "lookback", "horizon", "thresh", "entry", "exit",
                ]
                st.dataframe(
                    df_res[display_cols].head(25).style.format({
                        "score":        "{:.4f}",
                        "sharpe_med":   "{:.3f}",
                        "winrate_med":  "{:.1f}",
                        "cagr_med":     "{:.2f}",
                        "maxdd_med":    "{:.2f}",
                        "thresh":       "{:.4f}",
                        "entry":        "{:.3f}",
                        "exit":         "{:.3f}",
                    }).background_gradient(subset=["score"], cmap="RdYlGn"),
                    use_container_width=True,
                )
                st.download_button(
                    "Optimierergebnisse als CSV",
                    to_csv_eu(df_res[display_cols]),
                    file_name="param_search_results.csv",
                    mime="text/csv",
                )


# ─────────────────────────────────────────────────────────────
# Haupt-Pipeline
# ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:36px;'>📈 PROTEUS – AI Modell (v2)</h1>",
    unsafe_allow_html=True,
)

mode_label = "Walk-Forward ✓" if USE_WALK_FORWARD else "Volltraining ⚠️"
ens_label  = "Ensemble (GBM+RF+LR)" if USE_ENSEMBLE else "Single GBM"
vs_label   = f"Vol-Sizing ({TARGET_VOL_ANNUAL*100:.0f}% Ziel)" if USE_VOL_SIZING else "Fixe Positionsgröße"
st.caption(f"Modus: **{mode_label}** | Modell: **{ens_label}** | Sizing: **{vs_label}**")

results: List[dict]                  = []
all_trades: Dict[str, List[dict]]    = {}
all_dfs:    Dict[str, pd.DataFrame]  = {}
all_feat:   Dict[str, pd.DataFrame]  = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session,
    exec_mode, int(moc_cutoff_min),
)

# Options-Aggregate
options_live: Dict[str, pd.DataFrame] = {}
if use_chain_live:
    st.info("Optionsketten je Aktie einlesen …")
    prog_opt2 = st.progress(0.0)
    tks       = list(price_map.keys())
    for i, tk in enumerate(tks):
        try:
            df_tk = price_map[tk]
            if df_tk is None or df_tk.empty:
                continue
            ref = float(df_tk["Close"].iloc[-1])
            ch  = get_equity_chain_aggregates_for_today(
                tk, ref, atm_band_pct, int(n_expiries), int(max_days_to_exp)
            )
            if not ch.empty:
                options_live[tk] = ch
        except Exception:
            pass
        finally:
            prog_opt2.progress((i + 1) / max(1, len(tks)))

live_forecasts_run: List[dict] = []

for ticker in TICKERS:
    if ticker not in price_map:
        continue
    df    = price_map[ticker]
    meta  = meta_map.get(ticker, {})

    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker} — {get_ticker_name(ticker)}")
        try:
            last_timestamp_info(df, meta)

            exog_tk = None
            if (
                use_chain_live
                and ticker in options_live
                and not options_live[ticker].empty
            ):
                ch = options_live[ticker].copy()
                ch.index = [df.index[-1].normalize()]
                exog_tk  = ch

            feat, df_bt, trades, metrics = make_features_and_train(
                df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS,
                ENTRY_PROB, EXIT_PROB,
                min_hold_days=int(MIN_HOLD_DAYS),
                cooldown_days=int(COOLDOWN_DAYS),
                exog_df=exog_tk,
                walk_forward=USE_WALK_FORWARD,
                use_ensemble=USE_ENSEMBLE,
                use_vol_sizing=USE_VOL_SIZING,
                target_vol_annual=TARGET_VOL_ANNUAL,
            )
            # Interne Felder aus metrics-Dict herausziehen
            last_fi = metrics.pop("_feature_importance", None)
            x_cols  = metrics.pop("_x_cols", BASE_FEATURE_COLS)

            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker]    = df_bt
            all_feat[ticker]   = feat

            def _decide_action_local(p: float, entry_thr: float, exit_thr: float) -> str:
                if p > entry_thr:   return "Enter / Add"
                if p < exit_thr:    return "Exit / Reduce"
                return "Hold / No Trade"

            live_ts    = pd.Timestamp(feat.index[-1])
            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            tail_info  = "intraday" if meta.get("tail_is_intraday") else "daily"

            row = {
                "AsOf":   live_ts.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker,
                "Name":   get_ticker_name(ticker),
                f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob, 4),
                "Action": _decide_action_local(live_prob, ENTRY_PROB, EXIT_PROB),
                "Close":  round(live_close, 4),
                "Bar":    tail_info,
            }
            if use_chain_live and exog_tk is not None:
                vals = exog_tk.iloc[-1]
                for col in [
                    "PCR_vol", "PCR_oi", "VOI_call", "VOI_put",
                    "IV_skew_p_minus_c", "VOL_tot", "OI_tot",
                ]:
                    if col in vals and pd.notna(vals[col]):
                        row[col] = round(float(vals[col]), 4)
            live_forecasts_run.append(row)

            # ── KPI Tiles ───────────────────────────────────────
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)",  f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",       f"{metrics['Buy & Hold Net (%)']:.2f}")
            c3.metric("Sharpe",               f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric(
                "Sortino",
                f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–",
            )
            c5.metric("Max DD (%)",           f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Trades (Round-Trips)", f"{int(metrics['Number of Trades'])}")

            # ── Charts ──────────────────────────────────────────
            chart_cols = st.columns(2)

            # Preis + Signale (farbkodiert)
            df_plot    = feat.copy()
            price_fig  = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"],
                mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            ))
            signal_probs = df_plot["SignalProb"]
            norm = (signal_probs - signal_probs.min()) / (
                signal_probs.max() - signal_probs.min() + 1e-9
            )
            for idx in range(len(df_plot) - 1):
                seg_x = df_plot.index[idx: idx + 2]
                seg_y = df_plot["Close"].iloc[idx: idx + 2]
                color_seg = px.colors.sample_colorscale(
                    px.colors.diverging.RdYlGn, float(norm.iloc[idx])
                )[0]
                price_fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", showlegend=False,
                    line=dict(color=color_seg, width=2), hoverinfo="skip",
                ))
            # FIX 5: trades_df nur einmal definieren
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                entries = trades_df[trades_df["Typ"] == "Entry"]
                exits   = trades_df[trades_df["Typ"] == "Exit"]
                price_fig.add_trace(go.Scatter(
                    x=entries["Date"], y=entries["Price"],
                    mode="markers", name="Entry",
                    marker_symbol="triangle-up",
                    marker=dict(size=12, color="green"),
                    hovertemplate="Entry<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>",
                ))
                price_fig.add_trace(go.Scatter(
                    x=exits["Date"], y=exits["Price"],
                    mode="markers", name="Exit",
                    marker_symbol="triangle-down",
                    marker=dict(size=12, color="red"),
                    hovertemplate="Exit<br>Datum:%{x|%Y-%m-%d}<br>Preis:%{y:.2f}<extra></extra>",
                ))
            price_fig.update_layout(
                title=f"{ticker}: Preis mit Signal-Wahrscheinlichkeit (Daily)",
                xaxis_title="Datum", yaxis_title="Preis",
                height=420, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            # Intraday (letzte 5 Handelstage)
            intra = get_intraday_last_n_sessions(
                ticker, sessions=5, days_buffer=10, interval=intraday_interval
            )
            with chart_cols[1]:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar (Ticker/Intervall/Zeitraum).")
                else:
                    intr_fig = go.Figure()
                    if intraday_chart_type == "Candlestick (OHLC)":
                        intr_fig.add_trace(go.Candlestick(
                            x=intra.index,
                            open=intra["Open"], high=intra["High"],
                            low=intra["Low"],  close=intra["Close"],
                            name="OHLC (intraday)",
                            increasing_line_width=1, decreasing_line_width=1,
                        ))
                    else:
                        intr_fig.add_trace(go.Scatter(
                            x=intra.index, y=intra["Close"],
                            mode="lines", name="Close (intraday)",
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: %{y:.2f}<extra></extra>",
                        ))
                    if not trades_df.empty:
                        tdf        = trades_df.copy()
                        last_days  = set(pd.Index(intra.index.normalize().unique()))
                        ev_recent  = tdf[tdf["Date"].dt.normalize().isin(last_days)].copy()
                        for typ, color, symbol in [
                            ("Entry", "green", "triangle-up"),
                            ("Exit",  "red",   "triangle-down"),
                        ]:
                            xs, ys = [], []
                            for d, day_slice in intra.groupby(intra.index.normalize()):
                                hit = ev_recent[
                                    (ev_recent["Typ"] == typ)
                                    & (ev_recent["Date"].dt.normalize() == d)
                                ]
                                if hit.empty:
                                    continue
                                ts0   = day_slice.index.min()
                                y_val = (
                                    float(hit["Price"].iloc[-1])
                                    if intraday_chart_type == "Candlestick (OHLC)"
                                    else float(day_slice["Close"].iloc[0])
                                )
                                xs.append(ts0)
                                ys.append(y_val)
                            if xs:
                                intr_fig.add_trace(go.Scatter(
                                    x=xs, y=ys, mode="markers", name=typ,
                                    marker_symbol=symbol,
                                    marker=dict(size=11, color=color),
                                    hovertemplate=(
                                        f"{typ}<br>%{{x|%Y-%m-%d %H:%M}}"
                                        "<br>Preis: %{y:.2f}<extra></extra>"
                                    ),
                                ))
                    intr_fig.update_layout(
                        title=f"{ticker}: Intraday – letzte 5 Handelstage ({intraday_interval})",
                        xaxis_title="Zeit", yaxis_title="Preis",
                        height=420, margin=dict(t=50, b=30, l=40, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    for _, day_slice in intra.groupby(intra.index.normalize()):
                        intr_fig.add_vline(
                            x=day_slice.index.min(),
                            line_width=1, line_dash="dot", opacity=0.3,
                        )
                    st.plotly_chart(intr_fig, use_container_width=True)

            # Equity-Kurve
            eq = go.Figure()
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=df_bt["Equity_Net"],
                name="Strategy Net Equity (Next Open)",
                mode="lines",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>",
            ))
            bh_curve = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=bh_curve,
                name="Buy & Hold", mode="lines",
                line=dict(dash="dash", color="black"),
            ))
            eq.update_layout(
                title=f"{ticker}: Net Equity-Kurve vs. Buy & Hold",
                xaxis_title="Datum", yaxis_title="Equity (€)",
                height=400, margin=dict(t=50, b=30, l=40, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(eq, use_container_width=True)

            # NEW 4 – Feature Importance
            with st.expander("📊 Feature Importance", expanded=False):
                show_feature_importance(last_fi, x_cols)
                if x_cols:
                    st.caption(
                        f"Verwendete Features ({len(x_cols)}): {', '.join(x_cols)}"
                    )

            # Trades-Tabelle
            with st.expander(f"Trades (Next Open) für {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"]  = ticker
                    df_tr["Name"]    = get_ticker_name(ticker)
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%d.%m.%Y")
                    # FIX 3: .fillna(method="ffill") → .ffill()
                    df_tr["CumPnL"] = (
                        df_tr.where(df_tr["Typ"] == "Exit")["Net P&L"]
                        .cumsum().ffill().fillna(0)
                    )
                    df_tr = df_tr.rename(columns={
                        "Net P&L": "PnL",
                        "Prob":    "Signal Prob",
                        "HoldDays": "Hold (days)",
                        "PosFrac":  "Pos Frac",
                    })

                    disp_cols = [
                        "Ticker", "Name", "DateStr", "Typ", "Price",
                        "Shares", "Signal Prob", "Hold (days)", "Pos Frac",
                        "PnL", "CumPnL", "Fees",
                    ]
                    disp_cols = [c for c in disp_cols if c in df_tr.columns]
                    styled = df_tr[disp_cols].rename(
                        columns={"DateStr": "Date"}
                    ).style.format({
                        "Price":      "{:.2f}",
                        "Shares":     "{:.4f}",
                        "Signal Prob": "{:.4f}",
                        "Pos Frac":   "{:.4f}",
                        "PnL":        "{:.2f}",
                        "CumPnL":     "{:.2f}",
                        "Fees":       "{:.2f}",
                    })
                    show_styled_or_plain(
                        df_tr[disp_cols].rename(columns={"DateStr": "Date"}), styled
                    )
                    st.download_button(
                        label=f"Trades {ticker} als CSV",
                        data=to_csv_eu(
                            df_tr[[
                                "Ticker", "Name", "Date", "Typ", "Price", "Shares",
                                "Signal Prob", "Hold (days)", "Pos Frac",
                                "PnL", "CumPnL", "Fees",
                            ] if "Pos Frac" in df_tr.columns else [
                                "Ticker", "Name", "Date", "Typ", "Price", "Shares",
                                "Signal Prob", "Hold (days)", "PnL", "CumPnL", "Fees",
                            ]],
                            float_format="%.4f",
                        ),
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key=f"dl_trades_{ticker}",
                    )
                else:
                    st.info("Keine Trades vorhanden.")

        except Exception as e:
            st.error(f"Fehler bei {ticker}: {e}")
            import traceback
            st.caption(traceback.format_exc())


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
    live_df["Target_5d"] = (
        pd.to_numeric(live_df["Close"], errors="coerce") * (1.0 + float(THRESH))
    ).round(2)

    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"
    if prob_col not in live_df.columns:
        cand = [c for c in live_df.columns if c.startswith("P(") and c.endswith("d)")]
        if cand:
            prob_col = cand[0]

    if use_chain_live:
        for c in ["PCR_oi", "PCR_vol", "VOI_call", "VOI_put"]:
            if c in live_df.columns:
                s = pd.to_numeric(live_df[c], errors="coerce")
                live_df[c]       = s
                live_df[c + "_z"] = (s - s.mean()) / (s.std(ddof=0) + 1e-9)

        def col_or_zero(name: str) -> pd.Series:
            return (
                pd.to_numeric(live_df[name], errors="coerce")
                if name in live_df.columns
                else pd.Series(0.0, index=live_df.index)
            )

        comp = (
            -0.6 * col_or_zero("PCR_oi_z").fillna(0.0)
            - 0.3 * col_or_zero("PCR_vol_z").fillna(0.0)
            + 0.5 * (
                col_or_zero("VOI_call_z").fillna(0.0)
                - col_or_zero("VOI_put_z").fillna(0.0)
            )
        )
        p_base = pd.to_numeric(live_df[prob_col], errors="coerce").fillna(0.0)
        live_df["P_adj"]      = np.clip(p_base + 0.07 * comp, 0.0, 1.0)
        live_df["Action_adj"] = live_df["P_adj"].apply(
            lambda p: (
                "Enter / Add" if p >= ENTRY_PROB
                else ("Exit / Reduce" if p <= EXIT_PROB else "Hold / No Trade")
            )
        )
        desired = [
            "AsOf", "Ticker", "Name", prob_col, "P_adj",
            "Action", "Action_adj",
            "PCR_oi", "PCR_vol", "VOI_call", "VOI_put",
            "Close", "Target_5d", "Bar",
        ]
        show_cols = [c for c in desired if c in live_df.columns]
    else:
        desired   = ["AsOf", "Ticker", "Name", prob_col, "Action", "Close", "Target_5d", "Bar"]
        show_cols = [c for c in desired if c in live_df.columns]

    st.markdown(f"### 🟣 Live–Forecast Board – {HORIZON}-Tage Prognose (heute)")
    styled_live = style_live_board(live_df[show_cols], prob_col, ENTRY_PROB)
    show_styled_or_plain(live_df[show_cols], styled_live)
    st.download_button(
        "Live-Forecasts als CSV",
        to_csv_eu(live_df),
        file_name=f"live_forecasts_today_{HORIZON}d.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# Summary / Open Positions / Round-Trips / Korrelation
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net_pnl          = summary_df["Net P&L (€)"].sum()
    total_fees             = summary_df["Fees (€)"].sum()
    total_gross_pnl        = total_net_pnl + total_fees
    total_trades           = summary_df["Number of Trades"].sum()
    total_capital          = INIT_CAP * len(summary_df)
    total_net_return_pct   = total_net_pnl / total_capital * 100
    total_gross_return_pct = total_gross_pnl / total_capital * 100

    st.subheader("📊 Summary of all Tickers (Next Open Backtest)")
    cols = st.columns(4)
    cols[0].metric("Cumulative Net P&L (€)",       f"{total_net_pnl:,.2f}")
    cols[1].metric("Cumulative Trading Costs (€)",  f"{total_fees:,.2f}")
    cols[2].metric("Cumulative Gross P&L (€)",      f"{total_gross_pnl:,.2f}")
    cols[3].metric("Total Number of Trades",        f"{int(total_trades)}")

    bh_total_pct = float(
        summary_df["Buy & Hold Net (%)"].dropna().mean()
    ) if "Buy & Hold Net (%)" in summary_df.columns else float("nan")

    cols_pct = st.columns(4)
    cols_pct[0].metric("Strategy Net (%) – total",   f"{total_net_return_pct:.2f}")
    cols_pct[1].metric("Strategy Gross (%) – total", f"{total_gross_return_pct:.2f}")
    cols_pct[2].metric("Buy & Hold Net (%) – total", f"{bh_total_pct:.2f}")
    cols_pct[3].metric(
        "Durchschn. CAGR (%)",
        f"{summary_df['CAGR (%)'].dropna().mean():.2f}"
        if "CAGR (%)" in summary_df else "–",
    )

    def color_phase_html(val: str) -> str:
        colors = {"Open": "#d0ebff", "Flat": "#f0f0f0"}
        return f"background-color: {colors.get(val, '#ffffff')};"

    # FIX 3: Styler.applymap → Styler.map
    styled_summary = (
        summary_df.style
        .format({
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
        })
        .map(
            lambda v: "font-weight: bold;" if isinstance(v, (int, float)) else "",
            subset=pd.IndexSlice[:, ["Sharpe-Ratio", "Sortino-Ratio"]],
        )
        .map(color_phase_html, subset=["Phase"])
        .set_caption("Strategy-Performance per Ticker (Next Open Execution)")
    )
    show_styled_or_plain(summary_df, styled_summary)
    st.download_button(
        "Summary als CSV herunterladen",
        to_csv_eu(summary_df.reset_index()),
        file_name="strategy_summary.csv",
        mime="text/csv",
    )

    # ── Open Positions ──────────────────────────────────────
    st.subheader("📋 Open Positions (Next Open Backtest)")
    NAME_OVERRIDES = {
        "QBTS": "D-Wave Quantum Inc.",
        "NOG":  "Northern Oil and Gas, Inc.",
        "LUMN": "Lumen Technologies, Inc.",
    }
    open_positions = []
    for ticker, trades in all_trades.items():
        if trades and trades[-1]["Typ"] == "Entry":
            last_entry = next(t for t in reversed(trades) if t["Typ"] == "Entry")
            entry_ts   = pd.to_datetime(last_entry["Date"])
            prob       = float(all_feat[ticker]["SignalProb"].iloc[-1])
            last_close = float(all_dfs[ticker]["Close"].iloc[-1])
            upnl       = (
                (last_close - float(last_entry["Price"])) * float(last_entry["Shares"])
            )
            name = NAME_OVERRIDES.get(ticker) or get_ticker_name(ticker) or ticker
            open_positions.append({
                "Ticker":              ticker,
                "Name":               name,
                "Entry Date":         entry_ts,
                "Entry Price":        round(float(last_entry["Price"]), 2),
                "Pos Frac":           round(float(last_entry.get("PosFrac", np.nan)), 4),
                "Current Prob.":      round(prob, 4),
                "Unrealized PnL (€)": round(upnl, 2),
            })

    if open_positions:
        open_df         = pd.DataFrame(open_positions).sort_values(
            "Entry Date", ascending=False
        )
        open_df_display = open_df.copy()
        open_df_display["Entry Date"] = open_df_display["Entry Date"].dt.strftime(
            "%Y-%m-%d"
        )
        styled_open = open_df_display.style.format({
            "Entry Price":        "{:.2f}",
            "Pos Frac":           "{:.4f}",
            "Current Prob.":      "{:.4f}",
            "Unrealized PnL (€)": "{:.2f}",
        })
        show_styled_or_plain(open_df_display, styled_open)
        st.download_button(
            "Offene Positionen als CSV",
            to_csv_eu(open_df),
            file_name="open_positions.csv",
            mime="text/csv",
        )
    else:
        st.success("Keine offenen Positionen.")

    # ── Round-Trips ──────────────────────────────────────────
    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips) – Filter")

        rt_df["Entry Date"] = pd.to_datetime(rt_df["Entry Date"])
        rt_df["Exit Date"]  = pd.to_datetime(rt_df["Exit Date"])
        for c in [
            "Entry Prob", "Exit Prob", "Return (%)", "PnL Net (€)", "Fees (€)", "Hold (days)"
        ]:
            if c not in rt_df.columns:
                rt_df[c] = np.nan

        r_min_d = rt_df["Entry Date"].min().date()
        r_max_d = rt_df["Entry Date"].max().date()
        r_ticks = sorted(rt_df["Ticker"].unique().tolist())

        def finite_minmax(series, fallback=(0.0, 1.0)):
            s  = pd.to_numeric(series, errors="coerce")
            lo = float(np.nanmin(s.values))
            hi = float(np.nanmax(s.values))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = fallback
            return lo, hi

        r1, r2, r3 = st.columns([1.1, 1.1, 1.5])
        with r1:
            rt_tick_sel = st.multiselect(
                "Ticker (Round-Trips)", options=r_ticks, default=r_ticks
            )
            hd_min = int(np.nanmin(rt_df["Hold (days)"].values))
            hd_max = int(np.nanmax(rt_df["Hold (days)"].values))
            if not np.isfinite(hd_min): hd_min = 0
            if not np.isfinite(hd_max): hd_max = 60
            rt_hold = st.slider(
                "Haltedauer (Tage)",
                min_value=hd_min, max_value=hd_max,
                value=(hd_min, hd_max), step=1, key="rt_hold",
            )
        with r2:
            rt_date = st.date_input(
                "Zeitraum (Entry-Datum)",
                value=(r_min_d, r_max_d),
                min_value=r_min_d, max_value=r_max_d,
                key="rt_date",
            )
            ep_lo, ep_hi = finite_minmax(rt_df["Entry Prob"], (0.0, 1.0))
            xp_lo, xp_hi = finite_minmax(rt_df["Exit Prob"],  (0.0, 1.0))
            rt_ep = st.slider("Entry-Prob.", 0.0, 1.0, (max(0.0, ep_lo), min(1.0, ep_hi)), step=0.01)
            rt_xp = st.slider("Exit-Prob.",  0.0, 1.0, (max(0.0, xp_lo), min(1.0, xp_hi)), step=0.01)
        with r3:
            ret_lo, ret_hi = finite_minmax(rt_df["Return (%)"],  (-100.0, 200.0))
            pnl_lo, pnl_hi = finite_minmax(rt_df["PnL Net (€)"], (-INIT_CAP, INIT_CAP))
            rt_ret = st.slider(
                "Return (%)", float(ret_lo), float(ret_hi),
                (float(ret_lo), float(ret_hi)), step=0.5,
            )
            rt_pnl = st.slider(
                "PnL Net (€)", float(pnl_lo), float(pnl_hi),
                (float(pnl_lo), float(pnl_hi)), step=10.0,
            )

        rds, rde = (
            rt_date if isinstance(rt_date, tuple) else (r_min_d, r_max_d)
        )
        mask_rt = (
            rt_df["Ticker"].isin(rt_tick_sel)
            & (rt_df["Entry Date"].dt.date.between(rds, rde))
            & (rt_df["Hold (days)"].fillna(-1).between(rt_hold[0], rt_hold[1]))
            & (rt_df["Entry Prob"].fillna(0.0).between(rt_ep[0], rt_ep[1]))
            & (rt_df["Exit Prob"].fillna(0.0).between(rt_xp[0], rt_xp[1]))
            & (
                pd.to_numeric(rt_df["Return (%)"], errors="coerce")
                .fillna(-9e9).between(rt_ret[0], rt_ret[1])
            )
            & (
                pd.to_numeric(rt_df["PnL Net (€)"], errors="coerce")
                .fillna(-9e9).between(rt_pnl[0], rt_pnl[1])
            )
        )
        rt_f      = rt_df.loc[mask_rt].copy()
        rt_f_disp = rt_f.copy()
        rt_f_disp["Entry Date"] = rt_f_disp["Entry Date"].dt.strftime("%Y-%m-%d")
        rt_f_disp["Exit Date"]  = rt_f_disp["Exit Date"].dt.strftime("%Y-%m-%d")
        if "Hold (days)" in rt_f_disp.columns:
            rt_f_disp["Hold (days)"] = rt_f_disp["Hold (days)"].round().astype("Int64")

        fmt_rt = {
            "Shares":       "{:.4f}",
            "Entry Price":  "{:.2f}",
            "Exit Price":   "{:.2f}",
            "PnL Net (€)":  "{:.2f}",
            "Fees (€)":     "{:.2f}",
            "Return (%)":   "{:.2f}",
            "Entry Prob":   "{:.4f}",
            "Exit Prob":    "{:.4f}",
        }
        if "PosFrac" in rt_f_disp.columns:
            fmt_rt["PosFrac"] = "{:.4f}"

        styled_rt = rt_f_disp.style.format(fmt_rt)
        show_styled_or_plain(rt_f_disp, styled_rt)
        st.download_button(
            "Round-Trips (gefiltert) als CSV",
            to_csv_eu(rt_f_disp),
            file_name="round_trips_filtered.csv",
            mime="text/csv",
        )

        # Histogramme
        st.markdown("### 📊 Verteilung der Round-Trip-Ergebnisse")
        bins = st.slider("Anzahl Bins", 10, 100, 30, step=5, key="rt_bins")

        ret = pd.to_numeric(rt_f.get("Return (%)"),  errors="coerce").dropna()
        pnl = pd.to_numeric(rt_f.get("PnL Net (€)"), errors="coerce").dropna()

        def pct(x):
            return f"{x:.2f}%"

        cstats = st.columns(5)
        cstats[0].metric("Anzahl",   f"{len(ret)}")
        cstats[1].metric("Winrate",  pct(100.0 * (ret > 0).mean()) if len(ret) else "–")
        cstats[2].metric("Ø Return", pct(ret.mean())   if len(ret) else "–")
        cstats[3].metric("Median",   pct(ret.median()) if len(ret) else "–")
        cstats[4].metric("Std-Abw.", pct(ret.std())    if len(ret) else "–")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            if ret.empty:
                st.info("Keine Rendite-Werte vorhanden.")
            else:
                fig_ret = go.Figure(go.Histogram(
                    x=ret, nbinsx=bins, marker_line_width=0
                ))
                fig_ret.add_vline(x=0,              line_dash="dash", opacity=0.5)
                fig_ret.add_vline(x=float(ret.mean()), line_dash="dot", opacity=0.9)
                fig_ret.update_layout(
                    title="Histogramm: Return (%)",
                    xaxis_title="Return (%)", yaxis_title="Häufigkeit",
                    height=360, margin=dict(t=40, l=40, r=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_ret, use_container_width=True)
        with col_h2:
            if pnl.empty:
                st.info("Keine PnL-Werte vorhanden.")
            else:
                fig_pnl = go.Figure(go.Histogram(
                    x=pnl, nbinsx=bins, marker_line_width=0
                ))
                fig_pnl.add_vline(x=0,              line_dash="dash", opacity=0.5)
                fig_pnl.add_vline(x=float(pnl.mean()), line_dash="dot", opacity=0.9)
                fig_pnl.update_layout(
                    title="Histogramm: PnL Net (€)",
                    xaxis_title="PnL Net (€)", yaxis_title="Häufigkeit",
                    height=360, margin=dict(t=40, l=40, r=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

        # Korrelation
        st.markdown("### 🔗 Portfolio-Korrelation (Close-Returns)")
        c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.2, 1.0])
        with c1:
            corr_freq = st.selectbox(
                "Return-Frequenz",
                ["täglich", "wöchentlich", "monatlich"],
                index=0, key="corr_freq",
            )
        with c2:
            corr_method = st.selectbox(
                "Korrelationsmethode",
                ["Pearson", "Spearman", "Kendall"],
                index=0, key="corr_method",
            )
        with c3:
            min_obs = st.slider(
                "Min. gemeinsame Zeitpunkte", 3, 60, 20, step=1, key="corr_min_obs"
            )
        with c4:
            use_ffill_corr = st.checkbox(
                "Lücken per FFill schließen", value=True, key="corr_ffill"
            )

        price_series = []
        for tk, dfbt in all_dfs.items():
            if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2:
                s      = dfbt["Close"].copy()
                s.name = tk
                price_series.append(s)

        corr = None
        if len(price_series) < 2:
            st.info("Mindestens zwei Ticker mit Daten nötig.")
        else:
            prices = pd.concat(price_series, axis=1, join="outer").sort_index()
            if use_ffill_corr:
                prices = prices.ffill()
            if corr_freq == "wöchentlich":
                prices = prices.resample("W-FRI").last()
            elif corr_freq == "monatlich":
                prices = prices.resample("ME").last()

            rets        = prices.pct_change().dropna(how="all")
            enough      = [c for c in rets.columns if rets[c].count() >= min_obs]
            rets        = rets[enough]
            common_rows = rets.dropna(how="any")

            if rets.shape[1] < 2 or len(common_rows) < min_obs:
                st.info("Zu wenige Datenüberschneidungen für eine Korrelationsmatrix.")
            else:
                corr = rets.corr(method=corr_method.lower(), min_periods=min_obs)
                fig_corr = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1,
                )
                fig_corr.update_layout(
                    height=560, margin=dict(t=40, l=40, r=30, b=40),
                    coloraxis_colorbar=dict(title="ρ"),
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption(
                    f"Basis: {len(common_rows)} gemeinsame Zeitpunkte "
                    f"· Frequenz: {corr_freq} · Methode: {corr_method}"
                )

        if corr is not None and corr.shape[0] >= 2:
            N         = corr.shape[0]
            tri_vals  = corr.where(~np.eye(N, dtype=bool)).stack()
            avg_pair  = float(tri_vals.mean())
            med_pair  = float(tri_vals.median())
            std_pair  = float(tri_vals.std())
            w         = np.full(N, 1.0 / N)
            ip_raw    = float(w @ corr.values @ w)
            ip_norm   = float((ip_raw - 1.0 / N) / (1.0 - 1.0 / N))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Ø Paar-Korrelation",              f"{avg_pair:.2f}")
            mc2.metric("Median",                          f"{med_pair:.2f}")
            mc3.metric("Streuung (σ)",                    f"{std_pair:.2f}")
            mc4.metric("Portfolio-Korrelation (normiert)", f"{ip_norm:.2f}")
            st.caption(
                f"IPC roh={ip_raw:.3f} · normiert={ip_norm:.3f} "
                f"· N={N} · Methode: {corr_method}"
            )

else:
    st.warning(
        "Noch keine Ergebnisse verfügbar. "
        "Prüfe Ticker-Eingaben und Datenabdeckung."
    )
