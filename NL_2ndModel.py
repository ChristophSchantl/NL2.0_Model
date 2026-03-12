# streamlit_app.py  –  NEXUS v3
#
# ── Changelog gegenüber v2 ───────────────────────────────────────────────────
#  V3-A  HistGradientBoostingClassifier als Default (kein StandardScaler nötig,
#         schneller, robuster auf tabellarischen Daten)
#  V3-B  Ensemble standardmäßig AUS; nur als Research-Modus
#  V3-C  RF n_jobs=1 (kein verschachteltes Parallelisieren)
#  V3-D  as_completed() statt sequenzieller future.result() überall
#  V3-E  Walk-Forward blockweise: retrain_steps_set einmal gebaut,
#         dann batch-predict für wf_stride Zeilen → viel weniger Overhead
#  V3-F  Feature-Matrix als float32 NumPy-Array vorgebaut (kein slice+fillna
#         in jeder Iteration)
#  V3-G  Optionsfeatures RAUS aus Core-Modell, NUR als Live-Overlay / Adjustment
#  V3-H  Calmar = CAGR / |MaxDD| (nicht Total Return / MaxDD)
#  V3-I  "Buy & Hold (%)" statt "Net" (keine Kosten dort)
#  V3-J  "Completed Trades" statt "Number of Trades"
#  V3-K  Volatility-Sizing Beschreibung korrigiert (Einzel-Pos-Heuristik)
#  V3-L  Optimizer zweistufig: grob → fein
#  V3-M  Zusätzliche Features: z-score Returns, ATR, Gap, Rolling-Skew
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# V3-A: HistGradientBoostingClassifier als Default
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS v3 – Signal-basierte Strategie", layout="wide"
)
LOCAL_TZ    = ZoneInfo("Europe/Zurich")
MAX_WORKERS = 8
pd.options.display.float_format = "{:,.4f}".format

# V3-G: Core-Features OHNE Options (Options nur als Overlay)
CORE_FEATURE_COLS = [
    "Range", "SlopeHigh", "SlopeLow",
    "Ret_5d", "Ret_20d", "MA_ratio",
    "Volatility", "RSI", "Vol_ratio",
    "ZScore_5d", "ATR_ratio", "Gap", "Roll_Skew",
]
# V3-G: Optionsfeatures als separater Overlay – nicht in x_cols
OPTIONS_OVERLAY_COLS = [
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
            return _normalize_tickers(df[cols_lower[key]].astype(str).tolist())
    return _normalize_tickers(df[df.columns[0]].astype(str).tolist())


# ─────────────────────────────────────────────────────────────
# Sidebar
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
    st.sidebar.caption("CSV mit Spalte **ticker** (oder erste Spalte).")
    uploads = st.sidebar.file_uploader("CSV-Dateien", type=["csv"], accept_multiple_files=True)
    collected = []
    if uploads:
        for up in uploads:
            try:
                collected += parse_ticker_csv(up)
            except Exception as e:
                st.sidebar.error(f"Fehler: '{up.name}': {e}")
    base   = _normalize_tickers(collected)
    extra  = st.sidebar.text_input("Weitere Ticker manuell", value="", key="extra_csv")
    extras = _normalize_tickers([t for t in extra.split(",") if t.strip()]) if extra else []
    tickers_final = _normalize_tickers(base + extras)
    if tickers_final:
        st.sidebar.caption(f"Gefundene Ticker: {len(tickers_final)}")
        if st.sidebar.checkbox("Zufällig mischen", value=False):
            import random; random.seed(42); random.shuffle(tickers_final)
        max_n = st.sidebar.number_input("Max. Anzahl (0=alle)", 0, len(tickers_final), 0, step=10)
        if max_n and max_n < len(tickers_final):
            tickers_final = tickers_final[:int(max_n)]
        tickers_final = st.sidebar.multiselect("Auswahl", options=tickers_final, default=tickers_final)

if not tickers_final:
    tickers_final = _normalize_tickers(["REGN", "VOW3.DE", "LULU", "REI", "DDL"])

st.sidebar.download_button(
    "Ticker als CSV",
    to_csv_eu(pd.DataFrame({"ticker": tickers_final})),
    file_name="tickers.csv", mime="text/csv",
)
TICKERS = tickers_final

# ── Backtest-Parameter ────────────────────────────────────────
START_DATE = st.sidebar.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
END_DATE   = st.sidebar.date_input("End Date",   value=pd.to_datetime(datetime.now(LOCAL_TZ).date()))
LOOKBACK   = st.sidebar.number_input("Lookback (Tage)", 10, 252, 35, step=5)
HORIZON    = st.sidebar.number_input("Horizon (Tage)", 1, 10, 5)
THRESH     = st.sidebar.number_input("Threshold Target", 0.0, 0.1, 0.046, step=0.005, format="%.3f")
ENTRY_PROB = st.sidebar.slider("Entry Threshold P(Signal)", 0.0, 1.0, 0.62, step=0.01)
EXIT_PROB  = st.sidebar.slider("Exit Threshold P(Signal)",  0.0, 1.0, 0.48, step=0.01)
if EXIT_PROB >= ENTRY_PROB:
    st.sidebar.error("Exit-Threshold muss unter Entry-Threshold liegen.")
    st.stop()

MIN_HOLD_DAYS = st.sidebar.number_input("Mindesthaltedauer (Tage)", 0, 252, 5, step=1)
COOLDOWN_DAYS = st.sidebar.number_input("Cooling Phase (Tage)", 0, 252, 0, step=1)
COMMISSION    = st.sidebar.number_input("Commission (ad valorem)", 0.0, 0.02, 0.004, step=0.0001, format="%.4f")
SLIPPAGE_BPS  = st.sidebar.number_input("Slippage (bp)", 0, 50, 5, step=1)
POS_FRAC      = st.sidebar.slider("Positionsgröße (%)", 0.1, 1.0, 1.0, step=0.1)
INIT_CAP      = st.sidebar.number_input("Initial Capital (€)", min_value=1000.0, value=10_000.0, step=1000.0, format="%.2f")

# ── Intraday ──────────────────────────────────────────────────
use_live              = st.sidebar.checkbox("Intraday-Tail einlesen", value=True)
intraday_interval     = st.sidebar.selectbox("Intraday-Intervall", ["1m","2m","5m","15m"], index=2)
fallback_last_session = st.sidebar.checkbox("Fallback: letzte Session", value=False)
exec_mode             = st.sidebar.selectbox("Execution Mode", ["Next Open (backtest+live)", "Market-On-Close (live only)"])
moc_cutoff_min        = st.sidebar.number_input("MOC Cutoff (Min vor Close)", 5, 60, 15, step=5)
intraday_chart_type   = st.sidebar.selectbox("Intraday-Chart", ["Candlestick (OHLC)", "Close-Linie"], index=0)

# ── Modellparameter ───────────────────────────────────────────
st.sidebar.markdown("**Modellparameter**")

# V3-A: Modell-Auswahl mit HistGBM als Default
MODEL_TYPE = st.sidebar.selectbox(
    "Modell",
    ["HistGBM (empfohlen)", "GBM (klassisch)", "Ensemble GBM+RF+LR (langsam)"],
    index=0,
    help=(
        "HistGBM: schneller, robuster, kein Scaler nötig – empfohlen.\n"
        "GBM: klassisch, kompatibel.\n"
        "Ensemble: maximale Robustheit, aber deutlich langsamer."
    ),
)
USE_ENSEMBLE     = MODEL_TYPE.startswith("Ensemble")
USE_HIST_GBM     = MODEL_TYPE.startswith("HistGBM")
USE_CALIBRATION  = st.sidebar.checkbox(
    "Probability Calibration (Platt Scaling)",
    value=False,
    help=(
        "Kalibriert die Modell-Wahrscheinlichkeiten via Platt-Scaling.\n"
        "Empfohlen wenn Entry/Exit auf festen P-Schwellen basiert.\n"
        "Erhöht Rechenzeit leicht."
    ),
)
USE_WALK_FORWARD = st.sidebar.checkbox("Walk-Forward (kein Lookahead-Bias)", value=True)
n_estimators  = st.sidebar.number_input("n_estimators",  10, 500, 100, step=10)
learning_rate = st.sidebar.number_input("learning_rate", 0.01, 1.0, 0.1, step=0.01, format="%.2f")
max_depth     = st.sidebar.number_input("max_depth", 1, 10, 3, step=1)
MODEL_PARAMS  = dict(
    n_estimators=int(n_estimators),
    learning_rate=float(learning_rate),
    max_depth=int(max_depth),
    random_state=42,
)

# V3-E: Walk-Forward Stride
WF_STRIDE = st.sidebar.number_input(
    "Walk-Forward Stride (Tage)", min_value=1, max_value=30, value=5, step=1,
    help="Modell wird alle N Tage neu trainiert. Stride=5 → ~5× schneller (empfohlen).",
)

# ── Positionsgröße ────────────────────────────────────────────
st.sidebar.markdown("**Positionsgröße**")
USE_VOL_SIZING = st.sidebar.checkbox(
    "Volatilitäts-Sizing (Einzelposition)",
    value=False,
    # V3-K: korrekte Beschreibung
    help=(
        "Heuristik: Skaliert die Positionsgröße dieser Einzelaktie so, dass ihre "
        "annualisierte historische Vola annähernd der Ziel-Vola entspricht. "
        "Achtung: kein echtes Portfolio-Vol-Targeting – Korrelationen zwischen "
        "Tickern werden nicht berücksichtigt."
    ),
)
TARGET_VOL_ANNUAL = st.sidebar.number_input(
    "Ziel-Vola p.a. (%)", 5.0, 50.0, 15.0, step=1.0,
    help="Nur aktiv wenn Volatilitäts-Sizing aktiviert.",
) / 100.0

# ── Optionsdaten ──────────────────────────────────────────────
st.sidebar.markdown("**Optionsdaten (Live-Overlay)**")
use_chain_live  = st.sidebar.checkbox(
    "Live-Optionskette je Aktie (Overlay)", value=True,
    # V3-G: klar als Overlay deklariert
    help=(
        "Optionsdaten werden NICHT in das trainierte Modell eingespeist.\n"
        "Sie dienen nur als Live-Adjustment des Signals (P_adj)."
    ),
)
atm_band_pct    = st.sidebar.slider("ATM-Band (±%)", 1, 15, 5, step=1) / 100.0
max_days_to_exp = st.sidebar.slider("Max. Restlaufzeit (Tage)", 7, 45, 21, step=1)
n_expiries      = st.sidebar.slider("Nächste n Verfälle", 1, 4, 2, step=1)

# Housekeeping
c1, c2 = st.sidebar.columns(2)
if c1.button("🔄 Cache leeren"):
    st.cache_data.clear(); st.rerun()
if c2.button("📥 Summary"):
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
            raise AttributeError
    except Exception as e:
        st.warning(f"Styled-Tabelle fallback. ({e})")
        st.dataframe(df, use_container_width=True)


def last_timestamp_info(df: pd.DataFrame, meta: Optional[dict] = None):
    ts  = df.index[-1]
    msg = f"Letzter Datenpunkt: {ts.strftime('%Y-%m-%d %H:%M %Z')}"
    if meta and meta.get("tail_is_intraday") and meta.get("tail_ts") is not None:
        msg += f" (intraday bis {meta['tail_ts'].strftime('%H:%M %Z')})"
    st.caption(msg)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_ticker_name(ticker: str) -> str:
    try:
        info = {}
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception:
            info = getattr(yf.Ticker(ticker), "info", {}) or {}
        for k in ("shortName", "longName", "displayName", "companyName", "name"):
            if k in info and info[k]:
                return str(info[k])
    except Exception:
        pass
    return ticker


def style_live_board(df: pd.DataFrame, prob_col: str, entry_thr: float):
    def _row_color(row):
        act = str(row.get("Action_adj", row.get("Action", ""))).lower()
        if "enter" in act: return ["background-color: #D7F3F7"] * len(row)
        if "exit"  in act: return ["background-color: #FFE8E8"] * len(row)
        try:
            if float(row.get(prob_col, np.nan)) >= float(entry_thr):
                return ["background-color: #E6F7FF"] * len(row)
        except Exception:
            pass
        return ["background-color: #F7F7F7"] * len(row)

    fmt = {prob_col: "{:.4f}"}
    for c in ["P_adj","Close","Target_5d"] + OPTIONS_OVERLAY_COLS:
        if c in df.columns:
            fmt[c] = "{:.4f}" if c not in ("Close","Target_5d") else "{:.2f}"
    sty = df.style.format(fmt).apply(_row_color, axis=1)
    for col in [c for c in ["Action","Action_adj"] if c in df.columns]:
        sty = sty.set_properties(subset=[col], **{"font-weight": "600"})
    return sty


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def get_price_data_tail_intraday(
    ticker: str, years: int = 3,
    use_tail: bool = True, interval: str = "5m",
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
    df   = df.sort_index().drop_duplicates()
    meta = {"tail_is_intraday": False, "tail_ts": None}

    if not use_tail:
        df.dropna(subset=["High","Low","Close","Open"], inplace=True)
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
        last_bar = intraday.iloc[-1]
        day_key  = pd.Timestamp(last_bar.name.date(), tz=LOCAL_TZ)
        df.loc[day_key] = {
            "Open":   float(intraday["Open"].iloc[0]),
            "High":   float(intraday["High"].max()),
            "Low":    float(intraday["Low"].min()),
            "Close":  float(last_bar["Close"]),
            "Volume": float(intraday["Volume"].sum()),
        }
        df = df.sort_index()
        meta["tail_is_intraday"] = True
        meta["tail_ts"]          = last_bar.name

    df.dropna(subset=["High","Low","Close","Open"], inplace=True)
    return df, meta


@st.cache_data(show_spinner=False, ttl=120)
def get_intraday_last_n_sessions(ticker: str, sessions: int = 5, days_buffer: int = 10, interval: str = "5m") -> pd.DataFrame:
    tk   = yf.Ticker(ticker)
    intr = tk.history(period=f"{days_buffer}d", interval=interval, auto_adjust=True, actions=False, prepost=False)
    if intr.empty:
        return intr
    if intr.index.tz is None:
        intr.index = intr.index.tz_localize("UTC")
    intr.index = intr.index.tz_convert(LOCAL_TZ)
    intr = intr.sort_index()
    keep = set(pd.Index(intr.index.normalize().unique())[-sessions:])
    return intr.loc[intr.index.normalize().isin(keep)].copy()


def load_all_prices(
    tickers: List[str], start: str, end: str,
    use_tail: bool, interval: str, fallback_last: bool,
    exec_key: str, moc_cutoff: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    price_map: Dict[str, pd.DataFrame] = {}
    meta_map:  Dict[str, dict]         = {}
    if not tickers:
        return price_map, meta_map
    st.info(f"Kurse laden für {len(tickers)} Ticker …")
    prog = st.progress(0.0)
    done = 0

    # V3-D: as_completed() statt sequenzielles zip-Warten
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        fut_map = {
            ex.submit(get_price_data_tail_intraday, tk, 3, use_tail, interval,
                      fallback_last, exec_key, int(moc_cutoff)): tk
            for tk in tickers
        }
        for fut in as_completed(fut_map):
            tk = fut_map[fut]
            try:
                df_full, meta = fut.result()
                price_map[tk] = df_full.loc[str(start):str(end)].copy()
                meta_map[tk]  = meta
            except Exception as e:
                st.error(f"Fehler {tk}: {e}")
            finally:
                done += 1
                prog.progress(done / len(tickers))
    return price_map, meta_map


# ─────────────────────────────────────────────────────────────
# Optionsketten (nur noch als Live-Overlay)
# ─────────────────────────────────────────────────────────────
def _atm_strike(ref_px: float, strikes: np.ndarray) -> float:
    if not np.isfinite(ref_px) or strikes.size == 0:
        return np.nan
    return float(strikes[np.argmin(np.abs(strikes - ref_px))])


def _band_mask(strikes: pd.Series, atm: float, band: float) -> pd.Series:
    if not np.isfinite(atm):
        return pd.Series(False, index=strikes.index)
    return strikes.between(atm * (1 - band), atm * (1 + band))


@st.cache_data(show_spinner=False, ttl=180)
def get_equity_chain_aggregates_for_today(
    ticker: str, ref_price: float, atm_band: float, n_exps: int, max_days: int,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options or []
    except Exception:
        exps = []
    if not exps:
        return pd.DataFrame()

    today     = pd.Timestamp.today(tz=LOCAL_TZ).normalize()
    exps_filt = sorted(
        [(pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize(), e)
         for e in exps
         if (pd.Timestamp(e).tz_localize("UTC").tz_convert(LOCAL_TZ).normalize() - today).days <= max_days],
        key=lambda x: x[0]
    )
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
        for df_ in (calls, puts):
            for c in ["volume","openInterest","impliedVolatility","strike"]:
                if c not in df_.columns: df_[c] = np.nan

        strikes = np.sort(pd.concat([calls["strike"], puts["strike"]]).dropna().unique())
        atm = _atm_strike(ref_price, strikes)
        mC  = calls[_band_mask(calls["strike"], atm, atm_band)]
        mP  = puts [_band_mask(puts ["strike"], atm, atm_band)]

        vC  = float(np.nansum(mC["volume"])); vP  = float(np.nansum(mP["volume"]))
        oiC = float(np.nansum(mC["openInterest"])); oiP = float(np.nansum(mP["openInterest"]))
        ivC = float(np.nanmean(mC["impliedVolatility"])) if len(mC) else np.nan
        ivP = float(np.nanmean(mP["impliedVolatility"])) if len(mP) else np.nan
        rows.append(dict(vol_c=vC, vol_p=vP, oi_c=oiC, oi_p=oiP,
                         voi_c=vC/max(oiC,1.), voi_p=vP/max(oiP,1.), iv_c=ivC, iv_p=ivP))

    if not rows:
        return pd.DataFrame()
    agg = pd.DataFrame(rows).agg({"vol_c":"sum","vol_p":"sum","oi_c":"sum","oi_p":"sum",
                                  "voi_c":"mean","voi_p":"mean","iv_c":"mean","iv_p":"mean"})
    out = pd.DataFrame([{
        "PCR_vol": agg["vol_p"]/max(agg["vol_c"],1.),
        "PCR_oi":  agg["oi_p"] /max(agg["oi_c"],1.),
        "VOI_call": float(agg["voi_c"]), "VOI_put": float(agg["voi_p"]),
        "IV_skew_p_minus_c": float(agg["iv_p"] - agg["iv_c"]),
        "VOL_tot": float(agg["vol_c"]+agg["vol_p"]),
        "OI_tot":  float(agg["oi_c"] +agg["oi_p"]),
    }])
    out.index = [pd.Timestamp.today(tz=LOCAL_TZ).normalize()]
    return out


# ─────────────────────────────────────────────────────────────
# Feature-Engineering (V3-F: float32 NumPy-first, V3-M: neue Features)
# ─────────────────────────────────────────────────────────────
def _rolling_slope_vec(series: pd.Series, window: int) -> pd.Series:
    """Vektorisierte Rolling-OLS-Slope (~30× schneller als rolling().apply())."""
    from numpy.lib.stride_tricks import sliding_window_view
    x     = np.arange(window, dtype=np.float32)
    xm    = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    w_vec  = (x - xm) / denom
    vals   = series.to_numpy(dtype=np.float32)
    wins   = sliding_window_view(vals, window_shape=window)
    slopes = ((wins - wins.mean(axis=1, keepdims=True)) * w_vec).sum(axis=1)
    out    = np.full(len(vals), np.nan, dtype=np.float32)
    out[window - 1:] = slopes
    return pd.Series(out, index=series.index)


def make_features(df: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    """
    Berechnet ausschliesslich historisch vollständig verfügbare
    Preis-/Volumen-/Volatilitätsfeatures (V3-G: KEINE Optionsdaten).
    """
    feat = df.copy()
    c    = feat["Close"]
    h    = feat["High"]
    lo   = feat["Low"]

    # ── Trend/Volatilität ────────────────────────────────────
    feat["Range"]     = h.rolling(lookback).max() - lo.rolling(lookback).min()
    feat["SlopeHigh"] = _rolling_slope_vec(h,  lookback)
    feat["SlopeLow"]  = _rolling_slope_vec(lo, lookback)

    # ── Momentum ─────────────────────────────────────────────
    feat["Ret_5d"]   = c.pct_change(5)
    feat["Ret_20d"]  = c.pct_change(20)
    feat["MA_ratio"] = c / (c.rolling(20).mean() + 1e-9)

    # ── Realisierte Volatilität ───────────────────────────────
    daily_ret        = c.pct_change()
    feat["Volatility"] = daily_ret.rolling(lookback).std()

    # ── RSI (14) ─────────────────────────────────────────────
    delta      = c.diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    feat["RSI"] = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

    # ── Volumen-Ratio ─────────────────────────────────────────
    if "Volume" in feat.columns and feat["Volume"].gt(0).any():
        feat["Vol_ratio"] = feat["Volume"] / (feat["Volume"].rolling(20).mean().replace(0, np.nan))
    else:
        feat["Vol_ratio"] = 1.0

    # ── V3-M: z-score normalisierter Return ──────────────────
    ret_mu        = daily_ret.rolling(lookback).mean()
    ret_std       = daily_ret.rolling(lookback).std()
    feat["ZScore_5d"] = (feat["Ret_5d"] - ret_mu) / (ret_std + 1e-9)

    # ── V3-M: ATR-Ratio ──────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["ATR_ratio"] = atr / (c + 1e-9)

    # ── V3-M: Gap (Overnight-Gap) ────────────────────────────
    feat["Gap"] = (feat["Open"] / c.shift(1) - 1).fillna(0)

    # ── V3-M: Rolling Skewness ───────────────────────────────
    feat["Roll_Skew"] = daily_ret.rolling(lookback).skew()

    feat = feat.iloc[lookback - 1:].copy()
    feat["FutureRet"] = c.shift(-horizon) / c - 1
    return feat


# ─────────────────────────────────────────────────────────────
# Modell-Builder (V3-A/B/C)
# ─────────────────────────────────────────────────────────────
def build_model(model_params: dict, use_hist_gbm: bool, use_ensemble: bool):
    """
    V3-A: HistGBM als Default (kein StandardScaler nötig, schnell).
    V3-B: Ensemble nur als optionaler Research-Modus.
    V3-C: RF mit n_jobs=1 (kein verschachteltes Parallelisieren).
    """
    if use_hist_gbm:
        # HistGBM: braucht keinen StandardScaler, behandelt NaN nativ
        return HistGradientBoostingClassifier(
            max_iter=model_params.get("n_estimators", 100),
            learning_rate=model_params.get("learning_rate", 0.1),
            max_depth=model_params.get("max_depth", 3),
            random_state=42,
        )

    gbm = GradientBoostingClassifier(**model_params)

    if not use_ensemble:
        return gbm

    # V3-C: n_jobs=1 im RF – kein Oversubscription mit äußerem ThreadPool
    rf = RandomForestClassifier(
        n_estimators=max(50, model_params.get("n_estimators", 100) // 2),
        max_depth=model_params.get("max_depth", 4),
        random_state=42,
        n_jobs=1,
    )
    lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    return VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("lr", lr)],
        voting="soft", weights=[3, 2, 1],
    )


def needs_scaler(use_hist_gbm: bool) -> bool:
    """HistGBM braucht keinen StandardScaler."""
    return not use_hist_gbm


def extract_feature_importance(model, x_cols: List[str]) -> Optional[pd.Series]:
    est = model
    if hasattr(model, "named_estimators_"):
        est = model.named_estimators_.get("gbm", model)
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=x_cols)
    if hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_[0]), index=x_cols)
    return None


# ─────────────────────────────────────────────────────────────
# Training + Backtest (V3-E: blockweises Walk-Forward, V3-F: float32)
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
    walk_forward: bool = True,
    use_hist_gbm: bool = True,
    use_ensemble: bool = False,
    use_calibration: bool = False,
    use_vol_sizing: bool = False,
    target_vol_annual: float = 0.15,
    wf_stride: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], dict]:

    feat      = make_features(df, lookback, horizon)
    min_train = max(lookback + horizon + 5, 40)

    if len(feat) < min_train + 5:
        raise ValueError("Zu wenige Datenpunkte.")

    # V3-G: nur Core-Features (KEINE Optionsdaten im Modell)
    x_cols = [c for c in CORE_FEATURE_COLS if c in feat.columns]

    # V3-F: Feature-Matrix einmal als float32 vorbauen
    X_all = feat[x_cols].to_numpy(dtype=np.float32)
    # NaN mit 0 auffüllen (HistGBM kann NaN, GBM/Ensemble nicht)
    X_all_filled = np.where(np.isfinite(X_all), X_all, 0.0)

    last_model  = None
    last_scaler = None
    last_fi     = None

    if not walk_forward:
        # Volltraining (Lookahead möglich – nur für Quick-Check)
        hist      = feat.iloc[:-1].dropna(subset=["FutureRet"]).copy()
        hist["Target"] = (hist["FutureRet"] > threshold).astype(int)
        if hist["Target"].nunique() < 2:
            feat["SignalProb"] = 0.5
        else:
            idx_train = feat.index.get_indexer(hist.index)
            Xt = X_all_filled[idx_train]
            yt = hist["Target"].to_numpy()
            if needs_scaler(use_hist_gbm):
                last_scaler = StandardScaler().fit(Xt)
                Xt = last_scaler.transform(Xt)
            last_model = build_model(model_params, use_hist_gbm, use_ensemble)
            if use_calibration:
                last_model = CalibratedClassifierCV(last_model, method="sigmoid", cv=3)
            last_model.fit(Xt, yt)
            Xp = X_all_filled if last_scaler is None else last_scaler.transform(X_all_filled)
            feat["SignalProb"] = last_model.predict_proba(Xp)[:, 1]
    else:
        # V3-E: Blockweises Walk-Forward
        probs = np.full(len(feat), np.nan, dtype=np.float32)

        t = min_train
        while t < len(feat):
            # Training auf feat.iloc[:t]
            train_feat = feat.iloc[:t].dropna(subset=["FutureRet"])
            if len(train_feat) >= min_train:
                train_feat = train_feat.copy()
                train_feat["Target"] = (train_feat["FutureRet"] > threshold).astype(int)
                if train_feat["Target"].nunique() >= 2:
                    # V3-F: Indexgrenzen statt Slices
                    tr_idx = feat.index.get_indexer(train_feat.index)
                    Xt     = X_all_filled[tr_idx]
                    yt     = train_feat["Target"].to_numpy()
                    if needs_scaler(use_hist_gbm):
                        last_scaler = StandardScaler().fit(Xt)
                        Xt = last_scaler.transform(Xt)
                    last_model = build_model(model_params, use_hist_gbm, use_ensemble)
                    if use_calibration:
                        last_model = CalibratedClassifierCV(last_model, method="sigmoid", cv=3)
                    last_model.fit(Xt, yt)

            # V3-E: Blockweise vorhersagen (wf_stride Zeilen auf einmal)
            if last_model is not None:
                pred_end = min(t + int(wf_stride), len(feat))
                Xp = X_all_filled[t:pred_end]
                if last_scaler is not None:
                    Xp = last_scaler.transform(Xp)
                probs[t:pred_end] = last_model.predict_proba(Xp)[:, 1]

            t += int(wf_stride)

        feat["SignalProb"] = pd.Series(
            probs.astype(float), index=feat.index
        ).ffill().fillna(0.5)

    if last_model is not None:
        last_fi = extract_feature_importance(last_model, x_cols)

    feat_bt = feat.iloc[:-1].copy()
    df_bt, trades = backtest_next_open(
        feat_bt, entry_prob, exit_prob,
        COMMISSION, SLIPPAGE_BPS, INIT_CAP, POS_FRAC,
        min_hold_days=int(min_hold_days), cooldown_days=int(cooldown_days),
        use_vol_sizing=use_vol_sizing, target_vol_annual=target_vol_annual,
    )
    metrics = compute_performance(df_bt, trades, INIT_CAP)
    metrics["_feature_importance"] = last_fi
    metrics["_x_cols"]             = x_cols
    return feat, df_bt, trades, metrics


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────
def _vol_scaled_frac(vol_daily: float, pos_frac: float, target_vol_annual: float) -> float:
    if not np.isfinite(vol_daily) or vol_daily <= 1e-8:
        return pos_frac * 0.5
    return min((target_vol_annual / max(vol_daily * sqrt(252), 1e-6)) * pos_frac, pos_frac)


def backtest_next_open(
    df: pd.DataFrame,
    entry_thr: float, exit_thr: float,
    commission: float, slippage_bps: int,
    init_cap: float, pos_frac: float,
    min_hold_days: int = 0, cooldown_days: int = 0,
    use_vol_sizing: bool = False, target_vol_annual: float = 0.15,
) -> Tuple[pd.DataFrame, List[dict]]:
    df = df.copy()
    n  = len(df)
    if n < 2:
        raise ValueError("Zu wenige Datenpunkte für Backtest.")

    cash_gross = cash_net = init_cap
    shares = 0.0; in_pos = False
    cb_gross = cb_net = 0.0
    last_entry_idx = last_exit_idx = None
    eq_gross, eq_net, trades = [], [], []
    cum_pl_net = 0.0

    for i in range(n):
        if i > 0:
            open_t    = float(df["Open"].iloc[i])
            slip_buy  = open_t * (1 + slippage_bps / 10_000)
            slip_sell = open_t * (1 - slippage_bps / 10_000)
            prob_prev = float(df["SignalProb"].iloc[i - 1])
            date_exec = df.index[i]

            cool_ok = True
            if (not in_pos) and cooldown_days > 0 and last_exit_idx is not None:
                cool_ok = (i - last_exit_idx) > int(cooldown_days)

            # ENTRY
            if (not in_pos) and prob_prev > entry_thr and cool_ok:
                eff_frac = (
                    _vol_scaled_frac(float(df["Volatility"].iloc[i-1]) if "Volatility" in df.columns else np.nan,
                                     pos_frac, target_vol_annual)
                    if use_vol_sizing else pos_frac
                )
                invest_net    = cash_net * eff_frac
                fee_entry     = invest_net * commission
                target_shares = max((invest_net - fee_entry) / slip_buy, 0.0)
                if target_shares > 0 and (target_shares * slip_buy + fee_entry) <= cash_net + 1e-9:
                    shares         = target_shares
                    cb_gross       = shares * slip_buy
                    cb_net         = shares * slip_buy + fee_entry
                    cash_gross    -= cb_gross
                    cash_net      -= cb_net
                    in_pos         = True
                    last_entry_idx = i
                    trades.append(dict(Date=date_exec, Typ="Entry", Price=round(slip_buy,4),
                                       Shares=round(shares,4), GrossPnL=0.0,
                                       Fees=round(fee_entry,2), NetPnL=0.0,
                                       CumPnL=round(cum_pl_net,2), Prob=round(prob_prev,4),
                                       HoldDays=np.nan, PosFrac=round(eff_frac,4)))

            # EXIT
            elif in_pos and prob_prev < exit_thr:
                held = (i - last_entry_idx) if last_entry_idx is not None else 0
                if int(min_hold_days) > 0 and held < int(min_hold_days):
                    pass
                else:
                    gross_val   = shares * slip_sell
                    fee_exit    = gross_val * commission
                    pnl_gross   = gross_val - cb_gross
                    pnl_net     = (gross_val - fee_exit) - cb_net
                    cash_gross += gross_val
                    cash_net   += (gross_val - fee_exit)
                    in_pos      = False; shares = 0.0; cb_gross = cb_net = 0.0
                    cum_pl_net += pnl_net
                    trades.append(dict(Date=date_exec, Typ="Exit", Price=round(slip_sell,4),
                                       Shares=0.0, GrossPnL=round(pnl_gross,2),
                                       Fees=round(fee_exit,2), NetPnL=round(pnl_net,2),
                                       CumPnL=round(cum_pl_net,2), Prob=round(prob_prev,4),
                                       HoldDays=int(held), PosFrac=np.nan))
                    last_exit_idx  = i
                    last_entry_idx = None

        close_t = float(df["Close"].iloc[i])
        eq_gross.append(cash_gross + (shares * close_t if in_pos else 0.0))
        eq_net.append(cash_net   + (shares * close_t if in_pos else 0.0))

    df_bt                 = df.copy()
    df_bt["Equity_Gross"] = eq_gross
    df_bt["Equity_Net"]   = eq_net
    return df_bt, trades


# ─────────────────────────────────────────────────────────────
# Performance-Kennzahlen (V3-H: Calmar = CAGR/MaxDD, V3-I/J korrigiert)
# ─────────────────────────────────────────────────────────────
def _cagr_from_path(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    years = len(values) / 252.0
    v0, v1 = float(values.iloc[0]), float(values.iloc[-1])
    if years <= 0 or v0 <= 0 or not np.isfinite(v0) or not np.isfinite(v1):
        return np.nan
    return (v1 / v0) ** (1.0 / years) - 1.0


def _sortino(rets: pd.Series) -> float:
    if rets.empty:
        return np.nan
    mean     = rets.mean() * 252
    downside = rets[rets < 0]
    dd_std   = downside.std() * sqrt(252) if len(downside) else np.nan
    return mean / dd_std if dd_std and np.isfinite(dd_std) and dd_std > 0 else np.nan


def _winrate_roundtrips(trades: List[dict]) -> float:
    pnl, entry = [], None
    for ev in trades:
        if ev["Typ"] == "Entry": entry = ev
        elif ev["Typ"] == "Exit" and entry is not None:
            pnl.append(float(ev.get("NetPnL", 0.0))); entry = None
    return float((np.array(pnl) > 0).mean()) if pnl else np.nan


def compute_performance(df_bt: pd.DataFrame, trades: List[dict], init_cap: float) -> dict:
    eq   = df_bt["Equity_Net"]
    rets = eq.pct_change().dropna()
    dd   = (eq - eq.cummax()) / eq.cummax()

    net_ret   = (eq.iloc[-1] / init_cap - 1) * 100
    gross_ret = (df_bt["Equity_Gross"].iloc[-1] / init_cap - 1) * 100
    # V3-I: klar "Buy & Hold (%)" – keine fiktiven Kosten dort
    bh_ret    = (df_bt["Close"].iloc[-1] / df_bt["Close"].iloc[0] - 1) * 100
    vol_ann   = rets.std() * sqrt(252) * 100
    sharpe    = (rets.mean() * sqrt(252)) / (rets.std() + 1e-12)
    max_dd    = dd.min() * 100
    cagr      = _cagr_from_path(eq)

    # V3-H: Calmar = CAGR / |MaxDD| (nicht Total Return)
    max_dd_dec = abs(dd.min())
    calmar = (cagr / max_dd_dec) if (max_dd_dec > 0 and np.isfinite(cagr)) else np.nan

    fees      = sum(t["Fees"] for t in trades)
    phase     = "Open" if trades and trades[-1]["Typ"] == "Entry" else "Flat"
    # V3-J: "Completed Trades" statt "Number of Trades"
    completed = sum(1 for t in trades if t["Typ"] == "Exit")
    net_eur   = eq.iloc[-1] - init_cap
    sortino   = _sortino(rets)
    winrate   = _winrate_roundtrips(trades)

    return {
        "Strategy Net (%)":     round(net_ret, 2),
        "Strategy Gross (%)":   round(gross_ret, 2),
        "Buy & Hold (%)":       round(bh_ret, 2),   # V3-I
        "Volatility (%)":       round(vol_ann, 2),
        "Sharpe-Ratio":         round(sharpe, 2),
        "Sortino-Ratio":        round(sortino, 2) if np.isfinite(sortino) else np.nan,
        "Max Drawdown (%)":     round(max_dd, 2),
        "Calmar-Ratio":         round(calmar, 2) if np.isfinite(calmar) else np.nan,  # V3-H
        "Fees (€)":             round(fees, 2),
        "Phase":                phase,
        "Completed Trades":     completed,           # V3-J
        "Net P&L (€)":          round(net_eur, 2),
        "CAGR (%)":             round(100 * (cagr if np.isfinite(cagr) else np.nan), 2),
        "Winrate (%)":          round(100 * (winrate if np.isfinite(winrate) else np.nan), 2),
    }


def compute_round_trips(all_trades: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for tk, tr in all_trades.items():
        name = get_ticker_name(tk)
        entry = None
        for ev in tr:
            if ev["Typ"] == "Entry":
                entry = ev
            elif ev["Typ"] == "Exit" and entry is not None:
                ed = pd.to_datetime(entry["Date"]); xd = pd.to_datetime(ev["Date"])
                sh = float(entry.get("Shares", 0.0))
                ep = float(entry.get("Price", np.nan)); xp = float(ev.get("Price", np.nan))
                fe = float(entry.get("Fees", 0.0));    fx = float(ev.get("Fees", 0.0))
                pnl_net = float(ev.get("NetPnL", 0.0))
                cost    = sh * ep + fe
                rows.append(dict(
                    Ticker=tk, Name=name, EntryDate=ed, ExitDate=xd,
                    HoldDays=(xd-ed).days,
                    EntryProb=entry.get("Prob",np.nan), ExitProb=ev.get("Prob",np.nan),
                    PosFrac=entry.get("PosFrac",np.nan),
                    Shares=round(sh,4), EntryPrice=round(ep,4), ExitPrice=round(xp,4),
                    NetPnL=round(pnl_net,2), Fees=round(fe+fx,2),
                    ReturnPct=round(pnl_net/cost*100,2) if cost else np.nan,
                ))
                entry = None
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Feature-Importance-Chart
# ─────────────────────────────────────────────────────────────
def show_feature_importance(fi: Optional[pd.Series], x_cols: List[str]) -> None:
    if fi is None or fi.empty:
        st.caption("Feature-Importance nicht verfügbar.")
        return
    fi_sorted = fi.sort_values(ascending=True)
    colors    = ["#6366F1"] * len(fi_sorted)
    fig = go.Figure(go.Bar(
        x=fi_sorted.values, y=fi_sorted.index,
        orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Importance (Core-Modell) – nur Preis/Volumen-Features",
        xaxis_title="Importance",
        height=max(300, len(fi_sorted) * 28),
        margin=dict(t=45, l=160, r=20, b=35),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# V3-L: Zweistufiger Optimizer (Grob → Fein)
# ─────────────────────────────────────────────────────────────
def _composite_score(
    sharpe: float, winrate: float, cagr: float, max_dd: float, w_dd: float,
) -> float:
    if not all(np.isfinite(v) for v in [sharpe, winrate, cagr, max_dd]):
        return float("-inf")
    return sharpe * (winrate * 2.0) * (1.0 + max(cagr, -1.0)) - w_dd * abs(max_dd)


st.subheader("🧭 Parameter-Optimierung (Zweistufig)")
with st.expander("Optimizer – Grob → Fein · Composite Score", expanded=False):

    st.markdown(
        "**Score = Sharpe × (2·Winrate) × (1 + CAGR) − w_DD·|MaxDD|**  \n"
        "Zweistufig: breite Grobsuche → verfeinerte Feinsuche um Top-Kandidaten."
    )

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        n_coarse    = st.number_input("Grob-Trials",  10, 500,  80, step=10)
        n_fine      = st.number_input("Fein-Trials",  10, 500,  40, step=10)
        top_k       = st.number_input("Top-K für Feinsuche", 1, 20, 5, step=1)
    with oc2:
        w_dd_opt    = st.number_input("Drawdown-Gewicht w_DD", 0.0, 10.0, 1.5, step=0.1)
        opt_seed    = st.number_input("Seed", 0, 10_000, 42)
        fine_band   = st.number_input("Fein-Band (±%)", 1, 50, 15, step=5,
                                      help="Feinsuche bewegt sich ±N% um den besten Grob-Parameter") / 100.0
    with oc3:
        lb_lo, lb_hi   = st.slider("Lookback",   10, 252, (20, 120), step=5)
        hz_lo, hz_hi   = st.slider("Horizon",    1,  10,  (3,  8))
        thr_lo, thr_hi = st.slider("Thresh",     0.0, 0.10, (0.030, 0.10), step=0.005, format="%.3f")
        en_lo, en_hi   = st.slider("Entry Prob", 0.0, 1.0, (0.50, 0.90), step=0.01)
        ex_lo, ex_hi   = st.slider("Exit Prob",  0.0, 1.0, (0.25, 0.65), step=0.01)

    @st.cache_data(show_spinner=False)
    def _prices_for_opt(tickers, start, end, use_tail, interval, fallback, exec_key, moc):
        return load_all_prices(list(tickers), start, end, use_tail, interval, fallback, exec_key, moc)[0]

    def _sample(rng, lb_range, hz_range, thr_range, en_range, ex_range):
        return dict(
            lookback=rng.randrange(lb_range[0], lb_range[1]+1, 5),
            horizon =rng.randrange(hz_range[0], hz_range[1]+1, 1),
            thresh  =rng.uniform(*thr_range),
            entry   =rng.uniform(*en_range),
            exit    =rng.uniform(*ex_range),
        )

    def _bounded(val, lo, hi, pct):
        span = (hi - lo) * pct
        return (max(lo, val - span), min(hi, val + span))

    def _run_opt_trials(rng, n_trials, lb_r, hz_r, thr_r, en_r, ex_r,
                        price_map_opt, feasible_tickers, w_dd, status_tx, prog, offset, total):
        rows = []
        for t in range(n_trials):
            p = _sample(rng, lb_r, hz_r, thr_r, en_r, ex_r)
            if p["exit"] >= p["entry"]:
                prog.progress((offset + t + 1) / total); continue

            sharpes, winrates, cagrs, dds = [], [], [], []
            for tk in feasible_tickers:
                df = price_map_opt.get(tk)
                min_len = max(60, p["lookback"] + p["horizon"] + 5)
                if df is None or len(df) < min_len: continue
                mid = len(df) // 2
                for sub in (df.iloc[:mid], df.iloc[mid:]):
                    if len(sub) < min_len: continue
                    try:
                        _, df_bt, tr, mets = make_features_and_train(
                            sub, p["lookback"], p["horizon"], p["thresh"],
                            MODEL_PARAMS, p["entry"], p["exit"],
                            min_hold_days=int(MIN_HOLD_DAYS),
                            cooldown_days=int(COOLDOWN_DAYS),
                            walk_forward=True,
                            use_hist_gbm=USE_HIST_GBM,
                            use_ensemble=False,
                            use_calibration=False,
                            use_vol_sizing=USE_VOL_SIZING,
                            target_vol_annual=TARGET_VOL_ANNUAL,
                            wf_stride=max(int(WF_STRIDE), 5),
                        )
                        mets.pop("_feature_importance", None); mets.pop("_x_cols", None)
                        sharpes.append(mets["Sharpe-Ratio"])
                        wr = mets.get("Winrate (%)", np.nan)
                        winrates.append(wr/100. if np.isfinite(wr) else np.nan)
                        cg = mets.get("CAGR (%)", np.nan)
                        cagrs.append(cg/100. if np.isfinite(cg) else np.nan)
                        dds.append(mets.get("Max Drawdown (%)", np.nan)/100.)
                    except Exception:
                        pass

            if not sharpes: prog.progress((offset+t+1)/total); continue
            sc = _composite_score(
                float(np.nanmedian(sharpes)), float(np.nanmedian(winrates)),
                float(np.nanmedian(cagrs)),   float(np.nanmedian(dds)),
                float(w_dd),
            )
            if not np.isfinite(sc): prog.progress((offset+t+1)/total); continue
            rows.append(dict(
                score=round(sc,4), sharpe=round(float(np.nanmedian(sharpes)),3),
                winrate=round(float(np.nanmedian(winrates))*100,1),
                cagr=round(float(np.nanmedian(cagrs))*100,2),
                maxdd=round(float(np.nanmedian(dds))*100,2),
                **p,
            ))
            if rows:
                status_tx.caption(f"Bester Score: {max(r['score'] for r in rows):.3f}")
            prog.progress((offset+t+1)/total)
        return rows

    if st.button("🔎 Zweistufige Suche starten", type="primary", use_container_width=True):
        import random
        rng = random.Random(int(opt_seed))

        price_map_opt = _prices_for_opt(
            tuple(TICKERS), str(START_DATE), str(END_DATE),
            use_live, intraday_interval, fallback_last_session, exec_mode, int(moc_cutoff_min),
        )
        feasible = [tk for tk, df in price_map_opt.items() if df is not None and len(df) >= 80]

        if not feasible:
            st.warning("Keine ausreichenden Preisdaten.")
        else:
            total_trials = int(n_coarse) + int(n_fine)
            prog_opt     = st.progress(0.0)
            status_tx    = st.empty()

            # ── Stufe 1: Grobe Suche ──────────────────────────
            st.markdown("#### Stufe 1 – Grobe Suche")
            rows_coarse = _run_opt_trials(
                rng, int(n_coarse),
                (lb_lo, lb_hi), (hz_lo, hz_hi), (thr_lo, thr_hi), (en_lo, en_hi), (ex_lo, ex_hi),
                price_map_opt, feasible, float(w_dd_opt), status_tx,
                prog_opt, 0, total_trials,
            )

            if not rows_coarse:
                st.warning("Grobe Suche: keine Kandidaten.")
            else:
                df_coarse = pd.DataFrame(rows_coarse).sort_values("score", ascending=False)
                best_k    = df_coarse.head(int(top_k))
                st.success(f"Grob: Bester Score = {df_coarse['score'].iloc[0]:.3f}")
                st.dataframe(best_k.style.format({
                    "score":"{:.4f}","sharpe":"{:.3f}","winrate":"{:.1f}",
                    "cagr":"{:.2f}","maxdd":"{:.2f}","thresh":"{:.4f}",
                    "entry":"{:.3f}","exit":"{:.3f}",
                }).background_gradient(subset=["score"], cmap="RdYlGn"), use_container_width=True)

                # ── Stufe 2: Feinsuche ────────────────────────
                st.markdown("#### Stufe 2 – Feinsuche um Top-Kandidaten")
                best = df_coarse.iloc[0]
                rows_fine = _run_opt_trials(
                    rng, int(n_fine),
                    _bounded(best["lookback"], lb_lo, lb_hi, fine_band),
                    _bounded(best["horizon"],  hz_lo, hz_hi, fine_band),
                    _bounded(best["thresh"],   thr_lo, thr_hi, fine_band),
                    _bounded(best["entry"],    en_lo, en_hi, fine_band),
                    _bounded(best["exit"],     ex_lo, ex_hi, fine_band),
                    price_map_opt, feasible, float(w_dd_opt), status_tx,
                    prog_opt, int(n_coarse), total_trials,
                )
                status_tx.empty()

                all_rows = rows_coarse + rows_fine
                df_all   = pd.DataFrame(all_rows).sort_values("score", ascending=False).reset_index(drop=True)
                best_row = df_all.iloc[0]

                st.success(
                    f"✅ Finale beste Parameter — Score **{best_row['score']:.3f}** | "
                    f"Sharpe **{best_row['sharpe']:.2f}** | Winrate **{best_row['winrate']:.1f}%** | "
                    f"CAGR **{best_row['cagr']:.1f}%** | MaxDD **{best_row['maxdd']:.1f}%**"
                )
                cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                cc1.metric("Lookback",  int(best_row["lookback"]))
                cc2.metric("Horizon",   int(best_row["horizon"]))
                cc3.metric("Threshold", f"{best_row['thresh']:.3f}")
                cc4.metric("Entry",     f"{best_row['entry']:.2f}")
                cc5.metric("Exit",      f"{best_row['exit']:.2f}")

                fig_sc = px.scatter(
                    df_all, x="sharpe", y="winrate", color="score",
                    size=df_all["score"].clip(lower=0)+0.01,
                    hover_data=["lookback","horizon","thresh","entry","exit","cagr","maxdd"],
                    color_continuous_scale="RdYlGn",
                    labels={"sharpe":"Sharpe","winrate":"Winrate %","score":"Score"},
                    title="Sharpe vs. Winrate – alle Trials",
                )
                fig_sc.update_layout(height=400, margin=dict(t=40,b=40,l=40,r=20))
                st.plotly_chart(fig_sc, use_container_width=True)

                st.dataframe(
                    df_all.head(30).style.format({
                        "score":"{:.4f}","sharpe":"{:.3f}","winrate":"{:.1f}",
                        "cagr":"{:.2f}","maxdd":"{:.2f}","thresh":"{:.4f}",
                        "entry":"{:.3f}","exit":"{:.3f}",
                    }).background_gradient(subset=["score"], cmap="RdYlGn"),
                    use_container_width=True,
                )
                st.download_button(
                    "Optimierergebnisse als CSV", to_csv_eu(df_all),
                    file_name="param_search_results_v3.csv", mime="text/csv",
                )


# ─────────────────────────────────────────────────────────────
# Haupt-Pipeline
# ─────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:36px;'>📈 PROTEUS – AI Modell (v3)</h1>", unsafe_allow_html=True)

mode_lbl = "Walk-Forward ✓" if USE_WALK_FORWARD else "Volltraining ⚠️"
mdl_lbl  = MODEL_TYPE.split("(")[0].strip()
vs_lbl   = f"Vol-Sizing ({TARGET_VOL_ANNUAL*100:.0f}%)" if USE_VOL_SIZING else "Fixe Pos."
cal_lbl  = " · Kalibriert" if USE_CALIBRATION else ""
st.caption(
    f"Modus: **{mode_lbl}** | Modell: **{mdl_lbl}{cal_lbl}** | "
    f"Sizing: **{vs_lbl}** | WF-Stride: **{int(WF_STRIDE)} Tage** | "
    f"Options: **Nur Live-Overlay** (nicht im Modell)"
)

results:    List[dict]               = []
all_trades: Dict[str, List[dict]]    = {}
all_dfs:    Dict[str, pd.DataFrame]  = {}
all_feat:   Dict[str, pd.DataFrame]  = {}

price_map, meta_map = load_all_prices(
    TICKERS, str(START_DATE), str(END_DATE),
    use_live, intraday_interval, fallback_last_session,
    exec_mode, int(moc_cutoff_min),
)

# Options-Overlay laden (V3-D: as_completed)
options_live: Dict[str, pd.DataFrame] = {}
if use_chain_live:
    st.info("Optionsketten einlesen (Live-Overlay) …")
    prog_opt2 = st.progress(0.0)
    tks       = [tk for tk in price_map if not price_map[tk].empty]
    done_opt  = 0

    def _fetch_chain(tk):
        df_tk = price_map.get(tk)
        if df_tk is None or df_tk.empty: return tk, pd.DataFrame()
        try:
            return tk, get_equity_chain_aggregates_for_today(
                tk, float(df_tk["Close"].iloc[-1]), atm_band_pct, int(n_expiries), int(max_days_to_exp)
            )
        except Exception:
            return tk, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tks)))) as ex:
        fut_map_opt = {ex.submit(_fetch_chain, tk): tk for tk in tks}
        for fut in as_completed(fut_map_opt):  # V3-D
            tk_res, ch_res = fut.result()
            if not ch_res.empty:
                options_live[tk_res] = ch_res
            done_opt += 1
            prog_opt2.progress(done_opt / max(1, len(tks)))

# Parallel-Training aller Ticker
live_forecasts_run: List[dict] = []

def _run_ticker(ticker: str):
    """Worker: Features + Training. Gibt exog_tk separat zurück (nur Overlay)."""
    df   = price_map.get(ticker)
    meta = meta_map.get(ticker, {})
    if df is None or df.empty:
        return ticker, None

    # V3-G: exog_tk wird NICHT ans Modell übergeben – nur für Overlay gespeichert
    exog_tk = None
    if use_chain_live and ticker in options_live and not options_live[ticker].empty:
        ch = options_live[ticker].copy()
        ch.index = [df.index[-1].normalize()]
        exog_tk  = ch

    try:
        feat, df_bt, trades, metrics = make_features_and_train(
            df, LOOKBACK, HORIZON, THRESH, MODEL_PARAMS,
            ENTRY_PROB, EXIT_PROB,
            min_hold_days=int(MIN_HOLD_DAYS), cooldown_days=int(COOLDOWN_DAYS),
            walk_forward=USE_WALK_FORWARD,
            use_hist_gbm=USE_HIST_GBM,
            use_ensemble=USE_ENSEMBLE,
            use_calibration=USE_CALIBRATION,
            use_vol_sizing=USE_VOL_SIZING,
            target_vol_annual=TARGET_VOL_ANNUAL,
            wf_stride=int(WF_STRIDE),
        )
        return ticker, (feat, df_bt, trades, metrics, meta, exog_tk)
    except Exception as e:
        return ticker, e


valid_tickers    = [tk for tk in TICKERS if tk in price_map]
compute_results: Dict[str, object] = {}

if valid_tickers:
    st.info(f"Modell-Training für {len(valid_tickers)} Ticker … (parallel, Stride={int(WF_STRIDE)})")
    prog_train = st.progress(0.0)
    done_train = 0
    # V3-D: as_completed() für echte Parallelität
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(valid_tickers))) as ex:
        fut_map_train = {ex.submit(_run_ticker, tk): tk for tk in valid_tickers}
        for fut in as_completed(fut_map_train):
            tk_res = fut_map_train[fut]
            compute_results[tk_res] = fut.result()[1]
            done_train += 1
            prog_train.progress(done_train / len(valid_tickers))

# ── Anzeige-Schleife ──────────────────────────────────────────
def _decide_action(p: float, entry_thr: float, exit_thr: float) -> str:
    if p > entry_thr: return "Enter / Add"
    if p < exit_thr:  return "Exit / Reduce"
    return "Hold / No Trade"

for ticker in TICKERS:
    if ticker not in price_map:
        continue
    df   = price_map[ticker]
    meta = meta_map.get(ticker, {})
    _res = compute_results.get(ticker)

    with st.expander(f"🔍 Analyse für {ticker}", expanded=False):
        st.subheader(f"{ticker} — {get_ticker_name(ticker)}")
        try:
            last_timestamp_info(df, meta)

            if _res is None or isinstance(_res, Exception):
                st.error(f"Fehler bei {ticker}: {str(_res) if isinstance(_res, Exception) else 'Keine Daten'}")
                continue

            feat, df_bt, trades, metrics, meta, exog_tk = _res
            last_fi = metrics.pop("_feature_importance", None)
            x_cols  = metrics.pop("_x_cols", CORE_FEATURE_COLS)

            metrics["Ticker"] = ticker
            results.append(metrics)
            all_trades[ticker] = trades
            all_dfs[ticker]    = df_bt
            all_feat[ticker]   = feat

            live_ts    = pd.Timestamp(feat.index[-1])
            live_prob  = float(feat["SignalProb"].iloc[-1])
            live_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else np.nan
            tail_info  = "intraday" if meta.get("tail_is_intraday") else "daily"

            row = {
                "AsOf":   live_ts.strftime("%Y-%m-%d %H:%M"),
                "Ticker": ticker, "Name": get_ticker_name(ticker),
                f"P(>{THRESH:.3f} in {HORIZON}d)": round(live_prob, 4),
                "Action": _decide_action(live_prob, ENTRY_PROB, EXIT_PROB),
                "Close":  round(live_close, 4), "Bar": tail_info,
            }
            # V3-G: Options als Overlay – getrennt von Modell-Output
            if use_chain_live and exog_tk is not None:
                vals = exog_tk.iloc[-1]
                for col in ["PCR_vol","PCR_oi","VOI_call","VOI_put","IV_skew_p_minus_c","VOL_tot","OI_tot"]:
                    if col in vals and pd.notna(vals[col]):
                        row[col] = round(float(vals[col]), 4)
            live_forecasts_run.append(row)

            # KPI Tiles (V3-J: "Completed Trades")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Strategie Netto (%)",  f"{metrics['Strategy Net (%)']:.2f}")
            c2.metric("Buy & Hold (%)",        f"{metrics['Buy & Hold (%)']:.2f}")
            c3.metric("Sharpe",                f"{metrics['Sharpe-Ratio']:.2f}")
            c4.metric("Sortino",               f"{metrics['Sortino-Ratio']:.2f}" if np.isfinite(metrics["Sortino-Ratio"]) else "–")
            c5.metric("Max DD (%)",            f"{metrics['Max Drawdown (%)']:.2f}")
            c6.metric("Completed Trades",      f"{int(metrics['Completed Trades'])}")

            # Charts
            chart_cols = st.columns(2)

            # Preis + Signale
            df_plot    = feat.copy()
            price_fig  = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["Close"], mode="lines", name="Close",
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                hovertemplate="Datum: %{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            ))
            sig    = df_plot["SignalProb"]
            norm   = (sig - sig.min()) / (sig.max() - sig.min() + 1e-9)
            for idx in range(len(df_plot) - 1):
                price_fig.add_trace(go.Scatter(
                    x=df_plot.index[idx:idx+2], y=df_plot["Close"].iloc[idx:idx+2],
                    mode="lines", showlegend=False,
                    line=dict(color=px.colors.sample_colorscale(
                        px.colors.diverging.RdYlGn, float(norm.iloc[idx]))[0], width=2),
                    hoverinfo="skip",
                ))
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df["Date"] = pd.to_datetime(trades_df["Date"])
                for typ, col, sym in [("Entry","green","triangle-up"),("Exit","red","triangle-down")]:
                    sub = trades_df[trades_df["Typ"]==typ]
                    price_fig.add_trace(go.Scatter(
                        x=sub["Date"], y=sub["Price"], mode="markers", name=typ,
                        marker_symbol=sym, marker=dict(size=12, color=col),
                        hovertemplate=f"{typ}<br>%{{x|%Y-%m-%d}}<br>Preis: %{{y:.2f}}<extra></extra>",
                    ))
            price_fig.update_layout(
                title=f"{ticker}: Preis + Signal",
                height=420, margin=dict(t=50,b=30,l=40,r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            with chart_cols[0]:
                st.plotly_chart(price_fig, use_container_width=True)

            # Intraday
            intra = get_intraday_last_n_sessions(ticker, sessions=5, days_buffer=10, interval=intraday_interval)
            with chart_cols[1]:
                if intra.empty:
                    st.info("Keine Intraday-Daten verfügbar.")
                else:
                    intr_fig = go.Figure()
                    if intraday_chart_type == "Candlestick (OHLC)":
                        intr_fig.add_trace(go.Candlestick(
                            x=intra.index, open=intra["Open"], high=intra["High"],
                            low=intra["Low"], close=intra["Close"], name="OHLC",
                        ))
                    else:
                        intr_fig.add_trace(go.Scatter(
                            x=intra.index, y=intra["Close"], mode="lines", name="Close",
                        ))
                    if not trades_df.empty:
                        last_days = set(pd.Index(intra.index.normalize().unique()))
                        ev_recent = trades_df[trades_df["Date"].dt.normalize().isin(last_days)]
                        for typ, col, sym in [("Entry","green","triangle-up"),("Exit","red","triangle-down")]:
                            xs, ys = [], []
                            for d, day_sl in intra.groupby(intra.index.normalize()):
                                hit = ev_recent[(ev_recent["Typ"]==typ) & (ev_recent["Date"].dt.normalize()==d)]
                                if hit.empty: continue
                                xs.append(day_sl.index.min())
                                ys.append(float(hit["Price"].iloc[-1]))
                            if xs:
                                intr_fig.add_trace(go.Scatter(
                                    x=xs, y=ys, mode="markers", name=typ,
                                    marker_symbol=sym, marker=dict(size=11, color=col),
                                ))
                    for _, day_sl in intra.groupby(intra.index.normalize()):
                        intr_fig.add_vline(x=day_sl.index.min(), line_width=1, line_dash="dot", opacity=0.3)
                    intr_fig.update_layout(
                        title=f"{ticker}: Intraday – letzte 5 Tage ({intraday_interval})",
                        height=420, margin=dict(t=50,b=30,l=40,r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(intr_fig, use_container_width=True)

            # Equity-Kurve
            eq = go.Figure()
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=df_bt["Equity_Net"], name="Strategy Net",
                mode="lines", hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}€<extra></extra>",
            ))
            bh = INIT_CAP * df_bt["Close"] / df_bt["Close"].iloc[0]
            eq.add_trace(go.Scatter(
                x=df_bt.index, y=bh, name="Buy & Hold",
                mode="lines", line=dict(dash="dash", color="black"),
            ))
            eq.update_layout(
                title=f"{ticker}: Net Equity vs. Buy & Hold",
                height=400, margin=dict(t=50,b=30,l=40,r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(eq, use_container_width=True)

            with st.expander("📊 Feature Importance", expanded=False):
                show_feature_importance(last_fi, x_cols)
                st.caption(f"Features ({len(x_cols)}): {', '.join(x_cols)}")
                st.info("ℹ️ Optionsdaten sind NICHT in diesem Modell – sie wirken nur als Live-Overlay auf P_adj.")

            with st.expander(f"Trades für {ticker}", expanded=False):
                if not trades_df.empty:
                    df_tr = trades_df.copy()
                    df_tr["Ticker"]  = ticker
                    df_tr["Name"]    = get_ticker_name(ticker)
                    df_tr["DateStr"] = df_tr["Date"].dt.strftime("%d.%m.%Y")
                    df_tr["CumPnL"]  = (
                        df_tr.where(df_tr["Typ"]=="Exit")["NetPnL"].cumsum().ffill().fillna(0)
                    )
                    df_tr = df_tr.rename(columns={"NetPnL":"PnL","Prob":"Signal Prob",
                                                   "HoldDays":"Hold (days)","PosFrac":"Pos Frac"})
                    disp = [c for c in ["Ticker","Name","DateStr","Typ","Price","Shares",
                                        "Signal Prob","Hold (days)","Pos Frac","PnL","CumPnL","Fees"]
                            if c in df_tr.columns]
                    styled = df_tr[disp].rename(columns={"DateStr":"Date"}).style.format({
                        "Price":"{:.2f}","Shares":"{:.4f}","Signal Prob":"{:.4f}",
                        "Pos Frac":"{:.4f}","PnL":"{:.2f}","CumPnL":"{:.2f}","Fees":"{:.2f}",
                    })
                    show_styled_or_plain(df_tr[disp].rename(columns={"DateStr":"Date"}), styled)
                    st.download_button(
                        f"Trades {ticker} als CSV",
                        to_csv_eu(df_tr[disp].rename(columns={"DateStr":"Date"}), float_format="%.4f"),
                        file_name=f"trades_{ticker}.csv", mime="text/csv", key=f"dl_trades_{ticker}",
                    )
                else:
                    st.info("Keine Trades vorhanden.")

        except Exception as e:
            import traceback
            st.error(f"Fehler bei {ticker}: {e}")
            st.caption(traceback.format_exc())


# ─────────────────────────────────────────────────────────────
# Live-Forecast Board (mit Options als Overlay)
# ─────────────────────────────────────────────────────────────
if live_forecasts_run:
    live_df = (
        pd.DataFrame(live_forecasts_run)
          .drop_duplicates(subset=["Ticker"], keep="last")
          .sort_values(["AsOf","Ticker"]).reset_index(drop=True)
    )
    live_df["Target_5d"] = (pd.to_numeric(live_df["Close"], errors="coerce") * (1. + THRESH)).round(2)

    prob_col = f"P(>{THRESH:.3f} in {HORIZON}d)"
    if prob_col not in live_df.columns:
        cand = [c for c in live_df.columns if c.startswith("P(") and c.endswith("d)")]
        if cand: prob_col = cand[0]

    # V3-G: Options-Overlay klar getrennt und beschriftet
    if use_chain_live and any(c in live_df.columns for c in OPTIONS_OVERLAY_COLS):
        for c in ["PCR_oi","PCR_vol","VOI_call","VOI_put"]:
            if c in live_df.columns:
                s = pd.to_numeric(live_df[c], errors="coerce")
                live_df[c]       = s
                live_df[c+"_z"]  = (s - s.mean()) / (s.std(ddof=0) + 1e-9)

        def _col_z(name):
            return pd.to_numeric(live_df[name], errors="coerce") if name in live_df.columns else pd.Series(0., index=live_df.index)

        comp = (
            -0.6 * _col_z("PCR_oi_z").fillna(0)
            -0.3 * _col_z("PCR_vol_z").fillna(0)
            +0.5 * (_col_z("VOI_call_z").fillna(0) - _col_z("VOI_put_z").fillna(0))
        )
        p_base = pd.to_numeric(live_df[prob_col], errors="coerce").fillna(0.)
        live_df["P_adj"] = np.clip(p_base + 0.07 * comp, 0., 1.)
        live_df["Action_adj"] = live_df["P_adj"].apply(
            lambda p: "Enter / Add" if p >= ENTRY_PROB else ("Exit / Reduce" if p <= EXIT_PROB else "Hold / No Trade")
        )
        desired   = ["AsOf","Ticker","Name", prob_col, "P_adj","Action","Action_adj",
                     "PCR_oi","PCR_vol","VOI_call","VOI_put","Close","Target_5d","Bar"]
    else:
        desired   = ["AsOf","Ticker","Name", prob_col, "Action","Close","Target_5d","Bar"]
    show_cols = [c for c in desired if c in live_df.columns]

    st.markdown(f"### 🟣 Live–Forecast Board – {HORIZON}-Tage Prognose")
    st.caption("P_adj = Modell-Signal angepasst durch Options-Overlay (PCR/VOI) · Kein historisches Training mit Optionen.")
    show_styled_or_plain(live_df[show_cols], style_live_board(live_df[show_cols], prob_col, ENTRY_PROB))
    st.download_button(
        "Live-Forecasts als CSV", to_csv_eu(live_df),
        file_name=f"live_forecasts_{HORIZON}d.csv", mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
if results:
    summary_df = pd.DataFrame(results).set_index("Ticker")
    summary_df["Net P&L (%)"] = (summary_df["Net P&L (€)"] / INIT_CAP) * 100

    total_net    = summary_df["Net P&L (€)"].sum()
    total_fees   = summary_df["Fees (€)"].sum()
    total_trades = summary_df["Completed Trades"].sum()
    total_cap    = INIT_CAP * len(summary_df)

    st.subheader("📊 Summary – alle Ticker")
    c = st.columns(4)
    c[0].metric("Cumul. Net P&L (€)",   f"{total_net:,.2f}")
    c[1].metric("Cumul. Kosten (€)",     f"{total_fees:,.2f}")
    c[2].metric("Gross P&L (€)",         f"{total_net+total_fees:,.2f}")
    c[3].metric("Completed Trades ges.", f"{int(total_trades)}")

    bh_avg   = float(summary_df["Buy & Hold (%)"].dropna().mean()) if "Buy & Hold (%)" in summary_df else np.nan
    net_pct  = total_net / total_cap * 100
    c2 = st.columns(4)
    c2[0].metric("Strategy Net (%) ges.", f"{net_pct:.2f}")
    c2[1].metric("Strategy Gross (%)",    f"{(total_net+total_fees)/total_cap*100:.2f}")
    c2[2].metric("Buy & Hold (%) ∅",      f"{bh_avg:.2f}")
    c2[3].metric("CAGR (%) ∅",            f"{summary_df['CAGR (%)'].dropna().mean():.2f}" if "CAGR (%)" in summary_df else "–")

    def _color_phase(v): return f"background-color: {'#d0ebff' if v=='Open' else '#f0f0f0'};"

    styled_s = (
        summary_df.style
        .format({
            "Strategy Net (%)":"{:.2f}","Strategy Gross (%)":"{:.2f}",
            "Buy & Hold (%)":"{:.2f}","Volatility (%)":"{:.2f}",
            "Sharpe-Ratio":"{:.2f}","Sortino-Ratio":"{:.2f}",
            "Max Drawdown (%)":"{:.2f}","Calmar-Ratio":"{:.2f}",
            "Fees (€)":"{:.2f}","Net P&L (%)":"{:.2f}","Net P&L (€)":"{:.2f}",
            "CAGR (%)":"{:.2f}","Winrate (%)":"{:.2f}",
        })
        .map(lambda v: "font-weight:bold;" if isinstance(v,(int,float)) else "",
             subset=pd.IndexSlice[:,["Sharpe-Ratio","Sortino-Ratio"]])
        .map(_color_phase, subset=["Phase"])
    )
    show_styled_or_plain(summary_df, styled_s)
    st.download_button("Summary als CSV", to_csv_eu(summary_df.reset_index()),
                       file_name="strategy_summary_v3.csv", mime="text/csv")

    # ── Open Positions ────────────────────────────────────────
    st.subheader("📋 Offene Positionen")
    open_pos = []
    for tk, trd in all_trades.items():
        if trd and trd[-1]["Typ"] == "Entry":
            last_e = next(t for t in reversed(trd) if t["Typ"]=="Entry")
            open_pos.append(dict(
                Ticker=tk, Name=get_ticker_name(tk),
                EntryDate=pd.to_datetime(last_e["Date"]),
                EntryPrice=round(float(last_e["Price"]),2),
                PosFrac=round(float(last_e.get("PosFrac",np.nan)),4),
                CurrentProb=round(float(all_feat[tk]["SignalProb"].iloc[-1]),4),
                UnrealPnL=round((float(all_dfs[tk]["Close"].iloc[-1])-float(last_e["Price"]))*float(last_e["Shares"]),2),
            ))
    if open_pos:
        op_df = pd.DataFrame(open_pos).sort_values("EntryDate", ascending=False)
        op_df["EntryDate"] = op_df["EntryDate"].dt.strftime("%Y-%m-%d")
        show_styled_or_plain(op_df, op_df.style.format({
            "EntryPrice":"{:.2f}","PosFrac":"{:.4f}","CurrentProb":"{:.4f}","UnrealPnL":"{:.2f}",
        }))
        st.download_button("Offene Positionen als CSV", to_csv_eu(op_df),
                           file_name="open_positions_v3.csv", mime="text/csv")
    else:
        st.success("Keine offenen Positionen.")

    # ── Round-Trips ───────────────────────────────────────────
    rt_df = compute_round_trips(all_trades)
    if not rt_df.empty:
        st.subheader("🔁 Abgeschlossene Trades (Round-Trips)")
        rt_df["EntryDate"] = pd.to_datetime(rt_df["EntryDate"])
        rt_df["ExitDate"]  = pd.to_datetime(rt_df["ExitDate"])

        r_ticks = sorted(rt_df["Ticker"].unique().tolist())
        r1, r2, r3 = st.columns([1.1, 1.1, 1.5])

        def _fmm(s, fb=(0.,1.)):
            v = pd.to_numeric(s, errors="coerce")
            lo,hi = float(np.nanmin(v)), float(np.nanmax(v))
            return (lo,hi) if (np.isfinite(lo) and np.isfinite(hi) and lo<hi) else fb

        with r1:
            rt_sel  = st.multiselect("Ticker", r_ticks, default=r_ticks)
            hd_r    = _fmm(rt_df["HoldDays"], (0,60))
            rt_hold = st.slider("Haltedauer", int(hd_r[0]), int(hd_r[1]), (int(hd_r[0]),int(hd_r[1])), key="rt_hold")
        with r2:
            rmin, rmax = rt_df["EntryDate"].min().date(), rt_df["EntryDate"].max().date()
            rt_date    = st.date_input("Entry-Datum", (rmin, rmax), min_value=rmin, max_value=rmax, key="rt_date")
            ep_r = _fmm(rt_df["EntryProb"]); xp_r = _fmm(rt_df["ExitProb"])
            rt_ep = st.slider("Entry-Prob.", 0., 1., (max(0.,ep_r[0]),min(1.,ep_r[1])), step=0.01)
            rt_xp = st.slider("Exit-Prob.",  0., 1., (max(0.,xp_r[0]),min(1.,xp_r[1])), step=0.01)
        with r3:
            ret_r = _fmm(rt_df["ReturnPct"], (-100.,200.))
            pnl_r = _fmm(rt_df["NetPnL"], (-INIT_CAP, INIT_CAP))
            rt_ret = st.slider("Return (%)", float(ret_r[0]), float(ret_r[1]), (float(ret_r[0]),float(ret_r[1])), step=0.5)
            rt_pnl = st.slider("PnL (€)",   float(pnl_r[0]), float(pnl_r[1]), (float(pnl_r[0]),float(pnl_r[1])), step=10.)

        rds, rde = rt_date if isinstance(rt_date, tuple) else (rmin, rmax)
        mask = (
            rt_df["Ticker"].isin(rt_sel) &
            rt_df["EntryDate"].dt.date.between(rds, rde) &
            rt_df["HoldDays"].fillna(-1).between(*rt_hold) &
            rt_df["EntryProb"].fillna(0.).between(*rt_ep) &
            rt_df["ExitProb"].fillna(0.).between(*rt_xp) &
            pd.to_numeric(rt_df["ReturnPct"],errors="coerce").fillna(-9e9).between(*rt_ret) &
            pd.to_numeric(rt_df["NetPnL"],errors="coerce").fillna(-9e9).between(*rt_pnl)
        )
        rt_f = rt_df.loc[mask].copy()
        rt_f["EntryDate"] = rt_f["EntryDate"].dt.strftime("%Y-%m-%d")
        rt_f["ExitDate"]  = rt_f["ExitDate"].dt.strftime("%Y-%m-%d")

        show_styled_or_plain(rt_f, rt_f.style.format({
            "Shares":"{:.4f}","EntryPrice":"{:.2f}","ExitPrice":"{:.2f}",
            "NetPnL":"{:.2f}","Fees":"{:.2f}","ReturnPct":"{:.2f}",
            "EntryProb":"{:.4f}","ExitProb":"{:.4f}","PosFrac":"{:.4f}",
        }))
        st.download_button("Round-Trips als CSV", to_csv_eu(rt_f),
                           file_name="round_trips_v3.csv", mime="text/csv")

        # Histogramme
        st.markdown("### 📊 Verteilung")
        bins  = st.slider("Bins", 10, 100, 30, step=5, key="rt_bins")
        ret   = pd.to_numeric(rt_f.get("ReturnPct"), errors="coerce").dropna()
        pnl   = pd.to_numeric(rt_f.get("NetPnL"),    errors="coerce").dropna()

        cs = st.columns(5)
        cs[0].metric("Anzahl",   f"{len(ret)}")
        cs[1].metric("Winrate",  f"{100*(ret>0).mean():.1f}%" if len(ret) else "–")
        cs[2].metric("Ø Return", f"{ret.mean():.2f}%"  if len(ret) else "–")
        cs[3].metric("Median",   f"{ret.median():.2f}%" if len(ret) else "–")
        cs[4].metric("Std",      f"{ret.std():.2f}%"   if len(ret) else "–")

        col_h1, col_h2 = st.columns(2)
        for col, data, title, xtitle in [
            (col_h1, ret, "Return (%)", "Return (%)"),
            (col_h2, pnl, "PnL Net (€)", "PnL Net (€)"),
        ]:
            with col:
                if data.empty:
                    st.info(f"Keine {xtitle}-Werte.")
                else:
                    fig_h = go.Figure(go.Histogram(x=data, nbinsx=bins, marker_line_width=0))
                    fig_h.add_vline(x=0,              line_dash="dash", opacity=0.5)
                    fig_h.add_vline(x=float(data.mean()), line_dash="dot",  opacity=0.9)
                    fig_h.update_layout(title=f"Histogramm: {title}", height=360,
                                        margin=dict(t=40,l=40,r=20,b=40), showlegend=False)
                    st.plotly_chart(fig_h, use_container_width=True)

    # ── Korrelation ───────────────────────────────────────────
    st.markdown("### 🔗 Portfolio-Korrelation")
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.0, 1.2, 1.0])
    with cc1: corr_freq   = st.selectbox("Frequenz", ["täglich","wöchentlich","monatlich"], key="corr_freq")
    with cc2: corr_method = st.selectbox("Methode",  ["Pearson","Spearman","Kendall"],      key="corr_method")
    with cc3: min_obs     = st.slider("Min. Obs.", 3, 60, 20, key="corr_min_obs")
    with cc4: use_ffill_c = st.checkbox("FFill Lücken", True, key="corr_ffill")

    price_series = [dfbt["Close"].rename(tk) for tk, dfbt in all_dfs.items()
                    if isinstance(dfbt, pd.DataFrame) and "Close" in dfbt.columns and len(dfbt) >= 2]
    corr = None
    if len(price_series) >= 2:
        prices = pd.concat(price_series, axis=1, join="outer").sort_index()
        if use_ffill_c: prices = prices.ffill()
        if corr_freq == "wöchentlich": prices = prices.resample("W-FRI").last()
        elif corr_freq == "monatlich":  prices = prices.resample("ME").last()
        rets   = prices.pct_change().dropna(how="all")
        rets   = rets[[c for c in rets.columns if rets[c].count() >= min_obs]]
        common = rets.dropna(how="any")
        if rets.shape[1] >= 2 and len(common) >= min_obs:
            corr = rets.corr(method=corr_method.lower(), min_periods=min_obs)
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_layout(height=560, margin=dict(t=40,l=40,r=30,b=40))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(f"{len(common)} Zeitpunkte · {corr_freq} · {corr_method}")
            N   = corr.shape[0]
            tri = corr.where(~np.eye(N,dtype=bool)).stack()
            w   = np.full(N, 1./N)
            ipc = float((w @ corr.values @ w - 1./N) / (1. - 1./N))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Ø Paar-Korr.",     f"{tri.mean():.2f}")
            mc2.metric("Median",           f"{tri.median():.2f}")
            mc3.metric("Streuung (σ)",     f"{tri.std():.2f}")
            mc4.metric("Portfolio-IPC",    f"{ipc:.2f}")
        else:
            st.info("Zu wenige Überschneidungen.")
    else:
        st.info("Mindestens zwei Ticker benötigt.")

else:
    st.warning("Keine Ergebnisse. Ticker und Datenabdeckung prüfen.")
