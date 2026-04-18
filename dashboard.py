import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
import io
from plotly.subplots import make_subplots
import yfinance as yf

# --- Config ---
SOCRATA_DATASET = "6dca-aqww"
BASE_URL = f"https://publicreporting.cftc.gov/resource/{SOCRATA_DATASET}.csv"
RSI_LEN = 14
CI_LEN = 52

MARKETS = {
    "SPX_Emini": {"cot": "cot_spx_emini.csv", "price": "price_spx_emini.csv", "label": "S&P 500 E-mini (^GSPC)", "yahoo": "^GSPC", "cot_code": "13874A"},
    "GOLD": {"cot": "cot_gold.csv", "price": "price_gold.csv", "label": "Gold (GC=F)", "yahoo": "GC=F", "cot_code": "088691"},
    "WTI": {"cot": "cot_wti.csv", "price": "price_wti.csv", "label": "WTI (CL=F)", "yahoo": "CL=F", "cot_code": "067651"},
}

# --- Indicator functions ---
def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).astype("float64")

def commitment_index(series, length=52):
    roll_min = series.rolling(length).min()
    roll_max = series.rolling(length).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((series - roll_min) / denom * 100).astype("float64")

def cot_signal_calc(ci, rsi_val):
    sig = pd.Series(0, index=ci.index)
    sig[(ci < 20) & (rsi_val < 40)] = 1
    sig[(ci > 80) & (rsi_val > 60)] = -1
    return sig

# --- Live data fetching with caching (refresh every 6 hours) ---
@st.cache_data(ttl=21600)
def fetch_cot_live(contract_code, start_date="2006-01-01"):
    select_fields = [
        "report_date_as_yyyy_mm_dd",
        "cftc_contract_market_code",
        "market_and_exchange_names",
        "open_interest_all",
        "noncomm_positions_long_all",
        "noncomm_positions_short_all",
        "comm_positions_long_all",
        "comm_positions_short_all",
        "nonrept_positions_long_all",
        "nonrept_positions_short_all",
    ]
    where = (
        f"cftc_contract_market_code='{contract_code}' "
        f"AND report_date_as_yyyy_mm_dd >= '{start_date}'"
    )
    params = {
        "$select": ",".join(select_fields),
        "$where": where,
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 50000,
    }
    url = BASE_URL + "?" + urlencode(params)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    num_cols = [c for c in df.columns if c not in ("report_date_as_yyyy_mm_dd", "market_and_exchange_names", "cftc_contract_market_code")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["noncomm_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["comm_net"]    = df["comm_positions_long_all"]    - df["comm_positions_short_all"]
    df["nonrept_net"] = df["nonrept_positions_long_all"] - df["nonrept_positions_short_all"]
    df["rsi_noncomm_net"] = rsi(df["noncomm_net"], RSI_LEN)
    df["rsi_comm_net"]    = rsi(df["comm_net"], RSI_LEN)
    df["rsi_nonrept_net"] = rsi(df["nonrept_net"], RSI_LEN)
    df["ci_noncomm_net"] = commitment_index(df["noncomm_net"], CI_LEN)
    df["ci_comm_net"]    = commitment_index(df["comm_net"], CI_LEN)
    df["ci_nonrept_net"] = commitment_index(df["nonrept_net"], CI_LEN)
    df["cot_signal"] = cot_signal_calc(df["ci_comm_net"], df["rsi_comm_net"])
    return df

@st.cache_data(ttl=21600)
def fetch_price_live(yahoo_ticker, start="2006-01-01"):
    df = yf.download(yahoo_ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    price_df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df["close"] = price_df["close"].squeeze()
    return price_df

# --- Load data ---
def load_data(market=None, yahoo_ticker=None):
    cot_df = None
    price_df = None
    if market in MARKETS:
        info = MARKETS[market]
        try:
            cot_df = fetch_cot_live(info["cot_code"])
        except Exception as e:
            st.warning(f"Kunne ikke hente live COT-data: {e}. Bruker lagret data.")
            cot_df = pd.read_csv(info["cot"], parse_dates=["report_date_as_yyyy_mm_dd"])
        try:
            price_df = fetch_price_live(info["yahoo"])
        except Exception as e:
            st.warning(f"Kunne ikke hente live prisdata: {e}. Bruker lagret data.")
            price_df = pd.read_csv(info["price"], parse_dates=["date"])
        if price_df is not None:
            price_df = price_df.dropna(subset=["date"])
    elif yahoo_ticker:
        try:
            price_df = fetch_price_live(yahoo_ticker)
        except Exception:
            price_df = pd.DataFrame(columns=["date", "close"])
    return cot_df, price_df

def load_seasonality(ticker):
    df = pd.read_csv("seasonality_weekly.csv", parse_dates=["date"])
    return df[df["ticker"] == ticker].copy()

# --- Sidebar ---
st.sidebar.header("Data og ticker")
use_custom = st.sidebar.checkbox("Bruk egendefinert Yahoo-ticker", value=False)
if use_custom:
    yahoo_ticker = st.sidebar.text_input("Yahoo-ticker (f.eks. AAPL, MSFT, SPY, BTC-USD)", value="^GSPC")
    cot_df, price_df = load_data(yahoo_ticker=yahoo_ticker)
    market = None
    ticker = yahoo_ticker
    seasonality_df = load_seasonality(yahoo_ticker)
else:
    market = st.selectbox("Velg marked", list(MARKETS.keys()), format_func=lambda x: MARKETS[x]["label"])
    cot_df, price_df = load_data(market)
    ticker = MARKETS[market]["yahoo"]
    seasonality_map = {"SPX_Emini": "^GSPC", "GOLD": "GC=F", "WTI": "CL=F"}
    seasonality_df = load_seasonality(seasonality_map[market])

st.sidebar.header("Visningsvalg")
smooth_window = st.sidebar.slider("Glatt signaler (uker)", 1, 26, 7)
years_back = st.sidebar.slider("Vis siste X år", 2, 20, 5)
show_cot = st.sidebar.checkbox("Vis COT-signal", value=True, disabled=(cot_df is None))
show_season = st.sidebar.checkbox("Vis Seasonality", value=True)
show_total = st.sidebar.checkbox("Vis Total Bias", value=True)
show_forecast = st.sidebar.checkbox("Vis Forecast", value=True)
show_rsi_comm = st.sidebar.checkbox("RSI Commercials", value=True, disabled=(cot_df is None))
show_rsi_noncomm = st.sidebar.checkbox("RSI Non-Commercials", value=False, disabled=(cot_df is None))
show_rsi_retail = st.sidebar.checkbox("RSI Retail", value=False, disabled=(cot_df is None))

# --- Merge price, seasonality and COT ---
merged = price_df.sort_values("date").copy()
if not seasonality_df.empty:
    merged = pd.merge_asof(
        merged,
        seasonality_df.sort_values("date"),
        left_on="date", right_on="date",
        direction="backward"
    )
    merged["seasonality_signal_weekly"] = merged["seasonality_signal_weekly"].fillna(0)
else:
    merged["seasonality_signal_weekly"] = 0

if cot_df is not None and not cot_df.empty:
    merged = pd.merge_asof(
        merged,
        cot_df.sort_values("report_date_as_yyyy_mm_dd"),
        left_on="date", right_on="report_date_as_yyyy_mm_dd",
        direction="backward"
    )
    merged["cot_signal"] = merged["cot_signal"].fillna(0)
else:
    merged["cot_signal"] = 0

merged["cot_signal"] = merged["cot_signal"].fillna(0)
merged["seasonality_signal_weekly"] = merged["seasonality_signal_weekly"].fillna(0)

# Smoothing and signals
merged["cot_signal_smooth"] = merged["cot_signal"].rolling(smooth_window, min_periods=1).mean()
merged["seasonality_signal_smooth"] = merged["seasonality_signal_weekly"].rolling(smooth_window, min_periods=1).mean()
merged["total_bias"] = merged["cot_signal"] + merged["seasonality_signal_weekly"]
merged["total_bias_smooth"] = merged["total_bias"].rolling(smooth_window, min_periods=1).mean()

# --- Forecast ---
forecast_weeks = st.slider("Antall uker frem i tid (forecast)", 4, 52, 26)
last_date = merged["date"].max()
future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, forecast_weeks + 1)]

if "iso_week" not in merged.columns:
    merged["iso_week"] = merged["date"].dt.isocalendar().week.astype(int)
seas_table = merged.groupby("iso_week")["seasonality_signal_weekly"].mean()
future_iso_weeks = [d.isocalendar()[1] for d in future_dates]
future_seas = [seas_table.get(w, 0) for w in future_iso_weeks]

last_cot = merged["cot_signal_smooth"].iloc[-1]
future_cot = [last_cot] * forecast_weeks
future_total_bias = [c + s for c, s in zip(future_cot, future_seas)]

forecast_df = pd.DataFrame({
    "date": future_dates,
    "cot_signal_smooth": future_cot,
    "seasonality_signal_smooth": future_seas,
    "total_bias_smooth": future_total_bias,
})

# --- COT RSI ---
cot_rsi = None
if cot_df is not None:
    needed_cols = {"report_date_as_yyyy_mm_dd", "rsi_comm_net", "rsi_noncomm_net", "rsi_nonrept_net"}
    if needed_cols.issubset(set(cot_df.columns)):
        cot_rsi = cot_df[[
            "report_date_as_yyyy_mm_dd",
            "rsi_comm_net",
            "rsi_noncomm_net",
            "rsi_nonrept_net"
        ]].copy()
        cot_rsi = cot_rsi.rename(columns={"report_date_as_yyyy_mm_dd": "date"})
        cot_rsi = cot_rsi.sort_values("date")
        cot_rsi["rsi_comm_net_smooth"] = cot_rsi["rsi_comm_net"].rolling(smooth_window, min_periods=1).mean()
        cot_rsi["rsi_noncomm_net_smooth"] = cot_rsi["rsi_noncomm_net"].rolling(smooth_window, min_periods=1).mean()
        cot_rsi["rsi_nonrept_net_smooth"] = cot_rsi["rsi_nonrept_net"].rolling(smooth_window, min_periods=1).mean()

# --- Cut all datasets by date before plotting ---
date_cut = merged["date"].max() - pd.DateOffset(years=years_back)
merged = merged[merged["date"] >= date_cut]
if cot_rsi is not None:
    cot_rsi = cot_rsi[cot_rsi["date"] >= date_cut]
forecast_df = forecast_df[forecast_df["date"] >= date_cut]

# --- Plot ---
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.07,
    subplot_titles=(f"Pris, COT, Seasonality og Total Bias for {ticker} (med forecast)", "COT RSI (glattet)")
)
if "close" in merged.columns:
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["close"], name="Pris", yaxis="y1", line=dict(width=2, color="#222")), row=1, col=1)
else:
    st.error("Prisdata mangler for valgt ticker. Sjekk at Yahoo-ticker er korrekt og at det finnes prisdata.")
if show_cot and cot_df is not None:
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["cot_signal_smooth"], name="COT Signal (glattet)", yaxis="y2", line=dict(color="orange", width=2, shape="hv")), row=1, col=1)
if show_season:
    fig.add_trace(go.Bar(x=merged["date"], y=merged["seasonality_signal_smooth"], name="Seasonality Histogram", yaxis="y2", marker_color="blue", opacity=0.3), row=1, col=1)
    if show_forecast:
        fig.add_trace(go.Bar(x=forecast_df["date"], y=forecast_df["seasonality_signal_smooth"], name="Seasonality Forecast Histogram", yaxis="y2", marker_color="blue", opacity=0.15), row=1, col=1)
if show_total:
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["total_bias_smooth"], name="Total Bias (glattet)", yaxis="y2", line=dict(color="green", dash="dash", width=3)), row=1, col=1)
if show_forecast:
    if cot_df is not None and show_cot:
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["cot_signal_smooth"], name="COT Forecast", yaxis="y2", line=dict(color="orange", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["seasonality_signal_smooth"], name="Seasonality Forecast", yaxis="y2", line=dict(color="blue", dash="dot", width=1), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["total_bias_smooth"], name="Total Bias Forecast", yaxis="y2", line=dict(color="green", dash="dot", width=1), opacity=0.7), row=1, col=1)
if cot_rsi is not None:
    if show_rsi_comm:
        fig.add_trace(go.Scatter(x=cot_rsi["date"], y=cot_rsi["rsi_comm_net_smooth"], name="Commercials RSI", line=dict(color="purple", width=2)), row=2, col=1)
    if show_rsi_noncomm:
        fig.add_trace(go.Scatter(x=cot_rsi["date"], y=cot_rsi["rsi_noncomm_net_smooth"], name="Non-Commercials RSI", line=dict(color="red", width=1)), row=2, col=1)
    if show_rsi_retail:
        fig.add_trace(go.Scatter(x=cot_rsi["date"], y=cot_rsi["rsi_nonrept_net_smooth"], name="Retail RSI", line=dict(color="gray", width=1)), row=2, col=1)
fig.update_yaxes(title_text="Pris", row=1, col=1, side="left")
fig.update_yaxes(title_text="Signal", row=1, col=1, side="right", overlaying="y", range=[-2.2, 2.2])
fig.update_yaxes(title_text="RSI (COT)", row=2, col=1, range=[0, 100])
fig.update_xaxes(title_text="Dato", row=2, col=1)
fig.update_layout(
    legend=dict(orientation="h"),
    height=900
)
st.plotly_chart(fig, use_container_width=True)

# --- Show data ---
st.subheader("Data (siste 20 rader)")
st.dataframe(merged.tail(20))

if cot_df is None:
    st.warning("COT-data vises kun for forhåndsdefinerte futures-markeder (S&P 500 E-mini, Gold, WTI). For andre tickere vises kun pris og seasonality.")
st.info("Dashboardet henter COT-data direkte fra CFTC og prisdata fra Yahoo Finance (oppdateres automatisk hver 6. time).")
