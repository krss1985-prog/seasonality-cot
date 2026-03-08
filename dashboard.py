import plotly.graph_objs as go
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import yfinance as yf

# --- File paths ---

MARKETS = {
    "SPX_Emini": {"cot": "cot_spx_emini.csv", "price": "price_spx_emini.csv", "label": "S&P 500 E-mini (^GSPC)", "yahoo": "^GSPC"},
    "GOLD": {"cot": "cot_gold.csv", "price": "price_gold.csv", "label": "Gold (GC=F)", "yahoo": "GC=F"},
    "WTI": {"cot": "cot_wti.csv", "price": "price_wti.csv", "label": "WTI (CL=F)", "yahoo": "CL=F"},
}

# --- Load data ---


def load_data(market=None, yahoo_ticker=None):
    cot_df = None
    price_df = None
    if market in MARKETS:
        cot_path = MARKETS[market]["cot"]
        price_path = MARKETS[market]["price"]
        cot_df = pd.read_csv(cot_path, parse_dates=["report_date_as_yyyy_mm_dd"])
        price_df = pd.read_csv(price_path, parse_dates=["date"])
        price_df = price_df.dropna(subset=["date"])
    elif yahoo_ticker:
        # Hent prisdata direkte fra Yahoo Finance
        df = yf.download(yahoo_ticker, start="2006-01-01", auto_adjust=True)
        if not df.empty:
            price_df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
            price_df["date"] = pd.to_datetime(price_df["date"])
        else:
            price_df = pd.DataFrame(columns=["date", "close"])
    return cot_df, price_df

def load_seasonality(ticker):
    # seasonality_weekly.csv har kolonnene: date, ticker, seasonality_signal_weekly

    df = pd.read_csv("seasonality_weekly.csv", parse_dates=["date"])
    return df[df["ticker"] == ticker].copy()


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



# --- Robust merge for pris, seasonality og COT ---
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


# --- Forbedret kontrollpanel ---
st.sidebar.header("Visningsvalg")
default_smooth = 7
smooth_window = st.sidebar.slider("Glatt signaler (uker)", 1, 26, default_smooth, key="smooth_slider")
years_back = st.sidebar.slider("Vis siste X år", 2, 20, 5, key="years_slider")
show_cot = st.sidebar.checkbox("Vis COT-signal", value=True, key="cot_checkbox", disabled=(cot_df is None))
show_season = st.sidebar.checkbox("Vis Seasonality", value=True, key="season_checkbox")
show_total = st.sidebar.checkbox("Vis Total Bias", value=True, key="total_checkbox")
show_forecast = st.sidebar.checkbox("Vis Forecast", value=True, key="forecast_checkbox")
show_rsi_comm = st.sidebar.checkbox("RSI Commercials", value=True, key="rsi_comm_checkbox", disabled=(cot_df is None))
show_rsi_noncomm = st.sidebar.checkbox("RSI Non-Commercials", value=False, key="rsi_noncomm_checkbox", disabled=(cot_df is None))
show_rsi_retail = st.sidebar.checkbox("RSI Retail", value=False, key="rsi_retail_checkbox", disabled=(cot_df is None))


# Lag glattet og total-bias kolonner
merged["cot_signal_smooth"] = merged["cot_signal"].rolling(smooth_window, min_periods=1).mean()
merged["seasonality_signal_smooth"] = merged["seasonality_signal_weekly"].rolling(smooth_window, min_periods=1).mean()
merged["total_bias"] = merged["cot_signal"] + merged["seasonality_signal_weekly"]
merged["total_bias_smooth"] = merged["total_bias"].rolling(smooth_window, min_periods=1).mean()




# --- Kutt alle datasett på dato etter at de er laget, rett før plotting ---

# Kutt på dato for merged først
date_cut = merged["date"].max() - pd.DateOffset(years=years_back)
merged = merged[merged["date"] >= date_cut]
# Kutt cot_rsi på dato etter at den er laget og glattet
# ...existing code...


# (Fjernet duplisert og feil merge-kode. Robust merge gjøres tidligere.)

# Fyll inn NaN med 0 for signaler
merged["cot_signal"] = merged["cot_signal"].fillna(0)
merged["seasonality_signal_weekly"] = merged["seasonality_signal_weekly"].fillna(0)

# --- Forbedret kontrollpanel ---
st.sidebar.header("Visningsvalg")
default_smooth = 7
smooth_window = st.sidebar.slider("Glatt signaler (uker)", 1, 26, default_smooth)
years_back = st.sidebar.slider("Vis siste X år", 2, 20, 5)
show_cot = st.sidebar.checkbox("Vis COT-signal", value=True)
show_season = st.sidebar.checkbox("Vis Seasonality", value=True)
show_total = st.sidebar.checkbox("Vis Total Bias", value=True)
show_forecast = st.sidebar.checkbox("Vis Forecast", value=True)
show_rsi_comm = st.sidebar.checkbox("RSI Commercials", value=True)
show_rsi_noncomm = st.sidebar.checkbox("RSI Non-Commercials", value=False)
show_rsi_retail = st.sidebar.checkbox("RSI Retail", value=False)

# Lag glattet og total-bias kolonner
merged["cot_signal_smooth"] = merged["cot_signal"].rolling(smooth_window, min_periods=1).mean()
merged["seasonality_signal_smooth"] = merged["seasonality_signal_weekly"].rolling(smooth_window, min_periods=1).mean()
merged["total_bias"] = merged["cot_signal"] + merged["seasonality_signal_weekly"]
merged["total_bias_smooth"] = merged["total_bias"].rolling(smooth_window, min_periods=1).mean()

# --- Forecast frem i tid ---
forecast_weeks = st.slider("Antall uker frem i tid (forecast)", 4, 52, 26)
last_date = merged["date"].max()
future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, forecast_weeks+1)]

# Bruk historisk seasonality for forecast (gjennomsnitt per uke)
if "iso_week" not in merged.columns:
    merged["iso_week"] = merged["date"].dt.isocalendar().week.astype(int)
seas_table = merged.groupby("iso_week")["seasonality_signal_weekly"].mean()
future_iso_weeks = [d.isocalendar()[1] for d in future_dates]
future_seas = [seas_table.get(w, 0) for w in future_iso_weeks]

# For COT: bruk siste kjente signal (kan evt. forbedres med modell)
last_cot = merged["cot_signal_smooth"].iloc[-1]
future_cot = [last_cot] * forecast_weeks

future_total_bias = [c + s for c, s in zip(future_cot, future_seas)]

# Lag forecast-df
forecast_df = pd.DataFrame({
    "date": future_dates,
    "cot_signal_smooth": future_cot,
    "seasonality_signal_smooth": future_seas,
    "total_bias_smooth": future_total_bias,
})
# Kutt forecast_df på dato etter at den er laget
forecast_df = forecast_df[forecast_df["date"] >= date_cut]


# --- COT RSI subplot for alle grupper ---
cot_rsi = None
if cot_df is not None:
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

# --- Kutt alle datasett på dato etter at de er laget, rett før plotting ---

date_cut = merged["date"].max() - pd.DateOffset(years=years_back)
merged = merged[merged["date"] >= date_cut]
if cot_rsi is not None:
    cot_rsi = cot_rsi[cot_rsi["date"] >= date_cut]
forecast_df = forecast_df[forecast_df["date"] >= date_cut]

# Lag forecast-df
forecast_df = pd.DataFrame({
    "date": future_dates,
    "cot_signal_smooth": future_cot,
    "seasonality_signal_smooth": future_seas,
    "total_bias_smooth": future_total_bias,
})





# --- COT RSI subplot for alle grupper ---
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


# --- Plot med forecast og subplot ---
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.07,
    subplot_titles=(f"Pris, COT, Seasonality og Total Bias for {ticker} (med forecast)", "COT RSI (glattet)")
)
 # Hovedchart
if "close" in merged.columns:
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["close"], name="Pris", yaxis="y1", line=dict(width=2, color="#222")), row=1, col=1)
else:
    st.error("Prisdata mangler for valgt ticker. Sjekk at Yahoo-ticker er korrekt og at det finnes prisdata.")
if show_cot and cot_df is not None:
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["cot_signal_smooth"], name="COT Signal (glattet)", yaxis="y2", line=dict(color="orange", width=2, shape="hv")), row=1, col=1)
if show_season:
    # Seasonality som histogram (historisk)
    fig.add_trace(go.Bar(x=merged["date"], y=merged["seasonality_signal_smooth"], name="Seasonality Histogram", yaxis="y2", marker_color="blue", opacity=0.3), row=1, col=1)
    # Seasonality forecast som histogram (fremover i tid)
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
fig.update_yaxes(title_text="Signal", row=1, col=1, side="right", overlaying="y", range=[-2.2,2.2])
fig.update_yaxes(title_text="RSI (COT)", row=2, col=1, range=[0,100])
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
st.info("Dashboardet bruker cot_*.csv, price_*.csv og seasonality_weekly.csv. Kjør cot_and_price_fetch.py og yahoo_seasonality_pack.py først hvis du mangler data.")
