import pandas as pd
import numpy as np

# -----------------------------
# Yahoo fetch
# -----------------------------
def fetch_yahoo_prices(
    tickers,
    start="1990-01-01",
    end=None,
    auto_adjust=True,
    progress=False,
):
    """
    Fetch daily price history from Yahoo for one or many tickers.
    Returns tidy DF: date, ticker, close
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Mangler yfinance. Installer: pip install yfinance") from e

    if isinstance(tickers, str):
        tickers = [tickers]

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        group_by="column",
        progress=progress,
        threads=True,
    )

    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker
        for t in tickers:
            if ("Close", t) not in raw.columns:
                continue
            tmp = raw[("Close", t)].dropna().to_frame("close")
            tmp["ticker"] = t
            tmp = tmp.reset_index().rename(columns={"Date": "date"})
            rows.append(tmp)
    else:
        # Single ticker
        if "Close" not in raw.columns:
            raise ValueError("Fant ikke 'Close' i Yahoo-data.")
        tmp = raw["Close"].dropna().to_frame("close")
        tmp["ticker"] = tickers[0]
        tmp = tmp.reset_index().rename(columns={"Date": "date"})
        rows.append(tmp)

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _safe_zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def add_forward_return(df: pd.DataFrame, price_col="close", periods=1) -> pd.DataFrame:
    df = df.copy()
    df[f"ret_fwd_{periods}"] = df[price_col].pct_change(periods=periods).shift(-periods)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["doy"] = d.dt.dayofyear
    iso = d.dt.isocalendar()
    df["iso_week"] = iso.week.astype(int)
    df["weekday"] = d.dt.weekday  # Mon=0..Sun=6
    df["year_digit"] = (df["year"] % 10).astype(int)
    return df


def build_seasonality_tables(df: pd.DataFrame, ret_col: str, buckets: list[str]) -> dict:
    x = df.dropna(subset=[ret_col]).copy()
    tables = {}
    for b in buckets:
        tables[b] = x.groupby(b)[ret_col].mean()
    return tables


def apply_seasonality_score(df: pd.DataFrame, tables: dict, buckets: list[str], score_col="seasonality_score") -> pd.DataFrame:
    df = df.copy()
    raw = 0.0
    for b in buckets:
        mu_col = f"seas_{b}_mu"
        df[mu_col] = df[b].map(tables[b])
        raw = raw + df[mu_col].fillna(0.0)
    df[score_col] = _safe_zscore(raw)
    sig_col = score_col.replace("_score", "_signal")
    df[sig_col] = 0
    df.loc[df[score_col] >= 0.7, sig_col] = 1
    df.loc[df[score_col] <= -0.7, sig_col] = -1
    return df


def to_weekly(df_ticker: pd.DataFrame, freq="W-FRI") -> pd.DataFrame:
    df = df_ticker.copy()
    df = df.sort_values("date").set_index("date")
    dfw = df.resample(freq).last().dropna(subset=["close"])
    return dfw.reset_index()


def build_daily_pack(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    buckets = ["doy", "weekday", "month", "year_digit"]
    out_all = []
    tables_by_ticker = {}
    for t in sorted(prices["ticker"].unique()):
        df = prices.loc[prices["ticker"] == t, ["date","ticker","close"]].copy()
        df = df.sort_values("date").reset_index(drop=True)
        df = add_forward_return(df, "close", periods=1)
        df = add_time_features(df)
        tables = build_seasonality_tables(df, ret_col="ret_fwd_1", buckets=buckets)
        df = apply_seasonality_score(df, tables, buckets=buckets, score_col="seasonality_score_daily")
        tables_by_ticker[t] = tables
        out_all.append(df)
    daily = pd.concat(out_all, ignore_index=True).sort_values(["ticker","date"])
    return daily, tables_by_ticker


def build_weekly_pack(prices: pd.DataFrame, weekly_freq="W-FRI") -> tuple[pd.DataFrame, dict]:
    buckets = ["iso_week", "month", "year_digit"]
    out_all = []
    tables_by_ticker = {}
    for t in sorted(prices["ticker"].unique()):
        df = prices.loc[prices["ticker"] == t, ["date","ticker","close"]].copy()
        dfw = to_weekly(df, freq=weekly_freq)
        dfw = add_forward_return(dfw, "close", periods=1)
        dfw = add_time_features(dfw)
        tables = build_seasonality_tables(dfw, ret_col="ret_fwd_1", buckets=buckets)
        dfw = apply_seasonality_score(dfw, tables, buckets=buckets, score_col="seasonality_score_weekly")
        tables_by_ticker[t] = tables
        out_all.append(dfw)
    weekly = pd.concat(out_all, ignore_index=True).sort_values(["ticker","date"])
    return weekly, tables_by_ticker


def build_daily_and_weekly_seasonality(
    tickers,
    start="1990-01-01",
    end=None,
    weekly_freq="W-FRI",
):
    prices = fetch_yahoo_prices(tickers, start=start, end=end, auto_adjust=True)
    daily_df, daily_tables = build_daily_pack(prices)
    weekly_df, weekly_tables = build_weekly_pack(prices, weekly_freq=weekly_freq)
    return prices, daily_df, weekly_df, daily_tables, weekly_tables


if __name__ == "__main__":
    tickers = ["^GSPC", "GC=F", "CL=F"]
    prices, daily_df, weekly_df, daily_tables, weekly_tables = build_daily_and_weekly_seasonality(
        tickers, start="1995-01-01"
    )
    prices.to_csv("prices_daily_raw.csv", index=False)
    daily_df.to_csv("seasonality_daily.csv", index=False)
    weekly_df.to_csv("seasonality_weekly.csv", index=False)
    print("Saved: prices_daily_raw.csv")
    print("Saved: seasonality_daily.csv")
    print("Saved: seasonality_weekly.csv")
    print("\nDAILY sample:")
    print(daily_df.tail(5)[["date","ticker","close","seasonality_score_daily","seasonality_signal_daily","year_digit"]])
    print("\nWEEKLY sample:")
    print(weekly_df.tail(5)[["date","ticker","close","seasonality_score_weekly","seasonality_signal_weekly","year_digit"]])
