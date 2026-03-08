
import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
import io

# -----------------------------
# CONFIG
# -----------------------------
SOCRATA_DATASET = "6dca-aqww"  # CFTC Public Reporting: Legacy - Futures Only (Socrata)
BASE_URL = f"https://publicreporting.cftc.gov/resource/{SOCRATA_DATASET}.csv"

MARKETS = {
    "SPX_Emini": {"cot_code": "13874A", "yahoo": "^GSPC"},   # E-MINI S&P 500 - CME
    "GOLD": {"cot_code": "088691", "yahoo": "GC=F"},        # GOLD - COMEX
    "WTI": {"cot_code": "067651", "yahoo": "CL=F"},         # WTI-PHYSICAL - NYMEX
}

RSI_LEN = 14
CI_LEN = 52

# -----------------------------
# INDICATORS
# -----------------------------
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.astype("float64")

def commitment_index(series: pd.Series, length: int = 52) -> pd.Series:
    roll_min = series.rolling(length).min()
    roll_max = series.rolling(length).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    return ((series - roll_min) / denom * 100).astype("float64")

def cot_signal(ci: pd.Series, rsi_val: pd.Series) -> pd.Series:
    sig = pd.Series(0, index=ci.index)
    sig[(ci < 20) & (rsi_val < 40)] = 1
    sig[(ci > 80) & (rsi_val > 60)] = -1
    return sig

# -----------------------------
# FETCH COT
# -----------------------------
def fetch_cot_legacy(contract_code: str, start_date: str = "2006-01-01") -> pd.DataFrame:
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
    return df

def build_cot_rsi_pack(contract_code: str, start_date: str = "2006-01-01", rsi_len: int = 14, ci_len: int = 52) -> pd.DataFrame:
    df = fetch_cot_legacy(contract_code, start_date=start_date)
    df["noncomm_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["comm_net"]    = df["comm_positions_long_all"]    - df["comm_positions_short_all"]
    df["nonrept_net"] = df["nonrept_positions_long_all"] - df["nonrept_positions_short_all"]
    df["rsi_noncomm_net"] = rsi(df["noncomm_net"], rsi_len)
    df["rsi_comm_net"]    = rsi(df["comm_net"], rsi_len)
    df["rsi_nonrept_net"] = rsi(df["nonrept_net"], rsi_len)
    df["ci_noncomm_net"] = commitment_index(df["noncomm_net"], ci_len)
    df["ci_comm_net"]    = commitment_index(df["comm_net"], ci_len)
    df["ci_nonrept_net"] = commitment_index(df["nonrept_net"], ci_len)
    df["cot_signal"] = cot_signal(df["ci_comm_net"], df["rsi_comm_net"])
    return df

# -----------------------------
# YAHOO PRISDATA
# -----------------------------
def fetch_yahoo_prices(ticker, start="1990-01-01", end=None, auto_adjust=True):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=auto_adjust)
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df

if __name__ == "__main__":
    for name, info in MARKETS.items():
        try:
            print(f"Henter COT for {name} ({info['cot_code']}) ...")
            cot_df = build_cot_rsi_pack(info["cot_code"], start_date="2006-01-01")
            cot_df.to_csv(f"cot_{name.lower()}.csv", index=False)
            print(f"Saved: cot_{name.lower()}.csv")
        except Exception as e:
            print(f"FEIL ved henting/lagring av COT for {name}: {e}")
        try:
            print(f"Henter prisdata fra Yahoo for {info['yahoo']} ...")
            price_df = fetch_yahoo_prices(info["yahoo"], start="2006-01-01")
            price_df.to_csv(f"price_{name.lower()}.csv", index=False)
            print(f"Saved: price_{name.lower()}.csv")
        except Exception as e:
            print(f"FEIL ved henting/lagring av prisdata for {name}: {e}")
