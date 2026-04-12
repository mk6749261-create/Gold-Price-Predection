"""
=======================================================
  Gold Price Dataset Collector
=======================================================
Install dependencies first:
    pip install yfinance pandas pandas-datareader requests

Run:
    python collect_gold_data.py

Output:
    gold_dataset.csv  -- full featured dataset ready for modeling
=======================================================
"""

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# -- Config ---------------------------------------------------
START_DATE  = "2000-01-01"
END_DATE    = "2026-4-12"
OUTPUT_FILE = "gold_dataset.csv"


# ============================================================
# 1. Yahoo Finance
# ============================================================
def fetch_yahoo(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """
    Downloads the daily Close price for each ticker and returns
    them merged into a single DataFrame (one column per ticker).

    Handles both old yfinance (flat columns) and new yfinance >= 0.2
    (MultiIndex columns like ('Close', 'GC=F')).
    """
    frames = []
    for col_name, ticker in tickers.items():
        print(f"  down  {col_name:20s}  ({ticker})")
        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                print(f"      Warning: no data for {ticker}")
                continue

            # yfinance >= 0.2 returns MultiIndex columns: ('Close', 'TICKER')
            # Flatten to a simple index so we can access 'Close' normally
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            s = raw["Close"].copy()
            s.name = col_name
            frames.append(s)
        except Exception as e:
            print(f"      Failed {ticker}: {e}")
    return pd.concat(frames, axis=1)


YAHOO_TICKERS = {
    # Target
    "Gold_Price":   "GC=F",       # Gold futures price (prediction target)

    # Metals & Commodities
    "Silver":       "SI=F",       # Silver -- most correlated metal to gold
    "Oil_WTI":      "CL=F",       # Crude oil -- inflation & risk proxy
    "Platinum":     "PL=F",       # Platinum -- industrial demand signal
    "Copper":       "HG=F",       # Copper -- global economic health indicator

    # Equity Indices
    "SP500":        "^GSPC",      # S&P 500 -- risk-on/off sentiment
    "Nasdaq":       "^IXIC",      # Nasdaq -- tech & growth appetite
    "Dow_Jones":    "^DJI",       # Dow Jones -- broad market
    "MSCI_World":   "URTH",       # Global equities ETF

    # Dollar & Currencies
    "DXY":          "DX-Y.NYB",   # US Dollar Index -- inverse relationship to gold
    "EUR_USD":      "EURUSD=X",   # Euro/Dollar exchange rate
    "JPY_USD":      "JPY=X",      # Yen/Dollar -- safe-haven currency
    "GBP_USD":      "GBPUSD=X",   # British Pound/Dollar
    "CNY_USD":      "CNY=X",      # Chinese Yuan/Dollar -- demand from China

    # Bonds & Interest Rates
    "US_10Y_Yield": "^TNX",       # 10-year Treasury yield -- key gold driver
    "US_2Y_Yield":  "^IRX",       # 2-year Treasury yield
    "TIP_ETF":      "TIP",        # Inflation-protected bonds ETF

    # Fear & Liquidity Indicators
    "VIX":          "^VIX",       # Volatility Index -- fear gauge
    "Gold_ETF_GLD": "GLD",        # GLD ETF -- tracks institutional gold flows
}


# ============================================================
# 2. FRED -- Macroeconomic Data (no API key needed)
# ============================================================
def fetch_fred(series: dict, start: str, end: str) -> pd.DataFrame:
    """
    Downloads macroeconomic series from the Federal Reserve (FRED)
    via pandas-datareader. Data is monthly or quarterly -- it will be
    forward-filled to daily frequency when merged with Yahoo data.
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        print("  Warning: pandas-datareader not installed -- skipping FRED")
        return pd.DataFrame()

    frames = []
    for col_name, code in series.items():
        print(f"  down  {col_name:20s}  (FRED: {code})")
        try:
            s = web.DataReader(code, "fred", start, end).squeeze()
            s.name = col_name
            frames.append(s)
        except Exception as e:
            print(f"      Failed {code}: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


FRED_SERIES = {
    "CPI":             "CPIAUCSL",             # Consumer Price Index -- inflation
    "Fed_Rate":        "FEDFUNDS",             # Federal funds rate
    "Real_Rate_10Y":   "REAINTRATREARAT10Y",   # Real 10Y interest rate (key gold driver)
    "PPI":             "PPIACO",               # Producer Price Index
    "M2_Money":        "M2SL",                 # M2 money supply -- liquidity measure
    "Treasury_Spread": "T10Y2Y",               # 10Y-2Y yield spread (recession signal)
    "Unemployment":    "UNRATE",               # Unemployment rate
    "GDP_Growth":      "A191RL1Q225SBEA",      # Real GDP growth rate (quarterly)
    "Industrial_Prod": "INDPRO",               # Industrial production index
    "Consumer_Conf":   "UMCSENT",              # University of Michigan consumer sentiment
}


# ============================================================
# 3. Feature Engineering -- Technical Indicators
# ============================================================
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators derived from the Gold price series.
    These help the model capture momentum, trend, and volatility patterns.
    """
    gold = df["Gold_Price"]

    # Moving averages -- capture trend at different time horizons
    for w in [7, 14, 21, 50, 200]:
        df[f"MA{w}"]  = gold.rolling(w).mean()
        df[f"EMA{w}"] = gold.ewm(span=w, adjust=False).mean()

    # Price returns -- percentage change over N days
    for lag in [1, 3, 7, 14, 30]:
        df[f"Return_{lag}d"] = gold.pct_change(lag)

    # Lagged prices -- past prices as explicit features for the model
    for lag in [1, 3, 7, 14, 21]:
        df[f"Lag_{lag}d"] = gold.shift(lag)

    # RSI (14-day) -- measures overbought/oversold conditions (0-100 scale)
    delta = gold.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD -- momentum indicator based on difference of two EMAs
    ema12             = gold.ewm(span=12, adjust=False).mean()
    ema26             = gold.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands -- price envelope based on standard deviation
    rolling_mean   = gold.rolling(20).mean()
    rolling_std    = gold.rolling(20).std()
    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pos"]   = (gold - df["BB_Lower"]) / df["BB_Width"].replace(0, 1e-10)

    # Volatility -- rolling standard deviation of daily returns
    df["Volatility_7d"]  = gold.pct_change().rolling(7).std()
    df["Volatility_30d"] = gold.pct_change().rolling(30).std()

    # Momentum -- raw price difference over 14 days
    df["Momentum_14d"] = gold - gold.shift(14)

    # Gold/Silver ratio -- historically signals relative metal demand shifts
    if "Silver" in df.columns:
        df["Gold_Silver_Ratio"] = gold / df["Silver"].replace(0, 1e-10)

    # Gold/Oil ratio -- links gold to inflation and energy cost cycles
    if "Oil_WTI" in df.columns:
        df["Gold_Oil_Ratio"] = gold / df["Oil_WTI"].replace(0, 1e-10)

    # Calendar features -- capture seasonal patterns in gold demand
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month
    df["Quarter"]   = df.index.quarter
    df["Year"]      = df.index.year

    return df


# ============================================================
# 4. Target Variables
# ============================================================
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three prediction targets -- choose one depending on your task:
      - Regression     : Target_NextDay or Target_Next7d
      - Classification : Target_Direction (1 = price went up, 0 = down)
    """
    df["Target_NextDay"]   = df["Gold_Price"].shift(-1)
    df["Target_Next7d"]    = df["Gold_Price"].shift(-7)
    df["Target_Direction"] = (df["Target_NextDay"] > df["Gold_Price"]).astype(int)
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*55)
    print("  Collecting gold price dataset...")
    print("="*55)

    # Step 1 -- fetch market data from Yahoo Finance
    print("\nYahoo Finance:")
    yahoo_df = fetch_yahoo(YAHOO_TICKERS, START_DATE, END_DATE)

    # Step 2 -- fetch macroeconomic data from FRED
    print("\nFRED (macroeconomic indicators):")
    fred_df = fetch_fred(FRED_SERIES, START_DATE, END_DATE)

    # Step 3 -- merge both sources on the date index.
    # FRED data is monthly/quarterly so we resample to daily and
    # forward-fill each value until the next official reading arrives.
    print("\nMerging datasets...")
    if not fred_df.empty:
        fred_df = fred_df.resample("D").ffill()
        df = yahoo_df.join(fred_df, how="left")
        df[fred_df.columns] = df[fred_df.columns].ffill()
    else:
        df = yahoo_df.copy()

    # Step 4 -- compute all technical indicators
    print("Computing technical features...")
    df = add_technical_features(df)

    # Step 5 -- add target columns
    df = add_target(df)

    # Step 6 -- drop rows with missing Gold_Price and trim the last 7 rows
    # (those rows have no Target_Next7d value since the future data does not exist yet)
    df = df.dropna(subset=["Gold_Price"])
    df = df[:-7]

    # Step 7 -- save to CSV
    df.to_csv(OUTPUT_FILE, index=True)

    print("\n" + "="*55)
    print(f"  Saved  : {OUTPUT_FILE}")
    print(f"  Range  : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print("\n  Feature list:")
    for col in df.columns:
        print(f"    - {col}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()