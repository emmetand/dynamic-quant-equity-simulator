# Make Imports
import os, datetime as dt
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Create constants where we will store Data Raw and Data Processed
DATA_RAW = "data/raw"
DATA_PROC = "data/processed"

# Creates the folders for storing data, or skips if it already exists
# Why? So when we save CSVs later, we don’t crash because folders are missing. Phew! 
def ensure_dirs():
    os.makedirs(DATA_RAW, exist_ok=True)
    os.makedirs(DATA_PROC, exist_ok=True)

#Create a function to get the universe of stocks I care about
def get_universe(): 
    """Function returns the list of tickers that define the “investment universe” (the stocks we're modeling)."""
    # hard code list of ticket symbols we want
    # doing this so other functions can just call this one instead of duplicating it. 
    tickers = [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","BRK-B","XOM","JNJ","JPM",
        "V","PG","AVGO","UNH","HD","MA","LLY","KO","PFE","PEP","ABBV","BAC","COST",
        "CSCO","ADBE","TMO","MCD","NFLX","DIS","CRM","WMT","NKE","TXN","INTC","AMD",
        "QCOM","AMAT","LIN","ACN","PM","DHR","UPS","MS","GS","BLK","CAT","HON","IBM",
        "BKNG","ORCL","BA","SPGI","GE","AMT","LOW","MDLZ","SBUX","CVX","NOW","ISRG"
    ]
    return tickers 

# create a function to get historical close prices
def download_prices(tickers, start = "2018-01-01"):
    """Price Data Ingestion Step of ETL! Take a list of tickets and optional start date, return df of prices"""
    # fetch OHLCV data (Open/High/Low/Close/Volume) from Yahoo. Only keep 'Close' prices column. 
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        # If only 1 ticker is passed then df might be a series. checks if this is true and adjusts to df.
        df = df.to_frame()
    df.index.name = "date"
    # Returns DF with rows = dates, columns = tickers, values = adjusted closing prices
    return df

# turn prices into returns! 
def compute_returns(prices):
    #calculates percent change between each row and the previous row. Drop first Null row
    returns = prices.pct_change().dropna()
    returns.index.name = "date" 
    # This DataFrame is now a time series of daily returns
    return returns

def get_macro(fred_key):
    """get macroeconomic series from FRED"""
    # function input expects fred api key
    # create Fred object
    fred = Fred(api_key=fred_key)
    # map the column names
    series = {
        "FEDFUNDS": "fed_funds",
        "CPIAUCSL": "cpi",
        "DGS10": "treasury_10y"
    }
    frames = []
    # creates a data frame of the column names. 
    for code, name in series.items():
        s = fred.get_series(code)
        s = s.rename(name).to_frame()
        frames.append(s)
    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    # Forward-fills the last known value through the days until the next data point. 
    # This is crucial so we can join daily returns with macro data cleanly.
    macro = macro.resample("D").ffill()
    # daily macro series aligned by date, ready to be merged with equity data.
    return macro

def main():

    # set up
    ensure_dirs()
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY", "")

    # get the equities prices and returns 
    tickers = get_universe()
    prices = download_prices(tickers)
    prices.to_csv(f"{DATA_RAW}/prices.csv")

    rets = compute_returns(prices)
    rets.to_csv(f"{DATA_PROC}/daily_returns.csv")

    # get and save FRED data if we have a fred key. 
    if fred_key:
        macro = fetch_macro(fred_key)
        macro.to_csv(f"{DATA_RAW}/macro.csv")

    # quick sanity merge for aligned dataset (Week 1 deliverable)
    # start with daily returns as base table. 
    df = rets.copy()
    # gives a single dataset where each date has: stock returns (columns per ticker), macro features (extra columns)
    if fred_key:
        df = df.join(macro, how="left").ffill()
    
    
    # simple factor seeds (momentum 30/90d, volatility 30d)
    # Momentum: how much it has run
    # Volatility: how bumpy it’s been

    mom30 = prices.pct_change(30)
    mom90 = prices.pct_change(90)
    vol30 = rets.rolling(30).std()

    mom30.columns = [f"{c}_mom30" for c in mom30.columns]
    mom90.columns = [f"{c}_mom90" for c in mom90.columns]
    vol30.columns = [f"{c}_vol30" for c in vol30.columns]


    # Combine everything into a single features table 
    # Concatenate all three factor DataFrames horizontally (side by side).
    features = pd.concat([mom30, mom90, vol30], axis=1).dropna(how="all")
    features.index.name = "date"
    features.to_csv(f"{DATA_PROC}/features_basic.csv")

    # Main starting artifact: an aligned factor dataset you can feed into modeling.

    print("✅ Data pipeline complete:",
          f"\n  prices -> {DATA_RAW}/prices.csv",
          f"\n  returns -> {DATA_PROC}/daily_returns.csv",
          f"\n  features -> {DATA_PROC}/features_basic.csv",
          f"\n  macro -> {DATA_RAW}/macro.csv (if key provided)")
    

if __name__ == "__main__":
    main()