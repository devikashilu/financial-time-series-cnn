import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_financial_data(ticker_symbol: str, years: int = 4):
    """
    Fetches financial data to form a multivariate signal:
    We use [Close, Volume, High, Low, Market_Index] as the 5 variables for the CNN.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    # Fetch primary stock
    ticker = yf.download(ticker_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    # Fetch Market Index (S&P 500)
    market_index = yf.download('^GSPC', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    
    # Flatten multi-index columns if returned by yfinance v0.2.x+
    if isinstance(ticker.columns, pd.MultiIndex):
        ticker.columns = ticker.columns.get_level_values(0)
    if isinstance(market_index.columns, pd.MultiIndex):
        market_index.columns = market_index.columns.get_level_values(0)

    # Compile the signal DataFrame
    signal_df = pd.DataFrame({
        'Close': ticker['Close'],
        'Volume': ticker['Volume'],
        'High': ticker['High'],
        'Low': ticker['Low'],
        'Market_Index': market_index['Close']
    })
    
    # Forward fill missing values then drop any remaining
    signal_df.ffill(inplace=True)
    signal_df.dropna(inplace=True)
    
    # Normalization (Min-Max Scaling)
    signal_min = signal_df.min()
    signal_max = signal_df.max()
    signal_norm = (signal_df - signal_min) / (signal_max - signal_min)
    
    return signal_df, signal_norm, signal_min, signal_max

def get_target_prices(df: pd.DataFrame, horizon: int = 1):
    """
    Returns the target prices (Close price shifted by `horizon` days).
    """
    # E.g., horizon=1 means predict next day's close
    target = df['Close'].shift(-horizon)
    return target
