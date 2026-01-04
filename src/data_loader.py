import pandas as pd
import yfinance as yf


def download_stock_data(stock, period):
    """
    Downloads stock price data from Yahoo Finance.
    
    Args:
        stock: Stock ticker symbol (e.g., 'AAPL')
        period: Time period like '1y', '2y', '5y'
    
    Returns:
        DataFrame with OHLCV columns
    """
    df = yf.download(stock, period=period)
    # yfinance sometimes returns MultiIndex columns - flatten them if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df