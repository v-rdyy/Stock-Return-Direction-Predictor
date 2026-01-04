import pandas as pd
import yfinance as yf


def download_stock_data(stock, period):
    df = yf.download(stock, period=period)
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df