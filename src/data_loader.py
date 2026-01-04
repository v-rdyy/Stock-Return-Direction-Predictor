import pandas as pd
import yfinance as yf


def download_stock_data(stock, period):
    return yf.download(stock, period=period)