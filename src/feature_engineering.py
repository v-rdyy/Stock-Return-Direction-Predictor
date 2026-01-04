import pandas as pd

def create_features(df):
    df['returns'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['momentum'] = df['Close'].pct_change(periods=5)
    df = df.dropna()
    return df

def create_labels(df):
    df['next_return'] = df['returns'].shift(-1)
    df['label'] = (df['next_return'] > 0).astype(int)
    df = df.dropna()
    return df