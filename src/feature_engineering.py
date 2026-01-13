import pandas as pd

def create_features(df):
    """
    Creates features from raw OHLCV data.
    All features use only past data (no look-ahead bias).
    """
    # Basic features - daily returns and moving averages
    df['returns'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['momentum'] = df['Close'].pct_change(periods=5)

    # RSI calculation - momentum indicator
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)  # Only positive changes
    loss = -delta.where(delta < 0, 0)  # Only negative changes (made positive)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))  # RSI ranges from 0-100

    # Price relative to moving average - shows if price is above/below trend
    df['price_to_ma'] = df['Close'] / df['ma20']
    
    # Volume ratio - shows if volume is high/low relative to average
    df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma20']

    # Drop rows with NaN (from rolling windows at the start)
    df = df.dropna()
    return df

def create_labels(df):
    """
    Creates binary labels: 1 if next day goes up, 0 if down.
    Uses shift(-1) to get tomorrow's return for today's row.
    """
    df['next_return'] = df['returns'].shift(-1)  # Tomorrow's return
    df['label'] = (df['next_return'] > 0).astype(int)  # 1 if up, 0 if down
    df['return_target'] = df['next_return']
    df = df.dropna()  # Drop last row (has no tomorrow)
    return df