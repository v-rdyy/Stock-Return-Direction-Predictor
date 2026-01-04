from platform import win32_edition
import pandas as pd

def create_features(df):
    df['returns'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['momentum'] = df['Close'].pct_change(periods=5)

    delta = df['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    # RSI: Momentum oscillator (0-100), indicates overbought (>70) or oversold (<30) conditions
    df['rsi'] = 100 - (100 / (1 + rs))

    # Price to MA ratio: Close price relative to 20-day moving average, shows trend position
    df['price_to_ma'] = df['Close'] / df['ma20']

    # Volume ratio: Current volume relative to 20-day average, indicates trading activity strength
    df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma20']   

    df = df.dropna()
    return df

def create_labels(df):
    df['next_return'] = df['returns'].shift(-1)
    df['label'] = (df['next_return'] > 0).astype(int)
    df = df.dropna()
    return df