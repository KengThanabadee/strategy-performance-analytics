import pandas as pd
import numpy as np

def compute_macd(data ,fast_ema=12, slow_ema=26, signal_ema=9):
    df = data.copy()
    ema_fast = df['Close'].ewm(span=fast_ema, adjust=False, min_periods=fast_ema).mean()
    ema_slow = df['Close'].ewm(span=slow_ema, adjust=False, min_periods=slow_ema).mean()

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_ema, adjust=False, min_periods=signal_ema).mean()
    histogram = macd - signal
    
    df['macd'] = macd
    df['signal'] = signal
    df['histogram'] = histogram
    return df

def generate_macd_events(data):
    df = data.copy()
    h = df['histogram']
    h1 = h.shift(1)

    df['event'] = np.where((h > 0) & (h1 <= 0), 1,
                        np.where((h < 0) & (h1 >= 0), -1, 0))
    return df

def events_to_position(data):
    df = data.copy()
    df['pos_raw'] = df['event'].replace(0, np.nan).ffill().fillna(0)
    return df

def macd_strategy(data, fast_ema=12, slow_ema=26, signal_ema=9):
    df = compute_macd(data, fast_ema, slow_ema, signal_ema)
    df = generate_macd_events(df)
    df = events_to_position(df)
    return df