import pandas as pd
import numpy as np

def compute_rsi(data, n=14):
    df = data.copy()
    close = df['Close'].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # wilder smoothing
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # edge cases handling
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)

    df[f'rsi_{n}'] = rsi
    return df

def rsi_position_state(data, n=14, lo=30, mid=50, hi=70):
    df = data.copy()
    r = df[f'rsi_{n}']

    pos = np.zeros(len(df), dtype=float)
    state = 0

    for i in range(len(df)):
        x = r.iat[i]
        if np.isnan(x):
            pos[i] = state
            continue

        if state == 0:
            if x < lo:
                state = 1
            elif x > hi:
                state = -1
        elif state == 1:
            if x >= mid:
                state = 0
        else: # state == -1
            if  x <= mid:
                state = 0

        pos[i] = state

    df['pos_raw'] = pos
    return df

def rsi_strategy(data, n=14, lo=30, mid=50, hi=70):
    df = compute_rsi(data, n)
    df = rsi_position_state(df, n, lo, mid, hi)
    return df