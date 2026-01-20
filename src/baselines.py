import numpy as np
import pandas as pd

def buy_and_hold(data: pd.DataFrame, side: int = 1) -> pd.DataFrame:
    df = data.copy()
    if side not in (-1, 1):
        raise ValueError('side must be -1 or 1')
    df['pos_raw'] = float(side)
    return df

def random_positions(
    data: pd.DataFrame,
    seed: int = 42,
    p_long: float = 0.5,
    p_short: float = 0.0,
    p_flat: float | None = None
) -> pd.DataFrame:
    df = data.copy()

    if p_flat is None:
        p_flat = 1.0 - p_long - p_short

    probs = np.array([p_short, p_flat, p_long], dtype=float)
    if (probs < 0).any() or not np.isclose(probs.sum(), 1.0):
        raise ValueError('p_long + p_short + p_flat must sum to 1.0 and be non-negative')

    rng = np.random.default_rng(seed)
    choices = rng.choice([-1.0, 0.0, 1.0], size=len(df), p=probs)
    df['pos_raw'] = choices.astype(float)
    return df
