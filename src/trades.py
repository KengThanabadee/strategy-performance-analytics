import pandas as pd
import numpy as np
import warnings

def build_trades(
    bt: pd.DataFrame,
    strict: bool = True,
    pos_col: str = 'pos',
    px_col: str = 'Open',
    ret_col: str = 'net_strat_ret',
    gross_ret_col: str = 'gross_strat_ret',
    close_at_end: bool = True
) -> pd.DataFrame:
    """
    Build trade ledger from position series.

    Definitions / Rules:
    - Entry when pos changes from 0 -> nonzero
    - Exit when pos changes from nonzero -> 0
    - Flip (eg. -1 -> +1) is treated as:
        exit old trade + entry new trade at the SAME timestamp/price (2 trades)

    Return:
        trades_df with one row per trade:
            entry_time, exit_time, side, pos_size, entry_px, exit_px,
            bars_held, time_held (if DateTimeIndex),
            trade_ret_net (DEFAULT, from net_strat_ret),
            trade_ret_gross (diagnostic, from gross_strat_ret)
    """
    if pos_col not in bt.columns:
        raise KeyError(f'Missing column {pos_col}. Run backtest first.')
    if px_col not in bt.columns:
        raise KeyError(f'Missing column {px_col}.')
    if ret_col not in bt.columns:
        raise KeyError(f'Missing column {ret_col}. Run backtest first.')
    if gross_ret_col not in bt.columns:
        raise KeyError(f'Missing column {gross_ret_col}. Run backtest first.')

    df = bt.copy()
    if len(df) == 0:
        return pd.DataFrame([])

    # Ensure monotonic index (required for sane trade windows)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    pos = df[pos_col].fillna(0).astype(float)
    px = df[px_col].astype(float)
    net_r = df[ret_col].astype(float).fillna(0)
    gross_r = df[gross_ret_col].astype(float).fillna(0)

    prev = pos.shift(1).fillna(0)
    changed = pos.ne(prev)

    trades = []
    open_trade = None # dict for current open trade

    # Loop only on change points for efficiency
    for t in df.index[changed]:
        p_prev = float(prev.loc[t])
        p_now = float(pos.loc[t])
        price = float(px.loc[t])

        # Close trade: whenever we had a position and it changes away (to 0 or flip)
        if p_prev != 0 and p_now != p_prev:
            if open_trade is None:
                msg = f'Inconsistent state: trying to close but open_trade is None at {t} (p_prev={p_prev}, p_now={p_now})'
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)

                # Defensive fallback: if state got inconsistent, create a minimal open trade
                open_trade = {
                    'entry_time': t,
                    'side': 'long' if p_prev > 0 else 'short',
                    'pos_size': p_prev,
                    'entry_px': price,
                    'fallback_created': True
                }

            open_trade['exit_time'] = t
            open_trade['exit_px'] = price
            trades.append(open_trade)
            open_trade = None

        # Open trade: whenever current position is nonzero and and differs from previous (from 0 or flip)
        if p_now != 0 and p_now != p_prev:
            open_trade = {
                'entry_time': t,
                'side': 'long' if p_now > 0 else 'short',
                'pos_size': p_now,
                'entry_px': price
            }

    # Close at end-of-sample if requested
    if open_trade is not None and close_at_end:
        t_end = df.index[-1]
        open_trade['exit_time'] = t_end
        open_trade['exit_px'] = float(px.iloc[-1])
        open_trade['closed_by_end'] = True
        trades.append(open_trade)
        open_trade = None

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return trades_df

    # Normalize dtypes
    trades_df['entry_px'] = trades_df['entry_px'].astype(float)
    trades_df['exit_px'] = trades_df['exit_px'].astype(float)
    if 'closed_by_end' not in trades_df.columns:
        trades_df['closed_by_end'] = False

    # Map timestamps to integer positions (robust for intraday as long as index is ordered)
    index_pos = pd.Series(np.arange(len(df)), index=df.index)
    entry_i = index_pos.loc[trades_df['entry_time']].to_numpy()
    exit_i = index_pos.loc[trades_df['exit_time']].to_numpy()

    # bars_held counts the number of return bars included in net window
    trades_df['bars_held'] = (exit_i - entry_i).astype(int)

    # Time held 
    if isinstance(df.index, pd.DatetimeIndex):
        trades_df['time_held'] = trades_df['exit_time'] - trades_df['entry_time']
        
    # Net trade return from net_strat_ret over the holding window
    # Use bars [entry_i, exit_i): include entry bar, exclude exit bar
    trade_ret_net = []
    for a, b in zip(entry_i, exit_i):
        seg = net_r.iloc[a:b]
        tr = float((1.0 + seg).prod() - 1.0)
        trade_ret_net.append(tr)
    trades_df['trade_ret_net'] = trade_ret_net

    # Diagnostic return from gross_strat_ret over the holding window
    trade_ret_gross = []
    for a, b in zip(entry_i, exit_i):
        seg = gross_r.iloc[a:b]
        tr = float((1.0 + seg).prod() - 1.0)
        trade_ret_gross.append(tr)
    trades_df['trade_ret_gross'] = trade_ret_gross

    return trades_df

def trade_stats(trades_df: pd.DataFrame, ret_col: str = 'trade_ret_net') -> dict:
    if trades_df is None or len(trades_df) == 0:
        return {
            'n_trades': 0,
            'win_rate': np.nan,
            'avg_win': np.nan,
            'avg_loss_abs': np.nan,
            'payoff': np.nan,
            'expectancy': np.nan,
            'kelly': np.nan,
            'half_kelly': np.nan,
            'quarter_kelly': np.nan
        }

    if ret_col not in trades_df.columns:
        raise KeyError(f'Missing column {ret_col} in trades_df')

    r = trades_df[ret_col].astype(float)

    wins = r[r > 0]
    losses = r[r < 0]

    p = float((r > 0).mean())
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss_abs = float((-losses).mean()) if len(losses) > 0 else 0

    payoff = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else np.nan
    expectancy = (p * avg_win) - ((1 - p) * avg_loss_abs)

    # Kelly (binary approximation): f* = (bp - q) / b, with b=payoff, q=1-p
    if np.isfinite(payoff) and payoff > 0:
        kelly = (payoff * p - (1 - p)) / payoff
    else:
        kelly = np.nan

    return {
        'n_trades': int(len(r)),
        'win_rate': p,
        'avg_win': avg_win,
        'avg_loss_abs': avg_loss_abs,
        'payoff': payoff,
        'expectancy': float(expectancy),
        'kelly': float(kelly) if np.isfinite(kelly) else np.nan,
        'half_kelly': float(0.5 * kelly) if np.isfinite(kelly) else np.nan,
        'quarter_kelly': float(0.25 * kelly) if np.isfinite(kelly) else np.nan
    }