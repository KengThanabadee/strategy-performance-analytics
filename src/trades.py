from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _max_consecutive(mask: pd.Series) -> int:
    max_run = 0
    run = 0
    for val in mask.fillna(False).astype(bool).to_numpy():
        if val:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)


def build_trades(
    bt: pd.DataFrame,
    strict: bool = True,
    pos_col: str = 'pos',
    px_col: str = 'Open',
    ret_col: str = 'net_strat_ret',
    gross_ret_col: str = 'gross_strat_ret',
    close_at_end: bool = True,
) -> pd.DataFrame:
    '''Build trade ledger from position series.

    Definitions:
    - Entry when position changes from 0 to non-zero.
    - Exit when position changes from non-zero to 0.
    - Flip (-1 to +1 or +1 to -1) is exit + entry at same timestamp.
    '''

    for col in [pos_col, px_col, ret_col, gross_ret_col]:
        if col not in bt.columns:
            raise KeyError(f'Missing column {col}. Run backtest first.')

    df = bt.copy()
    if len(df) == 0:
        return pd.DataFrame([])

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    pos = df[pos_col].fillna(0).astype(float)
    px = df[px_col].astype(float)
    net_r = df[ret_col].astype(float).fillna(0)
    gross_r = df[gross_ret_col].astype(float).fillna(0)

    prev = pos.shift(1).fillna(0)
    changed = pos.ne(prev)

    trades = []
    open_trade = None

    for t in df.index[changed]:
        p_prev = float(prev.loc[t])
        p_now = float(pos.loc[t])
        price = float(px.loc[t])

        if p_prev != 0 and p_now != p_prev:
            if open_trade is None:
                msg = f'Inconsistent state: close requested with no open trade at {t}'
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)
                open_trade = {
                    'entry_time': t,
                    'side': 'long' if p_prev > 0 else 'short',
                    'pos_size': p_prev,
                    'entry_px': price,
                    'fallback_created': True,
                }

            open_trade['exit_time'] = t
            open_trade['exit_px'] = price
            trades.append(open_trade)
            open_trade = None

        if p_now != 0 and p_now != p_prev:
            open_trade = {
                'entry_time': t,
                'side': 'long' if p_now > 0 else 'short',
                'pos_size': p_now,
                'entry_px': price,
            }

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

    trades_df['entry_px'] = trades_df['entry_px'].astype(float)
    trades_df['exit_px'] = trades_df['exit_px'].astype(float)
    if 'closed_by_end' not in trades_df.columns:
        trades_df['closed_by_end'] = False

    index_pos = pd.Series(np.arange(len(df)), index=df.index)
    entry_i = index_pos.loc[trades_df['entry_time']].to_numpy()
    exit_i = index_pos.loc[trades_df['exit_time']].to_numpy()
    trades_df['bars_held'] = (exit_i - entry_i).astype(int)

    if isinstance(df.index, pd.DatetimeIndex):
        trades_df['time_held'] = trades_df['exit_time'] - trades_df['entry_time']

    trade_ret_net = []
    trade_ret_gross = []
    for a, b in zip(entry_i, exit_i):
        net_seg = net_r.iloc[a:b]
        gross_seg = gross_r.iloc[a:b]
        trade_ret_net.append(float((1.0 + net_seg).prod() - 1.0))
        trade_ret_gross.append(float((1.0 + gross_seg).prod() - 1.0))
    trades_df['trade_ret_net'] = trade_ret_net
    trades_df['trade_ret_gross'] = trade_ret_gross

    return trades_df


def trade_stats(trades_df: pd.DataFrame, ret_col: str = 'trade_ret_net') -> dict:
    '''Compute trade-level statistics.

    This is the single source of truth for trade-level metrics such as win_rate.
    '''

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
            'quarter_kelly': np.nan,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_holding_bars': np.nan,
            'avg_holding_days': np.nan,
        }

    if ret_col not in trades_df.columns:
        raise KeyError(f'Missing column {ret_col} in trades_df')

    r = trades_df[ret_col].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]

    p = float((r > 0).mean())
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss_abs = float((-losses).mean()) if len(losses) > 0 else 0.0

    payoff = (avg_win / avg_loss_abs) if avg_loss_abs > 0 else np.nan
    expectancy = (p * avg_win) - ((1 - p) * avg_loss_abs)

    if np.isfinite(payoff) and payoff > 0:
        kelly = (payoff * p - (1 - p)) / payoff
    else:
        kelly = np.nan

    max_wins = _max_consecutive(r > 0)
    max_losses = _max_consecutive(r < 0)
    avg_holding_bars = float(trades_df['bars_held'].mean()) if 'bars_held' in trades_df.columns else np.nan

    if 'time_held' in trades_df.columns:
        avg_holding_days = float((trades_df['time_held'] / pd.Timedelta(days=1)).mean())
    else:
        avg_holding_days = np.nan

    return {
        'n_trades': int(len(r)),
        'win_rate': p,
        'avg_win': avg_win,
        'avg_loss_abs': avg_loss_abs,
        'payoff': payoff,
        'expectancy': float(expectancy),
        'kelly': float(kelly) if np.isfinite(kelly) else np.nan,
        'half_kelly': float(0.5 * kelly) if np.isfinite(kelly) else np.nan,
        'quarter_kelly': float(0.25 * kelly) if np.isfinite(kelly) else np.nan,
        'max_consecutive_wins': max_wins,
        'max_consecutive_losses': max_losses,
        'avg_holding_bars': avg_holding_bars,
        'avg_holding_days': avg_holding_days,
    }
