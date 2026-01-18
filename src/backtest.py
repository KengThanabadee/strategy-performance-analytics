import pandas as pd
import numpy as np

def backtest_positions(
    data, 
    long_or_short='long', 
    round_pos_decimals=None, 
    order_policy='order_proxy_flip2', 
    cost_bps=0
):
    df = data.copy()

    # execute next bar
    if long_or_short == 'both':
        df['pos'] = df['pos_raw'].shift(1).fillna(0)
    elif long_or_short == 'long':
        df['pos'] = df['pos_raw'].shift(1).fillna(0).clip(lower=0)
    elif long_or_short == 'short':
        df['pos'] = df['pos_raw'].shift(1).fillna(0).clip(upper=0)
    else:
        raise ValueError('long_or_short must be "long", "short" or "both"')

    # quantize position
    if round_pos_decimals is not None:
        df['pos'] = df['pos'].round(round_pos_decimals)

    # order count
    pos_prev = df['pos'].shift(1).fillna(0)
    changed = (df['pos'] != pos_prev)

    if order_policy == 'order_proxy_flip2':
        flipped = (pos_prev * df['pos'] < 0)
        df['order_count'] = changed.astype(int) + flipped.astype(int)
    elif order_policy == 'rebalance_events':
        df['order_count'] = changed.astype(int)
    else:
        raise ValueError('order_policy must be "order_proxy_flip2" or "rebalance_events"')
    
    # return
    df['ret_oo_fwd'] = df['Open'].shift(-1) / df['Open'] - 1
    df['gross_strat_ret'] = df['pos'] * df['ret_oo_fwd']

    # notional turnover (size traded)
    df['turnover'] = (df['pos'] - pos_prev).abs()

    # cost per notional
    df['cost'] = df['turnover'] * (cost_bps / 10000)
    df['net_strat_ret'] = df['gross_strat_ret'] - df['cost']
    
    # equity curve
    df['equity'] = (1.0 + df['net_strat_ret'].fillna(0.0)).cumprod()

    # metrics
    notional_traded = float(df['turnover'].sum())
    trade_days = int(changed.astype(int).sum())
    order_events = int(df['order_count'].sum())
    
    metrics = {
        'final_equity': float(df['equity'].iloc[-1]),
        'notional_traded': notional_traded,
        'trade_days': trade_days,
        'avg_daily_turnover': float(df['turnover'].mean()),
        'active_day_frac': float((df['turnover'] > 0).mean()),
        'notional_per_trade_day': float(notional_traded / trade_days) if trade_days > 0 else 0.0,
        'order_events': order_events
    }
    return df, metrics