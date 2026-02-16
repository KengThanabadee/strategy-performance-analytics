from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

try:
    from .execution import ExecutionModel, apply_execution_costs
except ImportError:  # pragma: no cover - supports script-style imports from examples/
    from execution import ExecutionModel, apply_execution_costs

_VALID_SIDE = {'long', 'short', 'both'}
_VALID_ORDER_POLICY = {'order_proxy_flip2', 'rebalance_events'}
_VALID_PRICE_MODE = {'open_to_open', 'close_to_close'}


@dataclass(frozen=True)
class BacktestConfig:
    price_mode: Literal['open_to_open', 'close_to_close'] = 'open_to_open'
    signal_lag_bars: int = 1
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    borrow_cost_annual: float = 0.0
    periods_per_year: int = 252


def _validate_config(config: BacktestConfig) -> None:
    if config.price_mode not in _VALID_PRICE_MODE:
        raise ValueError('price_mode must be \'open_to_open\' or \'close_to_close\'')
    if config.signal_lag_bars < 0:
        raise ValueError('signal_lag_bars must be >= 0')
    if config.periods_per_year <= 0:
        raise ValueError('periods_per_year must be > 0')


def _validate_inputs(data: pd.DataFrame, long_or_short: str, order_policy: str, config: BacktestConfig) -> None:
    required_cols = {'pos_raw'}
    if config.price_mode == 'open_to_open':
        required_cols.add('Open')
    else:
        required_cols.add('Close')

    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f'Missing required columns: {missing}')

    if long_or_short not in _VALID_SIDE:
        raise ValueError('long_or_short must be \'long\', \'short\' or \'both\'')
    if order_policy not in _VALID_ORDER_POLICY:
        raise ValueError('order_policy must be \'order_proxy_flip2\' or \'rebalance_events\'')


def _build_target_position(df: pd.DataFrame, long_or_short: str, round_pos_decimals: int | None) -> pd.Series:
    pos_target = df['pos_raw'].astype(float)
    if long_or_short == 'long':
        pos_target = pos_target.clip(lower=0.0)
    elif long_or_short == 'short':
        pos_target = pos_target.clip(upper=0.0)

    if round_pos_decimals is not None:
        pos_target = pos_target.round(round_pos_decimals)
    return pos_target


def _compute_order_count(pos_exec: pd.Series, order_policy: str) -> pd.Series:
    pos_prev = pos_exec.shift(1).fillna(0.0)
    changed = pos_exec.ne(pos_prev)

    if order_policy == 'order_proxy_flip2':
        flipped = (pos_prev * pos_exec < 0.0)
        return changed.astype(int) + flipped.astype(int)
    return changed.astype(int)


def _price_column(price_mode: str) -> str:
    return 'Open' if price_mode == 'open_to_open' else 'Close'


def backtest_positions(
    data: pd.DataFrame,
    config: BacktestConfig | None = None,
    *,
    long_or_short: str = 'long',
    round_pos_decimals: int | None = None,
    order_policy: str = 'order_proxy_flip2',
    cost_bps: float = 0.0,
):
    '''Run single-asset deterministic backtest.

    Execution contract:
    - A signal is observed at bar t from `pos_raw`.
    - The target position for that signal is `pos_target[t]`.
    - Executed position is shifted by `signal_lag_bars`, so fills occur at t + lag.
    - Return window follows `price_mode`:
      - `open_to_open`: Open[t+1] / Open[t] - 1
      - `close_to_close`: Close[t+1] / Close[t] - 1

    Output contract:
    - Canonical columns: `ret_gross`, `ret_cost`, `ret_net`, `equity_net`
    - Compatibility aliases: `gross_strat_ret`, `net_strat_ret`, `equity`, `cost`, `pos`
    '''

    cfg = config or BacktestConfig(commission_bps=float(cost_bps))
    _validate_config(cfg)
    _validate_inputs(data, long_or_short, order_policy, cfg)

    df = data.copy()
    n_rows = len(df)
    price_col = _price_column(cfg.price_mode)

    df['pos_target'] = _build_target_position(df, long_or_short, round_pos_decimals)
    df['pos_exec'] = df['pos_target'].shift(cfg.signal_lag_bars).fillna(0.0)
    df['pos'] = df['pos_exec']  # backward-compatible alias

    pos_prev = df['pos_exec'].shift(1).fillna(0.0)
    df['turnover'] = (df['pos_exec'] - pos_prev).abs()
    df['trade_notional'] = df['turnover']
    df['order_count'] = _compute_order_count(df['pos_exec'], order_policy)

    fwd_ret = df[price_col].shift(-1) / df[price_col] - 1.0
    active_return_bar = fwd_ret.notna()
    df['ret_price_fwd'] = fwd_ret
    df['ret_oo_fwd'] = fwd_ret if cfg.price_mode == 'open_to_open' else 0.0

    df['ret_gross'] = (df['pos_exec'] * fwd_ret).where(active_return_bar, 0.0)
    execution_model = ExecutionModel(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        spread_bps=cfg.spread_bps,
        borrow_cost_annual=cfg.borrow_cost_annual,
        periods_per_year=cfg.periods_per_year,
    )
    cost_df = apply_execution_costs(
        pos_exec=df['pos_exec'],
        turnover=df['turnover'],
        active_return_bar=active_return_bar,
        model=execution_model,
    )
    for col in ['cost_commission', 'cost_spread', 'cost_slippage', 'cost_borrow', 'ret_cost']:
        df[col] = cost_df[col]

    df['ret_net'] = df['ret_gross'] - df['ret_cost']

    df['equity_gross'] = (1.0 + df['ret_gross']).cumprod()
    df['equity_net'] = (1.0 + df['ret_net']).cumprod()

    # TODO: remove legacy aliases in a major API cleanup phase.
    df['gross_strat_ret'] = df['ret_gross']
    df['net_strat_ret'] = df['ret_net']
    df['cost'] = df['ret_cost']
    df['equity'] = df['equity_net']

    notional_traded = float(df['trade_notional'].sum())
    trade_days = int((df['turnover'] > 0).sum())
    order_events = int(df['order_count'].sum())
    gross_final = float(df['equity_gross'].iloc[-1]) if n_rows > 0 else 1.0
    net_final = float(df['equity_net'].iloc[-1]) if n_rows > 0 else 1.0
    total_cost = float(df['ret_cost'].sum())
    avg_cost_bps_realized = float(total_cost / notional_traded * 10000.0) if notional_traded > 0 else 0.0

    metrics = {
        'final_equity': net_final,
        'final_equity_net': net_final,
        'final_equity_gross': gross_final,
        'notional_traded': notional_traded,
        'trade_days': trade_days,
        'avg_daily_turnover': float(df['turnover'].mean()) if n_rows > 0 else 0.0,
        'active_day_frac': float((df['turnover'] > 0).mean()) if n_rows > 0 else 0.0,
        'notional_per_trade_day': float(notional_traded / trade_days) if trade_days > 0 else 0.0,
        'order_events': order_events,
        'avg_cost_bps_realized': avg_cost_bps_realized,
        'gross_vs_net_drag': float(gross_final - net_final),
        'price_mode': cfg.price_mode,
        'signal_lag_bars': cfg.signal_lag_bars,
    }
    return df, metrics
