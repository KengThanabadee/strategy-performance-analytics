from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionModel:
    '''Deterministic execution-cost model for single-asset backtests.

    Costs are modeled in return space and applied per bar:
    - commission: turnover * commission_bps / 10000
    - spread: turnover * (spread_bps / 2) / 10000
    - slippage: turnover * slippage_bps / 10000
    - borrow: abs(min(pos_exec, 0)) * borrow_cost_annual / periods_per_year
    '''

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    borrow_cost_annual: float = 0.0
    periods_per_year: int = 252


def _validate_execution_model(model: ExecutionModel) -> None:
    if model.commission_bps < 0:
        raise ValueError('commission_bps must be >= 0')
    if model.slippage_bps < 0:
        raise ValueError('slippage_bps must be >= 0')
    if model.spread_bps < 0:
        raise ValueError('spread_bps must be >= 0')
    if model.borrow_cost_annual < 0:
        raise ValueError('borrow_cost_annual must be >= 0')
    if model.periods_per_year <= 0:
        raise ValueError('periods_per_year must be > 0')


def apply_execution_costs(
    pos_exec: pd.Series,
    turnover: pd.Series,
    active_return_bar: pd.Series,
    model: ExecutionModel,
) -> pd.DataFrame:
    '''Compute execution cost components for each bar.

    Output columns:
    - `cost_commission`
    - `cost_spread`
    - `cost_slippage`
    - `cost_borrow`
    - `ret_cost` (sum of all components)
    '''

    _validate_execution_model(model)

    if not (pos_exec.index.equals(turnover.index) and pos_exec.index.equals(active_return_bar.index)):
        raise ValueError('pos_exec, turnover, and active_return_bar must share the same index')

    active = active_return_bar.astype(bool)

    commission = turnover * (model.commission_bps / 10000.0)
    spread = turnover * ((model.spread_bps / 2.0) / 10000.0)
    slippage = turnover * (model.slippage_bps / 10000.0)
    borrow = pos_exec.clip(upper=0.0).abs() * (model.borrow_cost_annual / model.periods_per_year)

    out = pd.DataFrame(index=pos_exec.index)
    out['cost_commission'] = commission.where(active, 0.0)
    out['cost_spread'] = spread.where(active, 0.0)
    out['cost_slippage'] = slippage.where(active, 0.0)
    out['cost_borrow'] = borrow.where(active, 0.0)
    out['ret_cost'] = out['cost_commission'] + out['cost_spread'] + out['cost_slippage'] + out['cost_borrow']
    return out
