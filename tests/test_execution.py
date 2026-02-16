import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, backtest_positions
from src.execution import ExecutionModel, apply_execution_costs


def test_slippage_reduces_final_equity_vs_no_slippage() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            'pos_raw': [0.0, 1.0, 1.0, -1.0, -1.0, 0.0],
        }
    )

    cfg_no_slippage = BacktestConfig(commission_bps=5.0, slippage_bps=0.0, spread_bps=0.0)
    cfg_with_slippage = BacktestConfig(commission_bps=5.0, slippage_bps=8.0, spread_bps=0.0)

    _, m0 = backtest_positions(df, config=cfg_no_slippage, long_or_short='both')
    _, m1 = backtest_positions(df, config=cfg_with_slippage, long_or_short='both')

    assert m1['final_equity'] < m0['final_equity']


def test_spread_cost_applied_on_turnover_rows() -> None:
    pos_exec = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0], dtype=float)
    turnover = pd.Series([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)
    active = pd.Series([True, True, True, True, False], dtype=bool)
    model = ExecutionModel(spread_bps=20.0)

    costs = apply_execution_costs(pos_exec, turnover, active, model)

    expected_spread = pd.Series([0.0, 0.001, 0.0, 0.002, 0.0], dtype=float)
    assert np.allclose(costs['cost_spread'], expected_spread, rtol=1e-12, atol=1e-12)


def test_ret_cost_equals_component_sum_and_terminal_is_zero() -> None:
    pos_exec = pd.Series([0.0, 1.0, -1.0, -1.0], dtype=float)
    turnover = pd.Series([0.0, 1.0, 2.0, 0.0], dtype=float)
    active = pd.Series([True, True, True, False], dtype=bool)
    model = ExecutionModel(
        commission_bps=10.0,
        spread_bps=20.0,
        slippage_bps=5.0,
        borrow_cost_annual=0.365,
        periods_per_year=365,
    )

    costs = apply_execution_costs(pos_exec, turnover, active, model)

    comp_sum = (
        costs['cost_commission']
        + costs['cost_spread']
        + costs['cost_slippage']
        + costs['cost_borrow']
    )

    assert np.allclose(costs['ret_cost'], comp_sum, rtol=1e-12, atol=1e-12)
    assert costs['ret_cost'].iloc[-1] == 0.0
