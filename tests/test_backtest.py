import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, backtest_positions


def test_execution_timing_signal_lag_one_bar() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0],
            'Close': [100.5, 101.5, 102.5, 103.5],
            'pos_raw': [1.0, 0.0, -1.0, 1.0],
        }
    )
    cfg = BacktestConfig(signal_lag_bars=1)

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='both')

    expected_pos_exec = pd.Series([0.0, 1.0, 0.0, -1.0])
    assert np.allclose(bt_df['pos_exec'], expected_pos_exec, rtol=1e-12, atol=1e-12)


def test_open_to_open_return_convention_matches_known_values() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 110.0, 121.0, 133.1],
            'Close': [99.0, 109.0, 120.0, 132.0],
            'pos_raw': [1.0, 1.0, 1.0, 1.0],
        }
    )
    cfg = BacktestConfig(price_mode='open_to_open', signal_lag_bars=1)

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='long')

    expected_fwd = pd.Series([0.1, 0.1, 0.1, np.nan])
    assert np.allclose(bt_df['ret_price_fwd'].iloc[:-1], expected_fwd.iloc[:-1], rtol=1e-12, atol=1e-12)
    assert np.isclose(bt_df['ret_gross'].iloc[1], 0.1, rtol=1e-12, atol=1e-12)


def test_cost_decomposition_components_are_applied() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'pos_raw': [0.0, 1.0, 1.0, -1.0, -1.0],
        }
    )
    cfg = BacktestConfig(
        commission_bps=10.0,
        spread_bps=20.0,
        slippage_bps=5.0,
        borrow_cost_annual=0.0,
        signal_lag_bars=1,
    )

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='both')

    expected_turnover = pd.Series([0.0, 0.0, 1.0, 0.0, 2.0])
    expected_commission = expected_turnover * (10.0 / 10000.0)
    expected_spread = expected_turnover * ((20.0 / 2.0) / 10000.0)
    expected_slippage = expected_turnover * (5.0 / 10000.0)
    expected_total = expected_commission + expected_spread + expected_slippage
    expected_total.iloc[-1] = 0.0

    expected_commission.iloc[-1] = 0.0
    expected_spread.iloc[-1] = 0.0
    expected_slippage.iloc[-1] = 0.0

    assert np.allclose(bt_df['turnover'], expected_turnover, rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['cost_commission'], expected_commission, rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['cost_spread'], expected_spread, rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['cost_slippage'], expected_slippage, rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['ret_cost'], expected_total, rtol=1e-12, atol=1e-12)


def test_flip_from_long_to_short_has_turnover_two() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'pos_raw': [0.0, 1.0, 1.0, -1.0, -1.0],
        }
    )
    cfg = BacktestConfig(signal_lag_bars=1)

    bt_df, metrics = backtest_positions(df, config=cfg, long_or_short='both', order_policy='order_proxy_flip2')

    assert np.isclose(bt_df['turnover'].iloc[4], 2.0, rtol=1e-12, atol=1e-12)
    assert bt_df['order_count'].iloc[4] == 2
    assert metrics['order_events'] == int(bt_df['order_count'].sum())


def test_borrow_cost_applies_only_on_short_exposure() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0],
            'pos_raw': [-1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    cfg = BacktestConfig(borrow_cost_annual=0.365, periods_per_year=365, signal_lag_bars=1)

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='both')

    expected_borrow = pd.Series([0.0, 0.001, 0.001, 0.001, 0.0])
    assert np.allclose(bt_df['cost_borrow'], expected_borrow, rtol=1e-12, atol=1e-12)


def test_invariants_ret_and_equity_hold() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 102.0, 101.0, 103.0, 104.0],
            'Close': [100.5, 102.5, 101.5, 103.5, 104.5],
            'pos_raw': [0.0, 1.0, -1.0, 1.0, 0.0],
        }
    )
    cfg = BacktestConfig(commission_bps=10.0, spread_bps=5.0, slippage_bps=2.0, signal_lag_bars=1)

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='both')

    assert np.allclose(bt_df['ret_net'], bt_df['ret_gross'] - bt_df['ret_cost'], rtol=1e-12, atol=1e-12)
    expected_equity = (1.0 + bt_df['ret_net']).cumprod()
    assert np.allclose(bt_df['equity_net'], expected_equity, rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['net_strat_ret'], bt_df['ret_net'], rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['gross_strat_ret'], bt_df['ret_gross'], rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['cost'], bt_df['ret_cost'], rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_df['equity'], bt_df['equity_net'], rtol=1e-12, atol=1e-12)


def test_backward_compat_cost_bps_maps_to_commission_only() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0],
            'Close': [100.0, 101.0, 102.0, 103.0],
            'pos_raw': [0.0, 1.0, 1.0, 0.0],
        }
    )
    cfg = BacktestConfig(commission_bps=10.0, spread_bps=0.0, slippage_bps=0.0, borrow_cost_annual=0.0)

    bt_new, _ = backtest_positions(df, config=cfg, long_or_short='both')
    bt_old_style, _ = backtest_positions(df, long_or_short='both', cost_bps=10.0)

    assert np.allclose(bt_new['net_strat_ret'], bt_old_style['net_strat_ret'], rtol=1e-12, atol=1e-12)
    assert np.allclose(bt_new['equity'], bt_old_style['equity'], rtol=1e-12, atol=1e-12)


def test_close_to_close_mode_uses_close_prices() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 130.0, 90.0, 140.0],
            'Close': [100.0, 110.0, 121.0, 133.1],
            'pos_raw': [1.0, 1.0, 1.0, 1.0],
        }
    )
    cfg = BacktestConfig(price_mode='close_to_close', signal_lag_bars=1)

    bt_df, _ = backtest_positions(df, config=cfg, long_or_short='long')

    expected_fwd_close = pd.Series([0.1, 0.1, 0.1, np.nan])
    assert np.allclose(bt_df['ret_price_fwd'].iloc[:-1], expected_fwd_close.iloc[:-1], rtol=1e-12, atol=1e-12)
    assert np.isclose(bt_df['ret_gross'].iloc[1], 0.1, rtol=1e-12, atol=1e-12)
