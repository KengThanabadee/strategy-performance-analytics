import numpy as np
import pandas as pd

from src.backtest import backtest_positions
from src.performance import performance_report
from src.trades import trade_stats


def test_trade_stats_streaks_and_avg_holding_bars() -> None:
    trades_df = pd.DataFrame(
        {
            'trade_ret_net': [0.10, 0.05, -0.02, -0.01, -0.03, 0.04, 0.02],
            'bars_held': [2, 3, 1, 4, 2, 1, 5],
        }
    )

    stats = trade_stats(trades_df)

    assert stats['max_consecutive_wins'] == 2
    assert stats['max_consecutive_losses'] == 3
    assert np.isclose(stats['avg_holding_bars'], np.mean([2, 3, 1, 4, 2, 1, 5]), rtol=1e-12, atol=1e-12)


def test_trade_stats_avg_holding_days_with_timedelta_column() -> None:
    trades_df = pd.DataFrame(
        {
            'trade_ret_net': [0.01, -0.02, 0.03],
            'bars_held': [1, 2, 3],
            'time_held': [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=4)],
        }
    )

    stats = trade_stats(trades_df)

    assert np.isclose(stats['avg_holding_days'], (1 + 2 + 4) / 3, rtol=1e-12, atol=1e-12)


def test_performance_winrate_matches_trade_stats_single_source() -> None:
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
            'Close': [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
            'pos_raw': [0.0, 1.0, 1.0, -1.0, -1.0, 0.0],
        }
    )

    bt_df, _ = backtest_positions(df, long_or_short='both')

    # Build a compact trade-level view from backtest outputs for win-rate handoff.
    # This mirrors the integration contract: performance gets trade stats, not trade logic.
    trades_df = pd.DataFrame({'trade_ret_net': [0.02, -0.01, 0.01], 'bars_held': [1, 2, 1]})
    tstats = trade_stats(trades_df)
    report = performance_report(bt_df, trade_stats_report=tstats)

    assert np.isclose(report['WinRate'], tstats['win_rate'], rtol=1e-12, atol=1e-12)
