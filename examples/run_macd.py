from _bootstrap import *

from data import load_yf
from macd import macd_strategy
from backtest import backtest_positions
from performance import performance_report
from drawdown import drawdown_report
from trades import build_trades, trade_stats
from viz_basic import plot_pack

def main():
    # 1) Load data
    df = load_yf(
        ticker='NVDA',
        period='1y',
        interval='1d',
        auto_adjust=False,
        multi_level_index=False,
        progress=False
    )

    # 2) Strategy -> pos_raw
    df = macd_strategy(df)

    # 3) Backtest -> equity/returns
    bt = backtest_positions(df)

    # 4) Reports
    perf = performance_report(bt)
    bt2, dd_rep, spells = drawdown_report(bt)
    tr = build_trades(bt2)
    ts = trade_stats(tr)

    # 5) Print summary
    print('\n=== PERFORMANCE ===')
    print(perf)

    print('\n=== DRAWDOWN REPORT ===')
    print(dd_rep)

    print('\n=== TRADE STATS ===')
    print(ts)

    # 6) Quick plots
    plot_pack(bt2)

if __name__ == '__main__':
    main()