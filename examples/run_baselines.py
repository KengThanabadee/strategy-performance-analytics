from _bootstrap import *

from data import load_yf
from backtest import backtest_positions
from performance import performance_report
from drawdown import drawdown_report
from trades import build_trades, trade_stats
from baselines import buy_and_hold, random_positions

def run_one(name, df):
    bt, _ = backtest_positions(df)
    perf = performance_report(bt)
    bt2, dd_rep, _ = drawdown_report(bt)
    tr = build_trades(bt2)
    ts = trade_stats(tr)

    print(f'\n=== {name} ===')
    print('PERFORMANCE:', perf)
    print('DRAWDOWN:', dd_rep)
    print('TRADE STATS:', ts)

def main():
    df = load_yf(
        ticker='NVDA',
        period='1y',
        interval='1d',
        auto_adjust=False,
        multi_level_index=False,
        progress=False
    )

    run_one('BUY_AND_HOLD', buy_and_hold(df))
    run_one('RANDOM_POSITIONS', random_positions(df, seed=42, p_long=0.5, p_short=0.0))

if __name__ == '__main__':
    main()
