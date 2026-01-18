def analyze_backtest(bt_df):
    perf = performance_report(bt_df)
    dd_df, dd_rep, spells = drawdown_report(bt_df)
    trades = build_trades(bt_df)
    tstats  = trade_stats(trades)

    return {
        'performance': perf,
        'drawdown': dd_rep,
        'trade_stats': tstats,
        'trades': trades,
        'spells': spells
    }