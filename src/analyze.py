from backtest import backtest_positions
from drawdown import drawdown_report
from performance import performance_report
from trades import build_trades, trade_stats

def analyze_backtest(bt_df):
    trades = build_trades(bt_df)
    tstats = trade_stats(trades)
    perf = performance_report(bt_df, trade_stats_report=tstats)
    dd_df, dd_rep, spells = drawdown_report(bt_df)

    return {
        'performance': perf,
        'drawdown': dd_rep,
        'trade_stats': tstats,
        'trades': trades,
        'spells': spells
    }

def analyze_with_baselines(
    data,
    strategy_fn,
    baseline_fns=None,
    backtest_kwargs=None
):
    baseline_fns = baseline_fns or {}
    backtest_kwargs = backtest_kwargs or {}

    strat_df = strategy_fn(data.copy())
    bt, metrics = backtest_positions(strat_df, **backtest_kwargs)
    report = analyze_backtest(bt)
    report['backtest_metrics'] = metrics

    baselines = {}
    for name, fn in baseline_fns.items():
        base_df = fn(data.copy())
        bt_b, metrics_b = backtest_positions(base_df, **backtest_kwargs)
        rep_b = analyze_backtest(bt_b)
        rep_b['backtest_metrics'] = metrics_b
        baselines[name] = rep_b

    return {
        'strategy': report,
        'baselines': baselines
    }
