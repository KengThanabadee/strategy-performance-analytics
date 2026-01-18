import pandas as pd
import numpy as np

def performance_report(
    data,
    ret_col='net_strat_ret',
    equity_col='equity',
    periods_per_year=252,
    rf_annual=0.0,
    dd_report=None
):
    df = data.copy()

    # returns series (clean)
    r = df[ret_col].astype(float).fillna(0.0)

    # equity (if not provided)
    if equity_col not in df.columns:
        df[equity_col] = (1.0 + r).cumprod()

    eq = df[equity_col].astype(float)
    n = int(r.shape[0])

    # annualized returns / CAGR
    if n > 1:
        years = n / periods_per_year
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
    else:
        cagr = np.nan

    # sharpe ratio
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    vol = float(excess.std(ddof=1))
    sharpe = float((excess.mean() / vol) * np.sqrt(periods_per_year)) if vol > 0 else np.nan

    # max drawdon (prefer from dd_report, else compute quick)
    if dd_report is not None and 'max_drawdown' in dd_report:
        max_dd = float(dd_report['max_drawdown'])
    else:
        peak = eq.cummax()
        dd = eq / peak - 1
        max_dd = float(dd.min())

    # calmer
    calmar = float(cagr / abs(max_dd)) if (np.isfinite(cagr) and max_dd < 0) else np.nan

    # extra handy stats
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    total_return = float(eq.iloc[-1] - 1.0)

    report = {
        'total_return': total_return,
        'CAGR': cagr,
        'ann_vol': ann_vol,
        'Sharpe': sharpe,
        'max_drawdown': max_dd,
        'Calmar': calmar,
        'n_periods': n
    }
    return report