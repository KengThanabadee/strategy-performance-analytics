from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_profit_factor(returns: pd.Series) -> float:
    gross_profit = float(returns[returns > 0].sum())
    gross_loss_abs = float((-returns[returns < 0]).sum())
    if gross_loss_abs <= 0:
        return np.nan
    return gross_profit / gross_loss_abs


def _safe_sortino(excess_returns: pd.Series, periods_per_year: int) -> float:
    downside = excess_returns.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside * downside).mean()))
    if downside_dev <= 0:
        return np.nan
    return float((excess_returns.mean() / downside_dev) * np.sqrt(periods_per_year))


def performance_report(
    data: pd.DataFrame,
    ret_col: str = 'net_strat_ret',
    equity_col: str = 'equity',
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    dd_report: dict | None = None,
    trade_stats_report: dict | None = None,
) -> dict:
    '''Compute portfolio-level performance metrics from bar returns/equity.

    Note:
    - Trade-level metrics (for example `WinRate`) are sourced from `trade_stats_report`
      produced by `src/trades.py` to keep a single source of truth.
    '''

    df = data.copy()
    r = df[ret_col].astype(float).fillna(0.0)

    if equity_col not in df.columns:
        df[equity_col] = (1.0 + r).cumprod()

    eq = df[equity_col].astype(float)
    n = int(r.shape[0])

    if n > 1:
        years = n / periods_per_year
        cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
    else:
        cagr = np.nan

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    vol = float(excess.std(ddof=1))
    sharpe = float((excess.mean() / vol) * np.sqrt(periods_per_year)) if vol > 0 else np.nan
    sortino = _safe_sortino(excess, periods_per_year)

    if dd_report is not None and 'max_drawdown' in dd_report:
        max_dd = float(dd_report['max_drawdown'])
    else:
        peak = eq.cummax()
        dd = eq / peak - 1
        max_dd = float(dd.min())

    calmar = float(cagr / abs(max_dd)) if (np.isfinite(cagr) and max_dd < 0) else np.nan

    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    total_return = float(eq.iloc[-1] - 1.0)
    profit_factor = _safe_profit_factor(r)
    skew = float(r.skew()) if n > 2 else np.nan
    kurtosis = float(r.kurt()) if n > 3 else np.nan
    win_rate = float(trade_stats_report['win_rate']) if trade_stats_report and 'win_rate' in trade_stats_report else np.nan

    report = {
        'total_return': total_return,
        'CAGR': cagr,
        'ann_vol': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'ProfitFactor': profit_factor,
        'Skew': skew,
        'Kurtosis': kurtosis,
        'WinRate': win_rate,
        'max_drawdown': max_dd,
        'Calmar': calmar,
        'n_periods': n,
    }
    return report
