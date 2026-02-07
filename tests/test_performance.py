import math

import numpy as np
import pandas as pd

from src.performance import performance_report


def test_sharpe_matches_known_value_without_risk_free_rate() -> None:
    returns = pd.Series([0.01, -0.005, 0.015, 0.0, 0.02], dtype=float)
    df = pd.DataFrame({"net_strat_ret": returns})

    report = performance_report(df, periods_per_year=252, rf_annual=0.0)

    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
    assert np.isclose(report["Sharpe"], expected, rtol=1e-12, atol=1e-12)


def test_cagr_matches_hand_computed_value() -> None:
    equity = pd.Series([1.0, 1.1, 1.21], dtype=float)
    returns = equity.pct_change().fillna(0.0)
    df = pd.DataFrame({"equity": equity, "net_strat_ret": returns})

    report = performance_report(df, periods_per_year=1, rf_annual=0.0)

    expected = (equity.iloc[-1] / equity.iloc[0]) ** (1 / (len(df) / 1)) - 1
    assert np.isclose(report["CAGR"], expected, rtol=1e-12, atol=1e-12)


def test_calmar_is_nan_when_max_drawdown_is_zero() -> None:
    df = pd.DataFrame({"net_strat_ret": [0.0, 0.01, 0.0]})

    report = performance_report(df, periods_per_year=252, dd_report={"max_drawdown": 0.0})

    assert math.isnan(report["Calmar"])


def test_cagr_is_nan_for_single_period_input() -> None:
    df = pd.DataFrame({"net_strat_ret": [0.01]})

    report = performance_report(df, periods_per_year=252)

    assert math.isnan(report["CAGR"])
