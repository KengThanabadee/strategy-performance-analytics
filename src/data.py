from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import yfinance as yf

from .config import DATA_DIR


def _safe_token(value: str | None) -> str:
    if value is None or value == "":
        return "none"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _cache_path(
    ticker: str,
    period: str | None,
    start: str | None,
    end: str | None,
    interval: str,
    auto_adjust: bool,
    multi_level_index: bool,
    cache_dir: Path,
) -> Path:
    token = "_".join(
        [
            _safe_token(ticker),
            _safe_token(period),
            _safe_token(start),
            _safe_token(end),
            _safe_token(interval),
            "adj" if auto_adjust else "raw",
            "multi" if multi_level_index else "single",
        ]
    )
    return cache_dir / f"{token}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires 'pyarrow' or 'fastparquet'. "
            "Install one to enable caching."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed to read cache at {path}: {exc}")
        return None


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires 'pyarrow' or 'fastparquet'. "
            "Install one to enable caching."
        ) from exc


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


def _warn_gaps(index: pd.Index) -> None:
    if len(index) < 3:
        return
    freq = pd.infer_freq(index)
    if freq is not None:
        expected = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
        gaps = index.to_series().diff() > expected * 1.5
    else:
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return
        gaps = deltas > deltas.median() * 3
    if gaps.any():
        warnings.warn("Potential missing bars detected in price data.")

def load_yf(
    ticker: str,
    period: str | None = '1y',
    start: str | None = None,
    end: str | None = None,
    interval: str = '1d',
    auto_adjust: bool = False,
    progress: bool = False,
    multi_level_index: bool = False,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    cache_path = _cache_path(
        ticker=ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        multi_level_index=multi_level_index,
        cache_dir=cache_dir,
    )

    if use_cache and not force_refresh:
        cached = _read_cache(cache_path)
        if cached is not None and not cached.empty:
            df = _standardize(cached)
            _warn_gaps(df.index)
            return df

    df = yf.download(
        ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        multi_level_index=multi_level_index,
    )

    if df is None or df.empty:
        raise ValueError(f'No data returned from yfinance for {ticker}')

    df = _standardize(df)
    _warn_gaps(df.index)

    if use_cache:
        _write_cache(df, cache_path)

    return df


def load_yf_multi(
    tickers: list[str],
    *,
    return_as: str = "dict",
    **kwargs,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    data = {ticker: load_yf(ticker, **kwargs) for ticker in tickers}

    if return_as == "dict":
        return data
    if return_as == "multiindex":
        return pd.concat(data, axis=1)

    raise ValueError("return_as must be 'dict' or 'multiindex'")
