import pandas as pd
import yfinance as yf

def load_yf(
    ticker: str,
    period: str | None = '1y',
    start: str | None = None,
    end: str | None = None,
    interval: str = '1d',
    auto_adjust: bool = False,
    progress: bool = False,
    multi_level_index: bool = False
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        multi_level_index=multi_level_index
    )

    if df is None or df.empty:
        raise ValueError(f'No data returned from yfinance for {ticker}')

    # Standardize
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalize column names: Open/High/Low/Close/Adj close/Volume to Open/High/Low/Close/Adj Close/Volume
    df.columns = [str(c).strip().title() for c in df.columns]

    return df