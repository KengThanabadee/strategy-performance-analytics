import matplotlib.pyplot as plt
from drawdown import drawdown_report

def ensure_drawdown_cols(bt, equity_col='equity'):
    df = bt.copy()

    need_cols = ('drawdown' not in df.columns) or ('peak' not in df.columns)
    df, dd_rep, spells_df = drawdown_report(df, equity_col=equity_col, add_cols=need_cols)
    return df, dd_rep, spells_df

def plot_equity(df, equity_col='equity', title_prefix=''):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df[equity_col])
    ax.set_title(f'{title_prefix} Equity Curve'.strip())
    ax.grid(True)
    plt.show()

def plot_underwater(df, title_prefix=''):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(df.index, df['drawdown'], 0, color='red', alpha=0.35)
    ax.set_title(f'{title_prefix} Underwater (Drawdown)'.strip())
    ax.grid(True)
    plt.show()

def plot_equity_with_maxdd_episode(df, dd_rep, equity_col='equity', title_prefix=''):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df[equity_col])

    # highlight max DD episode: start -> recovery (or end)
    s = dd_rep.get('max_dd_start', None)
    rec = dd_rep.get('max_dd_recovery_date', None)
    e = rec if rec is not None else df.index[-1]
    if s is not None:
        ax.axvspan(s, e, color='red', alpha=0.2)

    ax.set_title(f'{title_prefix} Equity Curve'.strip())
    ax.grid(True)
    plt.show()


def plot_position(df, title_prefix=''):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(df.index, df['pos'])
    ax.set_title(f'{title_prefix} Position'.strip())
    ax.grid(True)
    plt.show()

def plot_pack(bt, equity_col='equity', title_prefix=''):
    df, dd_rep, spells_df = ensure_drawdown_cols(bt, equity_col=equity_col)

    plot_equity_with_maxdd_episode(df, dd_rep, equity_col=equity_col, title_prefix=title_prefix)
    plot_underwater(df, title_prefix=title_prefix)
    plot_position(df, title_prefix=title_prefix)

    return df, dd_rep, spells_df
