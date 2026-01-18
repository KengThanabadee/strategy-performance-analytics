import pandas as pd
import numpy as np
import mplfinance as mpf


def build_trade_marks(bt: pd.DataFrame, pos_col='pos', px_col='Open'):
    if pos_col not in bt.columns:
        raise KeyError(f'Missing column: {pos_col}. Run backtest first to create it.')
    if px_col not in bt.columns:
        raise KeyError(f'Missing column: {px_col}.')
    
    pos = bt[pos_col].fillna(0).astype(float)
    prev = pos.shift(1).fillna(0)
    changed = pos.ne(prev)

    is_entry_long = changed & (pos > 0) & (prev <= 0)
    is_exit_long = changed & (pos <= 0) & (prev > 0)
    is_entry_short = changed & (pos < 0) & (prev >= 0)
    is_exit_short = changed & (pos >= 0) & (prev < 0)

    px = bt[px_col].astype(float)

    marks = {
        'entry_long': px.where(is_entry_long),
        'exit_long': px.where(is_exit_long),
        'entry_short': px.where(is_entry_short),
        'exit_short': px.where(is_exit_short)
    }

    flags = pd.DataFrame({
        'changed': changed,
        'is_entry_long': is_entry_long,
        'is_exit_long': is_exit_long,
        'is_entry_short': is_entry_short,
        'is_exit_short': is_exit_short
    }, index=bt.index)

    return marks, flags

def plot_candles(bt, addplots=None, title='', style='yahoo', volume=False, panel_ratios=(3,1)):
    addplots = addplots or []
    mpf.plot(
        bt,
        type='candle',
        addplot=addplots,
        volume=volume,
        style=style,
        title=title,
        panel_ratios=panel_ratios,
        figsize=(14, 7)
    )

def build_trade_addplots(bt, marks, offset_up=1.05, offset_dn=0.95):
    aps = []

    if marks['entry_long'].notna().any():
        y = marks['entry_long'] * offset_dn
        aps.append(mpf.make_addplot(y, type='scatter', marker='^', markersize=30, color='g'))

    if marks['exit_long'].notna().any():
        y = marks['exit_long'] * offset_up
        aps.append(mpf.make_addplot(y, type='scatter', marker='v', markersize=30, color='r'))

    if marks['entry_short'].notna().any():
        y = marks['entry_short'] * offset_up
        aps.append(mpf.make_addplot(y, type='scatter', marker='v', markersize=30, color='r'))
    
    if marks['exit_short'].notna().any():
        y = marks['exit_short'] * offset_dn
        aps.append(mpf.make_addplot(y, type='scatter', marker='^', markersize=30, color='g'))
    
    return aps

def rsi_spec(n, panel=1):
    return {
        'type': 'line',
        'col': f'rsi_{n}',
        'panel': panel,
        'color': 'dodgerblue'
    }

def macd_spec(panel=1):
    return {
        'type': 'macd',
        'panel': panel,
        'colors': {
            'macd': 'blue',
            'signal': 'orange',
            'hist_pos': 'green',
            'hist_neg': 'red'
        }
    }

def rsi_bands_spec(panel=1, lo=30, mid=50, hi=70, color='gray', linestyle='--'):
    return {
        'type': 'hline',
        'panel': panel,
        'y': [lo, mid, hi],
        'color': color,
        'linestyle': linestyle
    }

def build_indicator_addplots(bt, specs):
    """
    Build mplfinance addplots for indicators.
    
    bt : pd.DataFrame
    specs : list of dict
    
    Supported spec types:
     - line: {'type':'line','col':'rsi_14','panel':1,'color':...}
     - bar : {'type':'bar','col':'xxx','panel':1,'color':...}
     - macd: {'type':'macd','panel':2,'colors':{'macd':...,'signal':...,'hist_pos':...,'hist_neg':...}}
     - hline: {'type':'hline','panel':1,'y':[30,50,70],'color':...,'linestyle':'--'}

    Returns: list of mpf addplot objects
    """
    aps = []

    for s in specs:
        t = s['type']

        # generic line
        if t == 'line':
            col = s['col']
            panel = s.get('panel', 1)
            if col not in bt.columns:
                raise KeyError(f'Missing column: {col}')

            kwargs = {}
            if 'color' in s:
                kwargs['color'] = s['color']

            aps.append(mpf.make_addplot(bt[col], panel=panel, **kwargs))

        # generic bar
        elif t == 'bar':
            col = s['col']
            panel = s.get('panel', 1)
            if col not in bt.columns:
                raise KeyError(f'Missing column: {col}')

            kwargs = {}
            if 'color' in s:
                kwargs['color'] = s['color']

            aps.append(mpf.make_addplot(bt[col], type='bar', panel=panel, **kwargs))

        # MACD bundle
        elif t == 'macd':
            panel = s.get('panel', 1)
            for col in ['macd', 'signal', 'histogram']:
                if col not in bt.columns:
                    raise KeyError(f'Missing column: {col}')

            colors = s.get('colors', {})
            c_macd = colors.get('macd', None)
            c_signal = colors.get('signal', None)
            c_pos = colors.get('hist_pos', None)
            c_neg = colors.get('hist_neg', None)

            # macd + signal line
            kw1 = {'color': c_macd} if c_macd is not None else {}
            kw2 = {'color': c_signal} if c_signal is not None else {}
            
            aps.append(mpf.make_addplot(bt['macd'], panel=panel, **kw1))
            aps.append(mpf.make_addplot(bt['signal'], panel=panel, **kw2))

            # histogram
            h = bt['histogram'].astype(float)
            h_pos = h.where(h >= 0)
            h_neg = h.where(h < 0)

            kwp = {'color': c_pos} if c_pos is not None else {}
            kwn = {'color': c_neg} if c_neg is not None else {}

            aps.append(mpf.make_addplot(h_pos, type='bar', panel=panel, **kwp))
            aps.append(mpf.make_addplot(h_neg, type='bar', panel=panel, **kwn))

        # Horizontal line
        elif t == 'hline':
            panel = s.get('panel', 1)
            y = s.get('y', None)
            if y is None:
                raise KeyError('hline spec missing y value')

            # normalize y into list
            if isinstance(y, (int, float)):
                y_levels = [float(y)]
            else:
                y_levels = [float(v) for v in y]

            color = s.get('color', None)
            linestyle = s.get('linestyle', '--')

            for lvl in y_levels:
                line = pd.Series(lvl, index=bt.index)
                kwargs = {'linestyle': linestyle}
                if color is not None:
                    kwargs['color'] = color
                aps.append(mpf.make_addplot(line, panel=panel, **kwargs))

        else:
            raise ValueError(f'Unknown spec type: {t}')

    return aps