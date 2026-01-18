import pandas as pd
import numpy as np

def drawdown_report(data, equity_col='equity', add_cols=True):
    df = data.copy()
    eq = df[equity_col].astype(float)

    peak = eq.cummax()
    dd = eq / peak - 1
    max_dd_cummin = dd.cummin()

    if add_cols:
        df['peak'] = peak
        df['drawdown'] = dd
        df['max_drawdown'] = max_dd_cummin

    # max DD event
    dd_end = dd.idxmin()
    dd_start = df.loc[:dd_end, equity_col].idxmax()
    max_dd = float(dd.loc[dd_end])

    # durations require datetime index (for days); bars always available
    is_time_index = isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex))

    # drawdown spells windows
    in_dd = dd < 0
    last_spell_unrecovered = bool(in_dd.iloc[-1])
    
    spell_starts = df.index[in_dd & ~in_dd.shift(1, fill_value=False)]
    spell_ends = df.index[~in_dd & in_dd.shift(1, fill_value=False)]

    if len(spell_ends) < len(spell_starts):
        # last spell still in drawdown at end of sample
        spell_ends = spell_ends.append(pd.Index([df.index[-1]]))

    spells = []
    for j, (s, e) in enumerate(zip(spell_starts, spell_ends)):
        seg = dd.loc[s:e]
        trough_date = seg.idxmin()
        trough_dd = float(seg.loc[trough_date])
        duration_bars = int(seg.shape[0])

        is_last_spell = (j == (len(spell_starts) - 1))
        recovered_spell = not (is_last_spell and last_spell_unrecovered)
        row = {
            'start': s,
            'end': e,
            'duration_bars': duration_bars,
            'trough_date': trough_date,
            'trough_dd': trough_dd,
            'recovered': recovered_spell
        }
        if is_time_index:
            td = (e-s)
            row['duration_timedelta'] = td
        
        spells.append(row)

    spells_df = pd.DataFrame(spells)

    # recovery info for max DD spells
    peak_level = float(eq.loc[dd_start])
    after = eq.loc[dd_end:]
    recov_dates = after.index[after >= peak_level]
    recovered = len(recov_dates) > 0
    dd_recovery = recov_dates[0] if recovered else None

    if is_time_index:
        max_dd_duration_days = float((dd_end - dd_start) / pd.Timedelta(days=1))
        max_spell_duration_days = (
            float(spells_df['duration_timedelta'].max() / pd.Timedelta(days=1))
            if len(spells_df) > 0 and 'duration_timedelta' in spells_df.columns
            else 0
        )
    else:
        max_dd_duration_days = None
        max_spell_duration_days = None

    report = {
        'max_drawdown': max_dd,
        'max_dd_start': dd_start,
        'max_dd_end': dd_end,
        'max_dd_recovered': recovered,
        'max_dd_recovery_date': dd_recovery,

        # time aware (float days; supports intraday)
        'max_dd_duration_days': max_dd_duration_days,
        'max_spell_duration_days': max_spell_duration_days,

        # always available (bars)
        'max_dd_duration_bars': int(dd.loc[dd_start:dd_end].shape[0]) if dd_start is not None else None,
        'max_spell_duration_bars': int(spells_df['duration_bars'].max()) if len(spells_df) > 0 else 0,

        'num_drawdown_spells': int(len(spells_df))
    }

    return df, report, spells_df