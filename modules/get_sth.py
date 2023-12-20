import numpy as np


def get_sth(df, window):
    roll_max = df['high_price'].rolling(window=window).max()
    roll_min = df['low_price'].rolling(window=window).min()
    roll_range = roll_max - roll_min
    roll_range = roll_range.replace(0, np.nan)
    sth = (df['close_price'] - roll_min) / roll_range
    return sth
