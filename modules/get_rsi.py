import numpy as np


def get_rsi(df, window):
    delta = df['close_price'] - df['open_price']
    u = delta.apply(lambda x: 1 if x > 0 else 0)
    d = delta.apply(lambda x: 1 if x < 0 else 0)
    u_ema = u.ewm(span=window, adjust=False, min_periods=window).mean()
    d_ema = d.ewm(span=window, adjust=False, min_periods=window).mean()
    sum_ud_ema = u_ema + d_ema
    sum_ud_ema = sum_ud_ema.replace(0, np.nan)
    rsi = u_ema / sum_ud_ema
    return rsi
