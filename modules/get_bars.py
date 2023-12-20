import logging

from modules.Client import Client


def get_bars(ticker: str,
             frame: str,
             total_bars: int) -> list[list]:

    batch_size = 499
    client = Client()
    bars = []

    limit = min(batch_size, total_bars)
    batch = client.get_bars(ticker=ticker, frame=frame, limit=limit)
    bars.extend(batch)
    s = len(bars)
    r = len(bars) / total_bars * 100
    logging.info(f'Bars downloaded: {s:,} ({r:.1f} %).')

    while len(bars) < total_bars:
        end_ts = batch[0][0]
        limit = min(batch_size, total_bars - len(bars) + 1)
        batch = client.get_bars(ticker=ticker, frame=frame, end_ts=end_ts, limit=limit)
        batch = batch[:-1]
        bars.extend(batch)
        s = len(bars)
        r = len(bars) / total_bars * 100
        logging.info(f'Bars downloaded: {s:,} ({r:.1f} %).')

    return bars
