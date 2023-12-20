from urllib.parse import urlencode

import requests


class Client:
    def __init__(self) -> None:
        self.session = requests.Session()

    def get_bars(self,
                 ticker: str,
                 frame: str,
                 start_ts: int | None = None,
                 end_ts: int | None = None,
                 limit: int | None = 499) -> list[list]:

        """
        limit [1,100) ---> weight = 1\n
        limit [100, 500) ---> weight = 2\n
        limit [500, 1_000] ---> weight = 5\n
        limit (1_000, 1_500} ---> weight = 10\n
        """

        # region Проверка входных параметров
        if type(ticker) != str:
            msg = 'type(ticker) != str'
            raise Exception(msg)

        if ticker != ticker.upper():
            msg = 'ticker != ticker.upper()'
            raise Exception(msg)

        frames = '1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M'.split()
        if frame not in frames:
            msg = f'frame not in {frames}'
            raise Exception(msg)

        if type(start_ts) not in (int, type(None)):
            msg = 'type(start_ts) not in (int, type(None))'
            raise Exception(msg)

        if start_ts is not None and start_ts < 0:
            msg = 'start_ts < 0'
            raise Exception(msg)

        if type(end_ts) not in (int, type(None)):
            msg = 'type(end_ts) not in (int, type(None))'
            raise Exception(msg)

        if end_ts is not None and end_ts < 0:
            msg = 'end_ts < 0'
            raise Exception(msg)

        if start_ts is not None and end_ts is not None and end_ts < start_ts:
            msg = 'end_ts < start_ts'
            raise Exception(msg)

        if type(limit) not in (int, type(None)):
            msg = 'type(limit) not in (int, type(None))'
            raise Exception(msg)

        if limit is not None and limit < 1:
            msg = 'limit < 1'
            raise Exception(msg)

        if limit is not None and limit > 1_500:
            msg = 'limit > 1_500'
            raise Exception(msg)
        # endregion

        base_url = 'https://fapi.binance.com'
        endpoint = '/fapi/v1/klines'
        params = {
            'symbol': ticker,
            'interval': frame,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': limit
        }
        url = base_url + endpoint + '?' + urlencode(params)
        response = self.session.get(url)
        data = response.json()

        return data
