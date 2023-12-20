# region Импорт

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('sklearnex').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

from time import sleep

import dill
import numpy as np
import pandas as pd

from modules.config_logging import config_logging
from modules.get_bars import get_bars
from modules.get_rsi import get_rsi
from modules.get_sth import get_sth

# endregion

# region Настройка логов

config_logging('i')

# endregion

# region Импорт модели
logging.info('Импорт модели ...')

model = None

try:
    with open('../model.pkl', 'rb') as file:
        model = dill.load(file)
except Exception as e:
    logging.error('Не удалось импортировать модель.')
    logging.exception(e)
    exit()

logging.info('Модель импортирована.')
# endregion

# region Получение исторических данных
logging.info('Получение исторических данных ...')

data = None

try:
    data = get_bars(ticker=model['ticker'], frame=model['frame'], total_bars=model['window'])
except Exception as e:
    logging.error('Не удалось получить исторические данные.')
    logging.exception(e)
    exit()

logging.info('Исторические данные получены.')
# endregion

# region Создание датафрейма
logging.info('Создание датафрейма ...')

types = {
    'open_ts': np.int64,
    'open_price': np.float64,
    'high_price': np.float64,
    'low_price': np.float64,
    'close_price': np.float64,
    'base_volume': np.float64,
    'close_ts': np.int64,
    'quote_volume': np.float64,
    'total_trades': np.int64,
    'taker_buy_base_volume': np.float64,
    'taker_buy_quote_volume': np.float64,
    'ignore': 'string'
}
columns = list(types.keys())
df = None

try:
    df = pd.DataFrame(data=data, columns=columns)
except Exception as e:
    logging.error('Не удалось создать датафрейм.')
    logging.exception(e)
    exit()

logging.info('Датафрейм создан.')
# endregion

# region Преобразование типов
logging.info('Преобразование типов ...')

try:
    df = df.astype(dtype=types, errors='raise')
except Exception as e:
    logging.error('Не удалось преобразовать типы.')
    logging.exception(e)
    exit()

logging.info('Типы преобразованы.')
# endregion

# region Проверка загруженных данных
logging.info('Проверка полученных данных ...')

if df.drop(columns='ignore').le(0).any().any():
    logging.error('Обнаружены отрицательные значения.')
    exit()

if df.isna().any().any():
    logging.error('Обнаружены пропуски.')
    exit()

logging.info('Полученные данные корректны.')
# endregion

# region Подготовка датафрейма
logging.info('Подготовка датафрейма ...')

df = df.sort_values(by='open_ts', ascending=True)
columns = ['open_ts', 'open_price', 'high_price', 'low_price', 'close_price']
df = df.loc[:, columns]

logging.info('Датафрейм подгтовлен.')
# endregion

# region Создание признаков
logging.info('Создание признаков ...')

df['rsi'] = get_rsi(df, model['window'])
df['sth'] = get_sth(df, model['window'])

logging.info('Признаки созданы.')
# endregion

# region Обработка признаков
logging.info('Обработка признаков ...')

df = df[~df.isna().any(axis=1)]
df = df.query('0 <= rsi <= 1')
df = df.query('0 <= sth <= 1')

logging.info('Признаки обработаны.')
# endregion

# region Создание входного вектора
logging.info('Создание входного вектора ...')

if df.shape[0] != 1:
    logging.error('Не удалось создать входной вектор.')
    logging.error('df.shape[0] != 1')
    exit()

features = ['rsi', 'sth']
x = df[features]
v = x.round(3).to_dict(orient='records')[0]

logging.info('Входной вектор создан.')
# endregion

# region Получение рекомендаций
logging.info('Получение рекомендаций.')

p, d, e, th, pr, re = [None] * 6

try:
    p = model['pipeline'].predict(x)[0]
    d = {0: 'FLAT', 1: 'BUY'}
    e = model['pipeline'].predict_proba(x)[0][1]
    i = np.abs(model['thresholds'] / 100 - e).argmin()
    th = model['thresholds'][i]
    pr = model['precisions'][i]
    re = model['recalls'][i]
except Exception as e:
    logging.error('Не удалось получить рекомендации.')
    logging.exception(e)
    exit()

logging.info('Рекомендации получены.')
# endregion

# region Вывод информации

sleep(0.2)
print('--------------------------------------------------')
print('Model type:', model['type'])
print('Model params:', model['params'])
print('Model cv_score_name:', model['cv_score_name'])
print('Model avg_cv_score:', f"{model['avg_cv_score']:.3f}")
print('Model std_cv_score:', f"{model['std_cv_score']:.3f}")
print('Model roc_auc:', f"{model['roc_auc']:.3f}")
print('Model ticker:', model['ticker'])
print('Model frame:', model['frame'])
print('Model window:', model['window'])
print('Model target_lim:', model['target_lim'])
print('Model total_bars:', f"{model['total_bars']:,}")
print('Model start_dt:', model['start_dt'])
print('Model end_dt:', model['end_dt'])
print('--------------------------------------------------')
print(f'Input vector: {v}')
print(f'Predict: {d[p]} (proba {e:.3f})')
print(f'Precision: {pr:.3f} (threshold {th / 100:.3f})')
print(f'Recall: {re:.3f} (threshold {th / 100:.3f})')
print('--------------------------------------------------')

# endregion
