# region Импорт

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('sklearnex').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearnex import patch_sklearn
patch_sklearn(verbose=False)

import dill as dill
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modules.MLP import MLP
from modules.config_logging import config_logging
from modules.cross_val_scores import cross_val_scores
from modules.get_bars import get_bars
from modules.get_pipeline import get_pipeline
from modules.get_rsi import get_rsi
from modules.get_sth import get_sth

# endregion

# region Входные параметры
TICKER = 'BTCUSDT'
FRAME = '1h'
TOTAL_BARS = 8_760
TARGET_LIM = 0.2
WINDOW = 24
LOG_LEVEL = 'i'
# endregion

# region Настройка логов
try:
    config_logging(LOG_LEVEL)
except Exception as e:
    config_logging('i')
    logging.error('Не удалось настроить логи.')
    logging.exception(e)
    exit()
# endregion

# region Получение исторических данных
logging.info('Получение исторических данных ...')

data = None

try:
    data = get_bars(ticker=TICKER, frame=FRAME, total_bars=TOTAL_BARS)
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

start_dt = pd.to_datetime(df['open_ts'].min(), unit='ms').strftime('%d-%m-%y %H-%M')
end_dt = pd.to_datetime(df['open_ts'].max(), unit='ms').strftime('%d-%m-%y %H-%M')

logging.info('Датафрейм подгтовлен.')
# endregion

# region Создание целевой переменной
logging.info('Создание целевой переменной ...')

change = df['close_price'] / df['open_price'] * 100 - 100
df['target'] = change.apply(lambda x: 1 if x > TARGET_LIM else 0)

logging.info('Целевая переменная создана.')
# endregion

# region Создание признаков
# TODO: Добавить признаки: MACD, delta, AROON, ADX, etc.
logging.info('Создание признаков ...')

df['rsi'] = get_rsi(df, WINDOW)
df['sth'] = get_sth(df, WINDOW)

logging.info('Признаки созданы.')
# endregion

# region Обработка признаков
# TODO: Проверить новые признаки: MACD, delta, AROON, ADX, etc.
logging.info('Обработка признаков ...')

df = df[~df.isna().any(axis=1)]
df = df.query('0 <= rsi <= 1')
df = df.query('0 <= sth <= 1')

logging.info('Признаки обработаны.')
# endregion

# region Подготовка тренировочных и тестовых данных
logging.info('Подготовка тренировочных и тестовых данных ...')

features = ['rsi', 'sth']
x_full = df[features]
y_full = df['target']
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, shuffle=False)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

logging.info('Тренировочные и тестовые данные подготовлены.')
# endregion

# region Подготовка моделей

logging.info('Подготовка моделей ...')

# region
models = []
transformers = [
    ('standard_scaler', StandardScaler(), ['rsi', 'sth']),
]
transformer = ColumnTransformer(transformers=transformers)
# endregion

# region LogisticRegression

type_ = 'lrc'
c_list = [0.01, 0.1, 1, 10]
for c in c_list:
    params = dict(
        C=c
    )
    lrc = LogisticRegression(class_weight='balanced', max_iter=10_000, random_state=0, **params)
    model = dict(
        type=type_,
        params=params,
        name=f'{type_} {str(params)}',
        pipeline=get_pipeline(transformer, lrc)
    )
    models.append(model)

# endregion

# region LGBMClassifier

type_ = 'lgc'
strategy_list = ['bagging', 'goss']
num_leaves_list = [2, 3]
for strategy in strategy_list:
    for num_leaves in num_leaves_list:
        params = dict(
            data_sample_strategy=strategy,
            num_leaves=num_leaves
        )
        lgc = LGBMClassifier(is_unbalance=True, random_state=0, verbosity=0, **params)
        model = dict(
            type=type_,
            params=params,
            name=f'{type_} {str(params)}',
            pipeline=get_pipeline(transformer, lgc)
        )
        models.append(model)

# endregion

# region RandomForestClassifier

type_ = 'rfc'
max_features_list = [1, 2]
max_samples_list = [0.1, 0.2]
for max_features in max_features_list:
    for max_samples in max_samples_list:
        params = dict(
            max_features=max_features,
            max_samples=max_samples
        )
        rfc = RandomForestClassifier(class_weight='balanced_subsample', random_state=0, **params)
        model = dict(
            type=type_,
            params=params,
            name=f'{type_} {str(params)}',
            pipeline=get_pipeline(transformer, rfc)
        )
        models.append(model)

# endregion

# region RandomForestClassifier

type_ = 'mlp'
layer_sizes_list = [[2], [3], [4], [5]]
for layer_sizes in layer_sizes_list:
    params = dict(
        layer_sizes=layer_sizes,
    )
    mlp = MLP(**params)
    model = dict(
        type=type_,
        params=params,
        name=f'{type_} {str(params)}',
        pipeline=get_pipeline(transformer, mlp)
    )
    models.append(model)

# endregion

logging.info('Модели подготовлены.')

# endregion

# region Кросс-валидация
logging.info('Кросс-валидация ...')

try:
    for i, model in enumerate(models):
        k_fold = KFold(n_splits=4, shuffle=False)
        cv_scores = cross_val_scores(model['pipeline'], x_train, y_train, cv=k_fold, scoring=roc_auc)
        model['cv_score_name'] = 'ROC-AUC'
        model['cv_scores'] = cv_scores
        model['avg_cv_score'] = cv_scores.mean()
        model['std_cv_score'] = cv_scores.std()
        j = i + 1
        s = len(models)
        r = j / s * 100
        logging.info(f'Model: {j}|{s} ({r:.1f} %).')
except Exception as e:
    logging.error('Не удалось выполнить кросс-валидацию.')
    logging.exception(e)
    exit()

logging.info('Кросс-валидация выполнена.')
# endregion

# region Выбор лучшей модели
logging.info('Выбор лучшей модели ...')

best_model = None
best_score = 0

for model in models:
    if model['avg_cv_score'] > best_score:
        best_model = model
        best_score = model['avg_cv_score']

m = best_model['name']
n = best_model['cv_score_name']
a = best_model['avg_cv_score']
s = best_model['std_cv_score']

logging.info(f'Лучшая модель: {m}.')
logging.info(f'{n} = {a:.3f} ± {s:.3f} (cross-val).')
# endregion

# region Оценка лучшей модели
logging.info('Оценка лучшей модели ...')

r = None

try:
    best_model['pipeline'].fit(x_train, y_train)
    p_test = best_model['pipeline'].predict(x_test)
    r = roc_auc(y_test, p_test)
except Exception as e:
    logging.error('Не удалось выполнить оценку лучшей модели.')
    logging.exception(e)
    exit()

best_model['roc_auc'] = r

logging.info(f'ROC-AUC = {r:.3f} (test).')
# endregion

# region Расчет зависимости precision и recall от threshold
logging.info('Расчет зависимости precision и recall от threshold ...')

thresholds = np.arange(50, 101)
precisions = None
recalls = None

try:
    e_test = best_model['pipeline'].predict_proba(x_test)[:, 1]
    precisions = np.array(
        [precision(y_test, e_test > threshold, zero_division=np.nan) for threshold in thresholds / 100]
    )
    recalls = np.array(
        [recall(y_test, e_test > threshold, zero_division=np.nan) for threshold in thresholds / 100]
    )
except Exception as e:
    logging.error('Не удалось рассчитать зависимости precision и recall от threshold.')
    logging.exception(e)
    exit()

best_model['thresholds'] = thresholds
best_model['precisions'] = precisions
best_model['recalls'] = recalls

logging.info('Зависимости precision и recall от threshold рассчитаны.')
# endregion

# region Обучение лучшей модели на всех данных
logging.info('Обучение лучшей модели на полном наборе данных ...')

try:
    best_model['pipeline'].fit(x_full, y_full)
except Exception as e:
    logging.error('Не удалось обучить лучшую модель на полном наборе данных.')
    logging.exception(e)
    exit()

logging.info('Лучшая модель обучена на полном наборе данных.')
# endregion

# region Сохранение модели
logging.info('Сохранение модели ...')

best_model['ticker'] = TICKER
best_model['frame'] = FRAME
best_model['window'] = WINDOW
best_model['target_lim'] = TARGET_LIM
best_model['total_bars'] = TOTAL_BARS
best_model['start_dt'] = start_dt
best_model['end_dt'] = end_dt

try:
    with open('model.pkl', 'wb') as file:
        dill.dump(best_model, file)
except Exception as e:
    logging.error('Не удалось сохранить модель.')
    logging.exception(e)
    exit()

logging.info('Модель сохранена.')
# endregion
