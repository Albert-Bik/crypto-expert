import numpy as np


def cross_val_scores(estimator, x, y, scoring, cv):
    scores = []
    for train_indices, test_indices in cv.split(x):
        x_train = x.loc[train_indices, ]
        y_train = y.loc[train_indices, ]
        x_test = x.loc[test_indices, ]
        y_test = y.loc[test_indices, ]
        estimator.fit(x_train, y_train)
        p_test = estimator.predict(x_test)
        score = scoring(y_test, p_test)
        scores.append(score)

    return np.array(scores)
