import numpy as np


def permutation_importances(estimator, x, y, scoring, n_repeats):
    importances_mat = []
    features = []
    p = estimator.predict(x)
    s = scoring(y, p)

    for col in x:
        features.append(col)
        importances_col = []
        for i in range(n_repeats):
            buf_x = x.apply(lambda x: x.sample(frac=1).reset_index(drop=True) if x.name == col else x)
            buf_p = estimator.predict(buf_x)
            buf_s = scoring(y, buf_p)
            importance = s - buf_s
            importances_col.append(importance)
        importances_mat.append(importances_col)
    importances = np.array(importances_mat)

    return {
        'features': np.array(features),
        'importances_mean': importances.mean(axis=1),
        'importances_std': importances.std(axis=1),
        'importances': importances
    }
