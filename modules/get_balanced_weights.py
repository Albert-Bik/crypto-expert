import numpy as np


def get_balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(ar=y, return_counts=True)
    weights = counts.min() / counts
    classes_weights = {c: w for c, w in zip(classes, weights)}
    balanced_weights = np.array([classes_weights[c] for c in y])

    return balanced_weights
