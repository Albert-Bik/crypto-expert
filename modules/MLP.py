import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential

from modules.get_balanced_weights import get_balanced_weights


class MLP:
    def __init__(self,
                 layer_sizes: list[int],
                 epochs: int = 200,
                 batch_size: int = 32,
                 patience: int = 10,
                 is_unbalance: bool = True) -> None:

        model = Sequential()
        for layer_size in layer_sizes:
            model.add(Dense(units=layer_size, activation='tanh'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(
            keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy'
        )
        self.model = model
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.is_unbalance = is_unbalance

    def fit(self, x, y):
        callbacks = [EarlyStopping(monitor='loss', patience=self.patience)]
        sample_weight = get_balanced_weights(y) if self.is_unbalance else None
        self.model.fit(
            x=x, y=y, epochs=self.epochs, batch_size=self.batch_size,
            sample_weight=sample_weight, callbacks=callbacks, verbose=0
        )

    def predict_proba(self, x):
        t = self.model.predict(x, verbose=0)
        ft = np.hstack([1 - t, t])
        return ft

    def predict(self, x):
        p = (self.model.predict(x, verbose=0) > 0.5).astype(int).flatten()
        return p
