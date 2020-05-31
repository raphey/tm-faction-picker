import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


FEATURE_COUNT = 119


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[FEATURE_COUNT]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


m = build_model()

print(m)