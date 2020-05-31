from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data import get_train_and_test_data_plus_raw_test_data


def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def main():
    (train_features, train_labels), (test_features, test_labels), _ = get_train_and_test_data_plus_raw_test_data()
    model = build_model(train_features.shape[1])
    model.summary()
    print('Training model')
    _ = model.fit(train_features,
                  train_labels,
                  epochs=20,
                  validation_split=0.1,
                  verbose=1,)
    print('Evaluating model')
    model.evaluate(test_features, test_labels)
    date_string = datetime.today().strftime('%Y%m%d-%H%M%S')
    model_save_path = 'tm_model_{}.h5'.format(date_string)
    print('Saving model to {}'.format(model_save_path))
    model.save(model_save_path)

main()
