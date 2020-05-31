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
    example_batch = train_features[:10]
    example_result = model.predict(example_batch)
    print(example_result)


main()
