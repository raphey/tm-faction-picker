from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from data import get_processed_features_for_all_possible_picks
from data import get_train_and_test_data_plus_raw_test_data
from data import PredictionGameState


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


def train_and_save():
    (train_features, train_labels), (test_features, test_labels), _ = get_train_and_test_data_plus_raw_test_data()
    model = build_model(train_features.shape[1])
    model.summary()
    print('Training model')
    _ = model.fit(train_features,
                  train_labels,
                  epochs=1,
                  validation_split=0.1,
                  verbose=1,)
    print('Evaluating model')
    model.evaluate(test_features, test_labels)
    date_string = datetime.today().strftime('%Y%m%d-%H%M%S')
    model_save_path = 'tm_model_{}.h5'.format(date_string)
    print('Saving model to {}'.format(model_save_path))
    model.save(model_save_path)


def play_with_model(model_path='tm_model_20200530-220911.h5'):
    model = keras.models.load_model(model_path)
    sample_game_1 = PredictionGameState(missing_bonuses=['pass:BON9', 'pass:BON6', 'pass:BON3'],
                                        missing_round_tiles=['SCORE3', 'SCORE5', 'SCORE9'],
                                        tile_r1='SCORE8',
                                        tile_r2='SCORE7',
                                        tile_r3='SCORE2',
                                        tile_r4='SCORE1',
                                        tile_r5='SCORE6',
                                        tile_r6='SCORE4',
                                        previous_factions_picked=[],
                                        previous_colors_picked=[],
                                        your_player_number=1
                                        )
    processed_features_all_factions = get_processed_features_for_all_possible_picks(sample_game_1)
    for faction, processed_features in processed_features_all_factions.items():
        print('Prediction for {}: {}'.format(faction, model.predict(np.array([processed_features]))))
    # TODO: should run all predictions at once and save time
    print('***********')
    # Try that same game, but put coin bonus in the first round, which should favor cultists
    sample_game_2 = PredictionGameState(missing_bonuses=['pass:BON9', 'pass:BON6', 'pass:BON3'],
                                        missing_round_tiles=['SCORE3', 'SCORE5', 'SCORE9'],
                                        tile_r1='SCORE1',
                                        tile_r2='SCORE7',
                                        tile_r3='SCORE2',
                                        tile_r4='SCORE8',
                                        tile_r5='SCORE6',
                                        tile_r6='SCORE4',
                                        previous_factions_picked=[],
                                        previous_colors_picked=[],
                                        your_player_number=1
                                        )
    processed_features_all_factions = get_processed_features_for_all_possible_picks(sample_game_2)
    for faction, processed_features in processed_features_all_factions.items():
        print('Prediction for {}: {}'.format(faction, model.predict(np.array([processed_features]))))
    print('***********')
    # One more attempt, this time skewing the bonuses against darklings
    sample_game_3 = PredictionGameState(missing_bonuses=['pass:BON8', 'pass:BON5', 'pass:BON3'],
                                        missing_round_tiles=['SCORE3', 'SCORE5', 'SCORE9'],
                                        tile_r1='SCORE1',
                                        tile_r2='SCORE7',
                                        tile_r3='SCORE2',
                                        tile_r4='SCORE8',
                                        tile_r5='SCORE6',
                                        tile_r6='SCORE4',
                                        previous_factions_picked=[],
                                        previous_colors_picked=[],
                                        your_player_number=1
                                        )
    processed_features_all_factions = get_processed_features_for_all_possible_picks(sample_game_3)
    for faction, processed_features in processed_features_all_factions.items():
        print('Prediction for {}: {}'.format(faction, model.predict(np.array([processed_features]))))


play_with_model()
