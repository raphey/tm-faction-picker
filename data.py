from collections import namedtuple

import numpy as np
import pandas as pd

from constants import COLORS
from constants import FACTION_TO_COLOR_DICT
from constants import FACTIONS
from constants import HEADERS_AND_TYPES
from constants import PASSING_BONUSES
from constants import ROUND_TILES


PREDICTION_FEATURES = [
    'missing_bonuses',
    'missing_round_tiles',
    'tile_r1',
    'tile_r2',
    'tile_r3',
    'tile_r4',
    'tile_r5',
    'tile_r6',
    'previous_factions_picked',
    'previous_colors_picked',
    'your_player_number',
]

TRAINING_FEATURES = PREDICTION_FEATURES + [
    'game_name',
    'your_faction',
    'your_color',
    'your_winning_score_pct'
]

TrainingGameState = namedtuple('TrainingGameState', TRAINING_FEATURES)

PredictionGameState = namedtuple('PredictionGameState', PREDICTION_FEATURES)


def get_n_hot_array(hot_values, all_values):
    return np.array([1. if v in hot_values else 0. for v in all_values])


def get_game_row_split_into_four_rows(game_row):
    """
    split a single namedtuple game row into four namedtuple rows
    """
    split_rows = []
    factions_picked = []
    colors_picked = []
    missing_bonuses = (game_row.missing_pass_1, game_row.missing_pass_2, game_row.missing_pass_3)
    present_round_tiles = [getattr(game_row, 'tile_r{}'.format(i)) for i in range(1, 7)]
    missing_round_tiles = [rt for rt in ROUND_TILES if rt not in present_round_tiles]
    winning_score = max(getattr(game_row, 'score_p{}'.format(i)) for i in range(1, 5))
    for i in range(1, 5):
        your_faction = getattr(game_row, 'faction_p{}'.format(i))
        your_color = FACTION_TO_COLOR_DICT[your_faction]
        your_winning_score_pct = getattr(game_row, 'score_p{}'.format(i)) / winning_score
        new_split_row_dict = {k: getattr(game_row, k) for k in ['game_name'] +
                              ['tile_r{}'.format(i) for i in range(1, 7)]}
        new_split_row_dict['missing_bonuses'] = missing_bonuses
        new_split_row_dict['missing_round_tiles'] = missing_round_tiles
        new_split_row_dict['previous_factions_picked'] = tuple(factions_picked)
        new_split_row_dict['previous_colors_picked'] = tuple(colors_picked)
        new_split_row_dict['your_player_number'] = i
        new_split_row_dict['your_faction'] = your_faction
        new_split_row_dict['your_color'] = your_color
        new_split_row_dict['your_winning_score_pct'] = your_winning_score_pct
        factions_picked.append(your_faction)
        colors_picked.append(your_color)
        split_rows.append(TrainingGameState(**new_split_row_dict))
    return split_rows


def filter_out_extra_factions(df):
    for i in range(1, 5):
        df = df[getattr(df, 'faction_p{}'.format(i)).isin(FACTIONS)]
    return df


def get_feature_array_and_label(split_row):
    missing_bonuses = get_n_hot_array(split_row.missing_bonuses, PASSING_BONUSES)
    missing_round_tiles = get_n_hot_array(split_row.missing_round_tiles, ROUND_TILES)
    round_tiles = np.concatenate([get_n_hot_array([getattr(split_row, 'tile_r{}'.format(i))], ROUND_TILES)
                                 for i in range(1, 7)])
    previous_factions_picked = get_n_hot_array(split_row.previous_factions_picked, FACTIONS)
    previous_colors_picked = get_n_hot_array(split_row.previous_colors_picked, COLORS)
    your_player_number = get_n_hot_array([split_row.your_player_number], list(range(1, 5)))
    your_faction = get_n_hot_array([split_row.your_faction], FACTIONS)
    your_color = get_n_hot_array([split_row.your_color], COLORS)

    feature_array = np.concatenate([missing_bonuses,
                                    missing_round_tiles,
                                    round_tiles,
                                    previous_factions_picked,
                                    previous_colors_picked,
                                    your_player_number,
                                    your_faction,
                                    your_color])
    label = split_row.your_winning_score_pct
    return feature_array, label


def get_df_from_file(training_data_path):
    df = pd.read_csv(training_data_path,
                     names=[h for h, _ in HEADERS_AND_TYPES],
                     dtype=dict(HEADERS_AND_TYPES))
    df = filter_out_extra_factions(df)
    return df


def get_processed_data(df):
    all_features, all_labels = [], []
    for row in df.itertuples():
        for split_row in get_game_row_split_into_four_rows(row):
            features, label = get_feature_array_and_label(split_row)
            all_features.append(features)
            all_labels.append(label)
    return np.array(all_features), np.array(all_labels)


def get_train_and_test_data_plus_raw_test_data(training_data_path='tm_training_data_20200706.csv'):
    df = get_df_from_file(training_data_path)
    raw_train_data = df.sample(frac=0.9, random_state=0)
    raw_test_data = df.drop(raw_train_data.index)
    train_data = get_processed_data(raw_train_data)
    test_data = get_processed_data(raw_test_data)
    return train_data, test_data, raw_test_data


def get_processed_features_for_all_possible_picks(prediction_game_state):
    faction_features = {}
    for faction in FACTIONS:
        color = FACTION_TO_COLOR_DICT[faction]
        hypothetical_game_state_dict = prediction_game_state._asdict()
        hypothetical_game_state_dict['your_faction'] = faction
        hypothetical_game_state_dict['your_color'] = color
        hypothetical_game_state_dict['game_name'] = 'hypothetical_game'  # dummy game name
        hypothetical_game_state_dict['your_winning_score_pct'] = 1.0     # dummy label
        hypothetical_game_state = TrainingGameState(**hypothetical_game_state_dict)
        faction_features[faction] = get_feature_array_and_label(hypothetical_game_state)[0]
        print(faction, faction_features[faction][-21:])
        print(faction, faction_features[faction][:-21])
    return faction_features
