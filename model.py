from collections import namedtuple

import numpy as np
import pandas as pd

from constants import COLORS
from constants import FACTION_TO_COLOR_DICT
from constants import HEADERS_AND_TYPES
from constants import PASSING_BONUSES
from constants import ROUND_TILES
from constants import SKIPPED_FACTIONS


SplitRow = namedtuple('SplitRow', ['game_name',
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
                                   'your_faction',
                                   'your_color',
                                   'your_winning_score_pct'
                                   ])


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
        split_rows.append(SplitRow(**new_split_row_dict))
    return split_rows


def main():
    df = pd.read_csv('tm_training_data.csv',
                     names=[h for h, _ in HEADERS_AND_TYPES],
                     dtype=dict(HEADERS_AND_TYPES))
    for row in df.itertuples():
        for split_row in get_game_row_split_into_four_rows(row):
            print(split_row)
        quit()
        # Need to filter out yetis, etc


main()
