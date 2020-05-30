import numpy as np
import pandas as pd


HEADERS_AND_TYPES = [
    ('game_name', str),
    ('missing_pass_1', str),
    ('missing_pass_2', str),
    ('missing_pass_3', str),
    ('tile_r1', str),
    ('tile_r2', str),
    ('tile_r3', str),
    ('tile_r4', str),
    ('tile_r5', str),
    ('tile_r6', str),
    ('faction_p1', str),
    ('faction_p2', str),
    ('faction_p3', str),
    ('faction_p4', str),
    ('score_p1', float),
    ('score_p2', float),
    ('score_p3', float),
    ('score_p4', float)
]


PASSING_BONUSES = ['pass:BON1', 'pass:BON2', 'pass:BON5', 'pass:BON6', 'pass:BON7', 'pass:BON8', 'pass:BON9']
FACTIONS_AND_COLORS = [('alchemists', 'black'),
                       ('auren', 'green'),
                       ('chaosmagicians', 'red'),
                       ('cultists', 'brown'),
                       ('darklings', 'black'),
                       ('dwarves', 'gray'),
                       ('engineers', 'gray'),
                       ('fakirs', 'yellow'),
                       ('giants', 'red'),
                       ('halflings', 'brown'),
                       ('mermaids', 'blue'),
                       ('nomads', 'yellow'),
                       ('swarmlings', 'blue'),
                       ('witches', 'green'),
                       ]
SKIPPED_FACTIONS = ['acolytes',
                    'dragonlords',
                    'icemaidens',
                    'riverwalkers',
                    'shapeshifters',
                    'yetis']
COLORS = sorted({c for f, c in FACTIONS_AND_COLORS})


def get_n_hot_array(hot_values, all_values):
    return np.array([1. if v in hot_values else 0. for v in all_values])


def main():
    # df = pd.read_csv('tm_training_data.csv',
    #                  names=[h for h, _ in HEADERS_AND_TYPES],
    #                  dtype=dict(HEADERS_AND_TYPES))
    print(get_n_hot_array(['black', 'green', 'red'], COLORS))


main()
