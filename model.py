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


def main():
    df = pd.read_csv('tm_training_data.csv',
                     names=[h for h, _ in HEADERS_AND_TYPES],
                     dtype=dict(HEADERS_AND_TYPES))
    print(df)


main()
