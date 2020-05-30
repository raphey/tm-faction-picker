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


PASSING_BONUSES = ['pass:BON{}'.format(i) for i in range(1, 11)]

ROUND_TILES = ['SCORE{}'.format(i) for i in range(1, 10)]

FACTION_TO_COLOR_DICT = {'alchemists': 'black',
                         'auren': 'green',
                         'chaosmagicians': 'red',
                         'cultists': 'brown',
                         'darklings': 'black',
                         'dwarves': 'gray',
                         'engineers': 'gray',
                         'fakirs': 'yellow',
                         'giants': 'red',
                         'halflings': 'brown',
                         'mermaids': 'blue',
                         'nomads': 'yellow',
                         'swarmlings': 'blue',
                         'witches': 'green',
                         }

FACTIONS = sorted([f for f in FACTION_TO_COLOR_DICT.keys()])

COLORS = sorted({c for c in FACTION_TO_COLOR_DICT.values()})
