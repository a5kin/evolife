"""
Crossbreading of Diamoeba (center area) and Conway (rest of the board).

Results could vary. In some cases, 'bliamba' structures may appear,
as well as other interesting artefacts.

'Bliamba seed' was originally obtained in this exact setting.

"""

import numpy as np
import random

DEATH_SPEED = 23
BIRTH_COST = 0
MAX_GENES = 9
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 1
RANDOM_SEED = 598275 # 'Bliamba' from almost nothing
FADE_IN = 6
FADE_OUT = 6


def coords2seed(x, y, conway, diamoeba):
    if (500 < x < 700 and 350 < y < 450):
        return random.choice([0, 1]) * diamoeba
    if (400 < x < 800 and 300 < y < 500):
        return random.choice([0, 1]) * random.choice([conway, diamoeba])
    return random.choice([0, 1]) * conway

def fld_init(a):
    conway = a.str2genome("3/23")
    diamoeba = a.str2genome("35678/5678")
    return np.asarray([[coords2seed(i, j, conway, diamoeba) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
