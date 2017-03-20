"""
Mutated Conway rules. One random gene of each cell mutated.

DESCRIBE ME, or at least pick up better params for new engine.

"""

import numpy as np
import random

DEATH_SPEED = 1
BIRTH_COST = 3
MAX_GENES = 14
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 1
RANDOM_SEED = None
FADE_IN = 6
FADE_OUT = 6

def fld_init(a):
    conway = a.str2genome("3/23")
    #return np.asarray([[random.choice([0, 1]) * (conway | (1 << random.randint(0, 17))) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    return np.asarray([[random.choice([0, 1]) * (conway ^ (1 << random.randint(0, 17))) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    #return np.asarray([[(random.choice([0, 1]) * conway) if (i > 400 and i < 800 and j > 300 and j < 500) else 0 for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
