"""
Entire field is randomly initialized with all genes possible.

From a 'white noise' a solid maze-like structures formed,
and slowly begin to devour 'lakes' of strange pixel art.

"""

import numpy as np
import random

DEATH_SPEED = 23
BIRTH_COST = 0
MAX_GENES = 14
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 6
RANDOM_SEED = None

def fld_init(a):
    #d = a.str2genome("3/23")
    #return np.asarray([[(random.choice([0, 1]) * d) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    return np.asarray([[(random.choice([0, 1]) * random.randint(0, 256*512)) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
