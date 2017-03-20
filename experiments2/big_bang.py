"""
A small 100x100 area is randomly initialized with all genes possible.

A burst of colors quickly filling in the whole field, then after a time
stabilizing in a map-like design.

"""

import numpy as np
import random

DEATH_SPEED = 0
BIRTH_COST = 0
MAX_GENES = 9
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 1
FADE_IN = 6
FADE_OUT = 6
RANDOM_SEED = None


def fld_init(a):
    return np.asarray([[(random.choice([0, 1]) * random.randint(0, 256*512) if (i < 100 and j < 100) else 0) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
