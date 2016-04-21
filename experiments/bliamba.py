"""
Field is initialized with so-called 'bliamba seed'
Very complex behaviour. Produces stable ecosystem of many species.

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
FRAME_SKIP = 16
RANDOM_SEED = None

def fld_init(a):
    fld = np.zeros((a.width, a.height), dtype=np.int32)
    sample = np.load("./fields/bliamba_seed.npy")
    w, h = sample.shape
    fld[:w,:h] = sample
    return fld
