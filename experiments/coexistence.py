"""
Coexistence test. Board is initialized with several pre-designed rules.

In the process of evolution, 18-22 species survived forming stable oscillators.

"""

import numpy as np
import random

DEATH_SPEED = 0
BIRTH_COST = 15
MAX_GENES = 9
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 1
RANDOM_SEED = None

def fld_init(a):
    species = map(a.str2genome, ["35678/5678", "35678/678", "23567/5678", "3567/35678", "35678/5678",  "35678/678", "35678/678"])
    return np.asarray([[random.choice([0, 1]) * random.choice(species) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
