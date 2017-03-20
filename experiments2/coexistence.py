"""
Coexistence test. Board is initialized with several pre-designed rules.

TODO: pick up good rules. This one doesn't work well with new version.

"""

import numpy as np
import random

DEATH_SPEED = 0
BIRTH_COST = 15
MAX_GENES = 9
FIELD_WIDTH = 1200
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 10
RANDOM_SEED = None
FADE_IN = 255
FADE_OUT = 6

def rnd_genome(num_genes):
    genome = 0
    b = str(random.randint(3, 8)) + str(random.randint(3, 8))
    s = str(random.randint(2, 8)) + str(random.randint(2, 8))
    if random.choice([True, False]):
        return b[1:] + "/" + s
    else:
        return b + "/" + s[:-1]

def fld_init(a):
    #species = map(a.str2genome, ["347/23456", "3/23", "3/234"])
    #return np.asarray([[random.choice([0, 1]) * (a.str2genome("347/23456") if i < 100 and j < 100 else a.str2genome("3/23")) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    species = map(a.str2genome, ["35678/5678", "35678/678", "23567/5678", "3567/35678", "35678/5678",  "35678/678", "35678/678"])
    return np.asarray([[random.choice([0, 1]) * random.choice(species) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    #return np.asarray([[random.choice([0, 1]) * random.choice(species) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
    #return np.asarray([[random.choice([0, 1]) * a.str2genome(rnd_genome(3)) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
