"""
Crossbreading of Diamoeba (center area) and Conway (rest of the board).

Results could vary. In rare case (<1%) 'bliamba' structures may appear,
as well as other interesting artefacts.

'Bliamba seed' was originally obtained in this exact setting.
Note, the environment constants are different from those in `bliamba.py`.

"""

import numpy as np
import random

DEATH_SPEED = 15
BIRTH_COST = 0
MAX_GENES = 9
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720
SAVE_FRAMES = False
DOWNSCALE_FACTOR = 1
FRAME_SKIP = 1
RANDOM_SEED = None

def fld_init(a):
    conway = a.str2genome("3/23")
    diamoeba = a.str2genome("35678/5678")
    return np.asarray([[(random.choice([0, 1]) * (diamoeba if (i > 400 and i < 800 and j > 300 and j < 500) else conway)) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
