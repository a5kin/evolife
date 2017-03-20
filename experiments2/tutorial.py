"""
This tutorial will explain how to set your own experiment for EvoLife.
Python / NumPy experience is mandatory.

"""

import numpy as np
import random

# 1. Declare main CA constants.

# 1a. DEATH_SPEED is how much energy each living cell is loosing with each step.
#     Each cell have 255 energy units on birth, so:
#     0 - no aging at all
#     1 - cell will die after 255 steps
#     2 - cell will die after 127 steps
#     4 - cell will die after 63 steps
#     ...and so on
DEATH_SPEED = 0 # Here, we're experimenting with original rules, so no aging

# 1b. BIRTH_COST is how much energy cell is loosing per each non-zero gene passed to its sibling
#     In example, if 3 cells with genome "34567/3456" are giving birth to a new cell,
#     each of them will pass 3 non-zero genes, so each will loose 3 * BIRTH_COST units of energy.
BIRTH_COST = 0 # Here, we're experimenting with original rules, so birth is free

# 1c. MAX_GENES is how much non-zero genes could be in a cell's genome.
#     In example, with MAX_GENES = 3, genome "34/34" could not exist, but "3/23" is OK.
MAX_GENES = 9 # Well, anyway it cannot be more than 9 in our experiment

# 2. Declare system settings.

# Field size - you got it. Set them both to None to init with your native screen size.
FIELD_WIDTH = 1280
FIELD_HEIGHT = 720

# Do we need to save each frame to .png file?
# Set it to True if you wanna shoot a movie.
# All your frames will be in a `./movie` folder.
SAVE_FRAMES = False

# Set it to more than 1 if your field is too large to fit on a screen.
# In example if your field is 12800x7200, you may set downscale to 10,
# and enjoy the process in 1280x720 window. Movie frames will be downscaled too.
DOWNSCALE_FACTOR = 1

# To make your simulation more deterministic, you can fix the initial random seed to certain value.
# Set it to None if you want more unpredictable behaviour.
RANDOM_SEED = 4123 # glider duel
# TODO: roll other spectacular seeds
#RANDOM_SEED = None

# A maximum amount by which a pixel may change its RBG value in 1 step.
# 255 - instant color change,
# 1 - slowest and smoothest color change.
FADE_IN = 255 # this one used when next value is greater than previous
FADE_OUT = 6 # this one used when next value is lesser than previous

# Finally, you may set initial frame skip rate. This is useful if you need to speedup a simulation,
# or if you are shooting a movie and you need to make it N times faster.
# 1 - every frame is displayed
# 2 - every 2nd frame is displayed
# 16 - every 16th frame is displayed
# ...and so on
# Note, you may also change frame skip rate interactively with "]" and "[" keys.
# This constant is just an initial value.
FRAME_SKIP = 1

# 3. Write a function with a seed for your experiment.

# This function always should be named `fld_init`.
def fld_init(a):
    """
    Function take an automaton object as argument,
    and return initialized numpy ndarray of type np.int32

    """
    # Let start with creating an empty board.
    # You can get board's width and height directly from an object, like `a.width`
    fld = np.zeros((a.width, a.height), dtype=np.int32)
    # Let say, you wanna see R-pentomino explosion in Conway rules.
    # So first, let get a single cell's genome, encoded with `str2genome` method
    conway = a.str2genome("3/23")
    # Now, place some cells with Conway genome on a board, somewhere in the middle-left
    cx, cy = 360, 360
    fld[0+cx,1+cy] = conway
    fld[0+cx,2+cy] = conway
    fld[0+cx,3+cy] = conway
    fld[1+cx,0+cy] = conway
    fld[2+cx,0+cy] = conway
    # Ok, placing a bomb in 34 Life rule to the right, hopefully some of the gliders will detonate it.
    life34 = a.str2genome("34/34")
    cx, cy = 903, 360
    fld[2+cx,0+cy] = life34
    fld[0+cx,1+cy] = life34
    fld[1+cx,1+cy] = life34
    fld[1+cx,2+cy] = life34
    fld[1+cx,3+cy] = life34
    fld[2+cx,2+cy] = life34
    fld[3+cx,2+cy] = life34
    # And to spice it up, some Pseudo Life on periphery
    pseudolife = a.str2genome("357/238")
    c = 150
    for i in xrange(500):
        r = (((i * 58321) + 11113))
        fld[r % 1280, r % 3 + c] = pseudolife
        fld[r % 3 + c, r % 720] = pseudolife
    # Dont't forget to return our board
    return fld

# That's all, now you can create a copy of this file in the same folder, and roll your own masterpiece.
# Feel free to share your findings, pull requests on experiments are welcome.


