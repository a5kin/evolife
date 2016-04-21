=======
EvoLife
=======

Life-like cellular automaton with evolutionary rules for each cell.

Requirements
============

- Python 2.7
- NumPy / SciPy
- PyCUDA
- PyGame
- scikit-image
- NVidia CUDA Toolkit
- Powerful NVidia GPU is recommended, but should work with any CUDA enabled card

If you're using a Debian-like distro:

``$ sudo apt-get install python-pycuda python-numpy python-scipy python-pygame nvidia-cuda-toolkit python-setuptools``

``$ sudo easy_install scikit-image``

Usage
=====

``$ python evolife.py [experiment_name]``

or just

``$ ./evolife.py [experiment_name]``

If no preset given, default 'big bang' is used.

Controls
--------

- **Arrows**:	move field around
- **+** / **-**:	zoom field in/out
- **]** / **[**:	increase/decrease frame skip
- **F**:	toggle fullscreen
- **S**:	save a field dump to `fields/field.npy` file
- **Q** / **ESC**:	quit

Every 100 steps, top 10 species will be printed to a console. SN is a total number of species currently on the board.

Automaton Rules
===============

- Each living cell has its own birth/sustain ruleset and an energy level;
- Cell is loosing all energy if number of neighbours is not in its sustain rule;
- Cell is born with max energy if there are exactly N neighbours with N in their birth rule;
  - Same is applied for living cells (re-occupation case), if new genome is different;
- If there are several birth situations with different N possible, we choose one with larger N;
- Newly born cell's ruleset calculated as crossover between 'parent' cells rulesets;
- If cell is involved in breeding as a 'parent', it's loosing `BIRTH_COST` units of energy per each non-zero gene passed;
  - This doesn't apply in re-occupation case;
- Every turn, cell is loosing `DEATH_SPEED` units of energy;
- Cell with zero energy is dying;
- Cell cannot have more than `MAX_GENES` non-zero genes in ruleset.

Experiments
===========

You may see a list of experimental presets in `experiments` folder. To run a particular experiment, provide an experiment's filename without `.py` extension. For example to run an experiment described in ``experiments/bliamba.py``, you have to run the following command:
``$ ./evolife.py bliamba``

Most of the provided experiments are set without fixed random seed. Run each of them several times, they could show different behaviours. 

If you are familiar with Python / NumPy, you can easily set up your own experiment. See ``experiments/tutorial.py`` for further instructions.
