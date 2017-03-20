#!/usr/bin/env python

"""
EvoLife Cellular Automaton implementation using CUDA.
Rules are:
- Each living cell has its own birth/sustain ruleset and an energy level;
- Cell is loosing all energy if number of neighbours is not in its sustain rule;
- Cell is born with max energy if there are exactly N neighbours with N in their birth rule;
  - Same is applied for living cells (re-occupation case), but only with different genomes;
- If there are several birth situations with different N possible, we choose one with larger N;
- Newly born cell's ruleset calculated as crossover between 'parent' cells rulesets;
- If cell is involved in breeding as a 'parent', it's loosing `BIRTH_COST` units of energy per each non-zero gene passed;
  - This doesn't apply in re-occupation case;
- Every turn, cell is loosing `DEATH_SPEED` units of energy;
- Cell with zero energy is dying;
- Cell cannot have more than `MAX_GENES` non-zero genes in ruleset.
Additional rule is: board has torus topology.

So, if all cells initially has B3/S23 ruleset, DEATH_SPEED = BIRTH_COST = 0, MAX_GENES >= 3, we have exact Conway rules.
But if there were more than one ruleset initially, evolution may begin.
There are 2^18 possible rulesets, only a small fraction of which have been
studied in any detail. So, who knows what we may discover with evolutionary rules :)

CONTROLS:
Arrows    move field
+/-       zoom in/out
]/[       speed up/down
F         toggle fullscreen
S         dump board state to a file
Q/ESC     quit

Prerequisites: pycuda, numpy, scipy, pygame, scikit-image
Debian: apt-get install python-pycuda python-numpy python-pygame python-scipy python-setuptools

Author: a5kin
Copyright: MIT License.

"""

import sys, time, math, colorsys, random, traceback
import pygame
from pygame.locals import *
import numpy as np
from scipy.misc import imsave
import scipy.ndimage.interpolation
from skimage import transform as tf
import importlib

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

try:
    expmod = importlib.import_module('experiments2.' + sys.argv[1])
    DEATH_SPEED = expmod.DEATH_SPEED
    BIRTH_COST = expmod.BIRTH_COST
    MAX_GENES = expmod.MAX_GENES
    FIELD_WIDTH = expmod.FIELD_WIDTH
    FIELD_HEIGHT = expmod.FIELD_HEIGHT
    SAVE_FRAMES = expmod.SAVE_FRAMES
    DOWNSCALE_FACTOR = expmod.DOWNSCALE_FACTOR
    FRAME_SKIP = expmod.FRAME_SKIP
    RANDOM_SEED = expmod.RANDOM_SEED
    FADE_IN = expmod.FADE_IN
    FADE_OUT = expmod.FADE_OUT
    fld_init = expmod.fld_init
except (ImportError, IndexError):
    print "No experiment preset found, loading default (big_bang)."
    DEATH_SPEED = 0
    BIRTH_COST = 0
    MAX_GENES = 9
    FIELD_WIDTH = 1280
    FIELD_HEIGHT = 720
    SAVE_FRAMES = False
    DOWNSCALE_FACTOR = 1
    FRAME_SKIP = 1
    RANDOM_SEED = None
    FADE_IN = 6
    FADE_OUT = 6
    def fld_init(a):
        return np.asarray([[(random.choice([0, 1]) * random.randint(0, 256*512) if (i < 100 and j < 100) else 0) for j in range(a.height)] for i in range(a.width)]).astype(np.int32)
except:
    print traceback.format_exc()
    sys.exit(0)
    

step_gpu = ElementwiseKernel("unsigned int *fld, unsigned int *fld_new, unsigned int *seeds, unsigned int *bufs, unsigned int *img, int w, int h", """
    int x = i / h;
    int y = i % h;
    // torus topology emulation
    int xm1 = x - 1; if (xm1 < 0) xm1 = w + xm1;
    int xp1 = x + 1; if (xp1 >= w) xp1 = xp1 - w;
    int ym1 = y - 1; if (ym1 < 0) ym1 = h + ym1;
    int yp1 = y + 1; if (yp1 >= h) yp1 = yp1 - h;
    // cache neighbours values
    uint f0 = fld[i];
    uint f1 = fld[xm1 * h + ym1];
    uint f2 = fld[x * h + ym1];
    uint f3 = fld[xp1 * h + ym1];
    uint f4 = fld[xm1 * h + y];
    uint f5 = fld[xp1 * h + y];
    uint f6 = fld[xm1 * h + yp1];
    uint f7 = fld[x * h + yp1];
    uint f8 = fld[xp1 * h + yp1];
    uint energy = (f0 >> 17);
    // total number of neighbours
    int N = EXISTS(f1) + EXISTS(f2) + EXISTS(f3) + EXISTS(f4) +
            EXISTS(f5) + EXISTS(f6) + EXISTS(f7) + EXISTS(f8);
    if (energy >= 0xff || N == 0 || f0 > 0 && (((f0 >> 8) & (1 << N)) == 0)) {
        // cell is dying
        fld_new[i] = 0;
        //img[i] = fadeout(img0, 5);
    } else {
        uint f00 = f0;
        for (int ni = 8; ni > 0; ni--) {
            // no re-occupation rule, breeding in empty cells only
            //if (f0 > 0) break;
            // cache neighbours breeding fitnesses 
            int ff1 = FIT(f1, ni);
            int ff2 = FIT(f2, ni);
            int ff3 = FIT(f3, ni);
            int ff4 = FIT(f4, ni);
            int ff5 = FIT(f5, ni);
            int ff6 = FIT(f6, ni);
            int ff7 = FIT(f7, ni);
            int ff8 = FIT(f8, ni);
            if (ff1 + ff2 + ff3 + ff4 + ff5 + ff6 + ff7 + ff8 == ni) {
                // neighbours able to breed, cell is born
                f0 = 0;
                uint gene_num = 0;
                //int genes_count = {2};
                //int gene;
                uint nit = (int) (ni / 2);
                uint seed = seeds[i];
                uint nonzero_genes_num = 0;
                while (gene_num < 17) {
                    // pseudorandom cross breeding
                    uint rng = ((((seed + gene_num) * 58321) + 11113)) % 65535;
                    uint fg1 = (f1 >> gene_num) & ff1;
                    uint fg2 = (f2 >> gene_num) & ff2;
                    uint fg3 = (f3 >> gene_num) & ff3;
                    uint fg4 = (f4 >> gene_num) & ff4;
                    uint fg5 = (f5 >> gene_num) & ff5;
                    uint fg6 = (f6 >> gene_num) & ff6;
                    uint fg7 = (f7 >> gene_num) & ff7;
                    uint fg8 = (f8 >> gene_num) & ff8;
                    int n1 = fg1 + fg2 + fg3 + fg4 + fg5 + fg6 + fg7 + fg8;

                    //if ((int) (n1 * 65535 / ni) < 65535 && (int) (n1 * 65535 / ni) > 0)
                    //    printf("%d %d | ", rng, (int) (n1 * 65535 / ni));
                    //if (n1 > nit) {
                    if ((int) (n1 * 65535 / ni) > rng) {
                        f0 += 1 << gene_num;
                        nonzero_genes_num += 1;
                        if ({1}) {
                            if (fg1) atomicAdd(&bufs[xm1 * h + ym1], ({1} << 17));
                            if (fg2) atomicAdd(&bufs[x * h + ym1], ({1} << 17));
                            if (fg3) atomicAdd(&bufs[xp1 * h + ym1], ({1} << 17));
                            if (fg4) atomicAdd(&bufs[xm1 * h + y], ({1} << 17));
                            if (fg5) atomicAdd(&bufs[xp1 * h + y], ({1} << 17));
                            if (fg6) atomicAdd(&bufs[xm1 * h + yp1], ({1} << 17));
                            if (fg7) atomicAdd(&bufs[x * h + yp1], ({1} << 17));
                            if (fg8) atomicAdd(&bufs[xp1 * h + yp1], ({1} << 17));
                        }
                    }
                    gene_num++;
                }
                if (nonzero_genes_num > {2}) f0 = 0;
                seeds[i] = (((seed * 58321) + 11113)) % 65535;
                //if (f0 != 3076 && f0 != 31820) printf("%d ", f0);
                break;
            }
        }
        if ((f00 & 0x1ffff) == (f0 & 0x1ffff)) {
            f0 = f00;
            if (f0 != 0) {
                f0 += ({0} << 17);
            }
        }
        fld_new[i] = f0;

    }
""".replace("{0}", str(DEATH_SPEED)).replace("{1}", str(BIRTH_COST)).replace("{2}", str(MAX_GENES)), "ca_step", preamble="""
#include <stdio.h>
#define EXISTS(x) (x > 0 ? 1 : 0)
//#define FIT(x, n) ((n == 0 || (x & (1 << (n - 1))) == 0) ? 0 : 1)
#define FIT(x, n) ((x >> (n - 1)) & 1)

__device__ uint fadeout(int val, int step) {
    uint red   = (val & 0x00ff0000) >> 16;
    if (red > step-1) red -= step; else red = 0;
    uint green = (val & 0x0000ff00) >> 8;
    if (green > step-1) green -= step; else green = 0;
    uint blue  = (val & 0x000000ff);
    if (blue > step-1) blue -= step; else blue = 0;
    return blue + (green << 8) + (red << 16);
}

""")

flush_bufs_gpu = ElementwiseKernel("unsigned int *fld_new, unsigned int *bufs, unsigned int *img, int w, int h", """
    uint f0 = fld_new[i];
    f0 += bufs[i];
    uint energy = (f0 >> 17);
    if (energy > 0xff) {
        energy = 0xff;
        f0 = 0;
    }
    fld_new[i] = f0;
    bufs[i] = 0;
    uint img0 = img[i];
    uint tc = hsv2rgb((f0 & 0x1ffff) % 360, 0xff - energy, 255);
    if (f0 == 0) tc = 0;
    int tr = (tc >> 16) & 0xff;
    int tg = (tc >> 8) & 0xff;
    int tb = tc & 0xff;
    int cr = (img0 >> 16) & 0xff;
    int cg = (img0 >> 8) & 0xff;
    int cb = img0 & 0xff;
    cr = max(min(tr, cr + FADE_IN), cr - FADE_OUT);
    cg = max(min(tg, cg + FADE_IN), cg - FADE_OUT);
    cb = max(min(tb, cb + FADE_IN), cb - FADE_OUT);
    img[i] = ((uint) cr << 16) + ((uint) cg << 8) + (uint) cb;
""", "ca_flush", preamble="""
#include <stdio.h>

#define FADE_IN {fade_in}
#define FADE_OUT {fade_out}

__device__ uint hsv2rgb(int hue, int sat, int val) {
	float r, g, b;
	float h, s, v;
	
	h = hue;
	s = fmin(255, (float) sat);
        s /= 255;
	v = fmin(255, (float) val);
	
	float f = ((float) h) / 60.0f;
	float hi = floorf(f);
	f = f - hi;
	int p = (int) (v * (1 - s));
	int q = (int) (v * (1 - s * f));
	int t = (int) (v * (1 - s * (1 - f)));
	
	if(hi == 0.0f || hi == 6.0f) {
		r = v; g = t; b = p;
	} else if (hi == 1.0f) {
		r = q; g = v; b = p;
	} else if (hi == 2.0f) {
		r = p; g = v; b = t;
	} else if (hi == 3.0f) {
		r = p; g = q; b = v;
	} else if (hi == 4.0f) {
		r = t; g = p; b = v;
	} else {
		r = v; g = p; b = q;
	}
	
	unsigned int color = b + g * 256 + r * 256 * 256;
	return color;
}
""".replace("{fade_in}", str(FADE_IN)).replace("{fade_out}", str(FADE_OUT)))


class EvoLife:

    def __init__(self, width=0, height=0, fullscreen=False, saveframes=False, downscale_factor=1, frame_skip=1):
        print "Initializing PyGame...",
        pygame.init()
        self.title = 'EvoLife Cellular Automaton /w CUDA'
        self.saveframes = saveframes
        self.downscale_factor = downscale_factor
        self.movie_frame = 0
        pygame.display.set_caption(self.title, 'CUDA Life')
        modes = pygame.display.list_modes()
        modes.sort()
        modes.reverse()
        self.width = width if width else modes[0][0]
        self.height = height if height else modes[0][1]
        self.frame_skip = frame_skip
        print "done."
        print "Initializing GPU stuff...",
        if RANDOM_SEED:
            random.seed(RANDOM_SEED)
        seeds = np.asarray([[random.randint(1, 50000) for j in range(self.height)] for i in range(self.width)]).astype(np.int32)
        bufs = np.zeros((self.width, self.height), dtype=np.int32)
        fld = fld_init(self)
        self.f1_gpu = gpuarray.to_gpu(fld)
        self.f2_gpu = gpuarray.to_gpu(fld.copy())
        self.seeds_gpu = gpuarray.to_gpu(seeds)
        self.bufs_gpu = gpuarray.to_gpu(bufs)
        self.img_gpu = gpuarray.to_gpu(np.asarray([[0 for v in row] for row in fld]).astype(np.int32))
        print "done."
        print "Initializing display...",
        self.srf = pygame.display.set_mode((self.width / self.downscale_factor, self.height / self.downscale_factor))
        if fullscreen:
            pygame.display.toggle_fullscreen()
        print "done: %sx%s." % (self.width / self.downscale_factor, self.height / self.downscale_factor)
        self.t = 0
        self.zoom = 1
        self.dx = 0
        self.dy = 0
        self.last_checked = time.time()
        self.last_t = 0

    def genome2str(self, g):
        f = ""
        for i in xrange(8):
            if ((1 << i) & g) != 0:
                f += str(i+1)
        f += "/"
        g = g >> 8
        for i in xrange(9):
            if ((1 << i) & g) != 0:
                f += str(i)
        return f

    def str2genome(self, s):
        g = 0
        b, s = s.split("/")
        for i in b:
            g += (1 << (int(i)-1))
        for i in s:
            g += (1 << (int(i)+8))
        return g

    def species_chart(self):
        world = self.f1_gpu.get()
        species = np.unique(world & 0x1ffff, return_counts=True)
        species = zip(species[1][1:], species[0][1:])
        species.sort()
        species.reverse()
        print "SN=%s |" % len(species),
        for s in species[:10]:
            print "%s (%s) |" % (self.genome2str(s[1]), s[0]),
        print

        
    def step(self):
        start_time = time.time()
        step_gpu(self.f1_gpu, self.f2_gpu, self.seeds_gpu, self.bufs_gpu, self.img_gpu, np.uint32(self.width), np.uint32(self.height))
        flush_bufs_gpu(self.f2_gpu, self.bufs_gpu, self.img_gpu, np.uint32(self.width), np.uint32(self.height))
        tmp = self.f1_gpu
        self.f1_gpu = self.f2_gpu
        self.f2_gpu = tmp
        self.t += 1
        self.last_t += 1
        if self.t % self.frame_skip == 0:
            dest = self.img_gpu.get()
            dest = np.reshape(dest, (self.width, self.height), order='F')
            if self.dx:
                dest = np.roll(dest, self.dx, axis=1)
            if self.dy:
                dest = np.roll(dest, self.dy, axis=0)
            if self.zoom > 1:
                dest = dest[:self.width // self.zoom + 1, :self.height // self.zoom + 1]
                dest = dest.repeat(self.zoom, axis=0).repeat(self.zoom, axis=1)
                dest = dest[:self.width, :self.height]
            if self.downscale_factor != 1:
                dest = dest.view(np.uint8).reshape(dest.shape+(4,))[..., :3]
                dest = (tf.resize(dest, (self.width / self.downscale_factor, self.height / self.downscale_factor, 3), order=1) * 255).astype(np.int32)
                tmp = dest[:,:,0].copy()
                dest[:,:,0] = dest[:,:,2]
                dest[:,:,2] = tmp
            if self.saveframes:
                pygame.image.save(self.srf, "movie/frame%s.png" % str(self.movie_frame).zfill(8))
                self.movie_frame += 1
            pygame.surfarray.blit_array(self.srf, dest)
            pygame.display.update()
        if self.t % 100 == 0:
            self.species_chart()
        end_time = time.time()
        if end_time - self.last_checked > 1:
            elapsed_time = end_time - self.last_checked
            pygame.display.set_caption(self.title + " | Step %s: %.2f steps/s @%sx" % (self.t, float(self.last_t) / elapsed_time, self.frame_skip), 'CUDA EvoLife')
            self.last_checked = time.time()
            self.last_t = 0

    def run(self):
        while True:
            self.step()
            events = pygame.event.get()
            need_exit = False
            for e in events:
                if e.type==QUIT or e.type==KEYDOWN and e.key==K_ESCAPE or e.type==KEYDOWN and e.key==K_q:
                    need_exit = True
                    break
                if e.type==KEYDOWN:
                    if e.key==K_KP_PLUS or e.key==K_EQUALS:
                        self.zoom *= 2
                    if e.key==K_MINUS or e.key==K_KP_MINUS:
                        self.zoom = max(1, self.zoom / 2)
                    if e.key==K_RIGHTBRACKET:
                        self.frame_skip += 5
                    if e.key==K_LEFTBRACKET:
                        self.frame_skip = max(1, self.frame_skip - 5)
                    if e.key==K_UP:
                        self.dx += 10
                    if e.key==K_DOWN:
                        self.dx -= 10
                    if e.key==K_LEFT:
                        self.dy += 10
                    if e.key==K_RIGHT:
                        self.dy -= 10
                    if e.key==K_f:
                        pygame.display.toggle_fullscreen()
                    if e.key==K_s:
                        np.save("fields/field.npy", self.f1_gpu.get())
            if need_exit:
                break

if __name__ == '__main__':
    ca = EvoLife(FIELD_WIDTH, FIELD_HEIGHT, saveframes=SAVE_FRAMES, downscale_factor=DOWNSCALE_FACTOR, frame_skip=FRAME_SKIP)
    ca.run()
    
