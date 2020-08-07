from math import log, sqrt, pow

import numpy as np
from scipy.integrate import romberg
from numba import njit
from scipy.optimize import linprog

def fc(c):
    return c * (0.5 - 0.5 / pow(1 + c, 2) - log(1 + c) / (1 + c)) / pow(log(1 + c) - c / (1 + c), 2)
    
def gc(c):
    return romberg(gc_int, 0, c)

def gc_int(x):
    return pow(log(1 + x) - x / (1 + x), 0.5) * pow(x, 1.5) / pow(1 + x, 2)

def rejection_sample(fn, maxval, N, xrng=[0, 1], overshoot=2., dtype=np.longdouble, fn_args={}):
	xwidth = xrng[1] - xrng[0]
	xstart = xrng[0]

	sample_list = np.array([])

	N_needed = np.copy(N)
	while(N_needed > 0):
		# print('drawing '+str(int(overshoot*N_needed)) + ' samples...')
		x = np.random.rand(int(overshoot*N_needed))
		y = np.random.rand(int(overshoot*N_needed))
	
		x = np.multiply(x, xwidth)
		x = np.add(x, xstart)
		y = np.multiply(y, maxval)

		fn_eval = fn(x, **fn_args)

		keys = np.where(fn_eval > maxval)[0]
		assert len(keys)==0, "Maxval provided is not big enough, maxval="+str(maxval)+" but got max(fn)="+str(np.max(fn_eval))
		
		keys = np.where(y < fn_eval)[0]
		N_this_iter = len(keys)
		if N_this_iter > 0:
			sample_list = np.concatenate((sample_list, x[keys]))
	
		N_needed -= N_this_iter

	return sample_list[:N]

@njit
def R2_method(N):
    g = 1.32471795724474602596
    a1 = 1.0/g
    a2 = 1.0/(g*g)

    x = np.zeros(N)
    y = np.zeros(N)

    x[0], y[0] = 0.5, 0.5

    for i in range(1, N):
        x[i] = np.mod(x[i-1] + a1, 1.0)
        y[i] = np.mod(y[i-1] + a2, 1.0)
    
    return x, y

@njit
def draw_golden_spiral(N, random_orientation=False):

    pos = np.zeros((N, 3))

    phi = 0
    theta = 0
    golden = (1. + 5.**0.5)/2.
    for i in range(N):
        pos[i][0] = np.cos(phi) * np.sin(theta)
        pos[i][1] = np.sin(phi) * np.sin(theta)
        pos[i][2] = np.cos(theta)

        phi += golden/np.sin(theta)
        theta += golden

    indices = np.arange(0, N) + 0.5

    phi = np.arccos(1. - 2.*indices/N)
    theta = np.pi * (1. + 5.**0.5) * indices

    if random_orientation:
        phi0 = np.multiply(2.*np.pi, np.random.rand())
        theta0 = np.arccos(np.random.rand() * 2. - 1.)
        phi -= phi0
        theta -= theta0

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    pos[:,0] = x
    pos[:,1] = y
    pos[:,2] = z

    return pos

@njit
def gen_3D_grid(t):
    N = len(t)**3
    grid = np.zeros((N, 3))
    ct = 0
    for x in t:
        for y in t:
            for z in t:
                grid[ct][0] = x
                grid[ct][1] = y
                grid[ct][2] = z
                ct += 1
    return grid
