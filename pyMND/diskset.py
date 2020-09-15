import numpy as np
from math import isinf, exp, log, cos, sin

from numba import jit, njit

@njit
def fraction_enclosed_disk(R):
    ans = 1 - (1 + R) * exp(-R)
    return ans

@njit
def fraction_enclosed_disk_derivative(R):
    ans = R * exp(-R)
    return ans

@njit
def draw_R_star(q):
    R = 1.0

    itr = 0
    Rold = 0.0

    while(abs(R - Rold)/ R > 1e-7):
        f = fraction_enclosed_disk(R) - q
        f_ = fraction_enclosed_disk_derivative(R)
        Rold = R
        R = R - f/f_
        itr += 1
  
    if(abs(fraction_enclosed_disk(R) - q) > 1e-5 or isinf(R)):
        print("WARNING: R draw of star particle does not seem to be converged")

    return R

@njit
def draw_disk_pos(p, u):
    N = p.N_DISK
    Rq, phiq, zq = np.random.rand(3, N)
    pos = np.zeros((N, 3))

    R = np.zeros(N)
    for i in range(N):
        R[i] = draw_R_star(Rq[i]) * p.H
    
    phi = phiq * 2. * u.PI

    z = np.zeros(N)
    for i in range(N):
        z[i] = (p.Z0 / 2) * log(zq[i] / (1. - zq[i]))
    
    for i in range(N):
        
        pos[i][0] = R[i] * cos(phi[i])
        pos[i][1] = R[i] * sin(phi[i])
        pos[i][2] = z[i]

    return pos

@njit
def draw_dummy_disk_pos(p, u):
    N = p.RMASSBINS * p.ZMASSBINS * p.PHIMASSBINS
    pos = np.zeros((N, 3))

    it = 0
    for i in range(p.RMASSBINS):
        q = (i + 0.5) / p.RMASSBINS
        R = draw_R_star(q) * p.H

        for j in range(p.ZMASSBINS):
            q = (j + 0.5) / p.ZMASSBINS
            z = (p.Z0 / 2) * log(q / (1. - q))

            for k in range(p.PHIMASSBINS):
                phi = 2. * u.PI * (k + 0.5) / p.PHIMASSBINS

                pos[it][0] = R * cos(phi)
                pos[it][1] = R * sin(phi)
                pos[it][2] = z
                it += 1
    
    return pos

    