from math import log, sqrt, pow

import numpy as np
from scipy.integrate import romberg
from numba import njit
from scipy.optimize import linprog

def fc(c):
    """
    The value of f(c), given by Equation 7 in Springel+2015.
    Parameters
    ----------
    c : float
        Concentration parameter.
    Returns
    -------
    fc : float
        The value of fc.
    """
    return c * (0.5 - 0.5 / pow(1 + c, 2) - log(1 + c) / (1 + c)) / pow(log(1 + c) - c / (1 + c), 2)
    
def gc(c):
    """
    The value of g(c), used in computing the halo spin factor.
    Parameters
    ----------
    c : float
        Concentration parameter.
    Returns
    -------
    gc : float
        The value of gc.
    """
    return romberg(_gc_int, 0, c)

def _gc_int(x):
    return pow(log(1 + x) - x / (1 + x), 0.5) * pow(x, 1.5) / pow(1 + x, 2)

@njit
def R2_method(N):
    """
    Returns draws from the R2 method for N particles.
    Parameters
    ----------
    N : int
        Number of draws to make.
    Returns
    -------
    x1 : `~numpy.ndarray` of shape `(N)`
        Draws along the first dimension.
    x2 : `~numpy.ndarray` of shape `(N)'
        Draws along the second dimension.
    """
    g = 1.32471795724474602596
    a1 = 1.0/g
    a2 = 1.0/(g*g)

    x1 = np.zeros(N)
    x2 = np.zeros(N)

    x1[0], x2[0] = 0.5, 0.5

    for i in range(1, N):
        x1[i] = np.mod(x1[i-1] + a1, 1.0)
        x2[i] = np.mod(x2[i-1] + a2, 1.0)
    
    return x1, x2

@njit
def gen_3D_grid(t):
    """
    Generates a regular 3D grid from a 1D array.
    Parameters
    ----------
    t : `~numpy.ndarray` of shape `(N)`
        1D array from which to create a 3D grid.
    Returns
    -------
    grid : `~numpy.ndarray` of shape `(N, 3)`
        Output 3D grid.
    """
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

def point_in_hull(point, hull, tolerance=1e-12): 
    """
    Generates a regular 3D grid from a 1D array.
    Parameters
    ----------
    point : `~numpy.ndarray` of shape `(3)`
        A 3-dimensional point.
    hull : `~scipy.spatial.ConvexHull`
        A `scipy` ConvexHull object.
    tolerance : `float`, optional
        Numerical tolerance for equations being less than zero.
    Returns
    -------
    in_hull : `bool`
        Whether or not the point lies within the convex hull.
    """
    return all( 
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance) 
        for eq in hull.equations) 
