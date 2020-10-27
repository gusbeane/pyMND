from math import log, sqrt, pow, cos, sin, exp

import numpy as np
from scipy.integrate import romberg
from numba import njit, jit
from scipy.optimize import linprog

@njit(cache=True)
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

@njit(cache=True)
def gc(c, tol=1.48e-08, rtol=1.48e-08):
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
    n = 10
    clist = c * np.arange(0, n+1)/n
    int0 = np.trapz(_gc_int(clist), clist)
    n *= 2
    clist = c * np.arange(0, n+1)/n
    int1 = np.trapz(_gc_int(clist), clist)
    while abs(int1-int0) > tol or abs((int1-int0)/int1) > rtol:
        int0 = int1
        n *= 2
        clist = c * np.arange(0, n+1)/n
        int1 = np.trapz(_gc_int(clist), clist)
    return int1

@njit(cache=True)
def _gc_int(x):
    return np.power(np.log(1 + x) - x / (1 + x), 0.5) * np.power(x, 1.5) / np.power(1 + x, 2)

@njit(cache=True)
def R1_method(N, g = 1.6180339887498948482):
    """
    Returns draws from the R2 method for N particles.
    Parameters
    ----------
    N : int
        Number of draws to make.
    g : float, optional
        Number to increment by, by default uses the optimal low-discrepancy number.
    Returns
    -------
    x1 : `~numpy.ndarray` of shape `(N)`
        Draws along the first dimension.
    x2 : `~numpy.ndarray` of shape `(N)'
        Draws along the second dimension.
    """
    a1 = 1.0/g

    x1 = np.zeros(N)

    x1[0] = 0.5

    for i in range(1, N):
        x1[i] = np.mod(x1[i-1] + a1, 1.0)
    
    return x1

@njit(cache=True)
def R2_method(N, g = 1.32471795724474602596):
    """
    Returns draws from the R2 method for N particles.
    Parameters
    ----------
    N : int
        Number of draws to make.
    g : float, optional
        Number to increment by, by default uses the optimal low-discrepancy number.
    Returns
    -------
    x1 : `~numpy.ndarray` of shape `(N)`
        Draws along the first dimension.
    x2 : `~numpy.ndarray` of shape `(N)'
        Draws along the second dimension.
    """
    a1 = 1.0/g
    a2 = 1.0/(g*g)

    x1 = np.zeros(N)
    x2 = np.zeros(N)

    x1[0], x2[0] = 0.5, 0.5

    for i in range(1, N):
        x1[i] = np.mod(x1[i-1] + a1, 1.0)
        x2[i] = np.mod(x2[i-1] + a2, 1.0)
    
    return x1, x2

@njit(cache=True)
def R3_method(N, g = 1.22074408460575947536):
    """
    Returns draws from the R3 method for N particles.
    Parameters
    ----------
    N : int
        Number of draws to make.
    g : float, optional
        Number to increment by, by default uses the optimal low-discrepancy number.
    Returns
    -------
    x1 : `~numpy.ndarray` of shape `(N)`
        Draws along the first dimension.
    x2 : `~numpy.ndarray` of shape `(N)'
        Draws along the second dimension.
    x3 : `~numpy.ndarray` of shape `(N)'
        Draws along the second dimension.
    """
    a1 = 1.0/g
    a2 = 1.0/(g*g)
    a3 = 1.0/(g*g*g)

    x1 = np.zeros(N)
    x2 = np.zeros(N)
    x3 = np.zeros(N)

    x1[0], x2[0], x3[0] = 0.5, 0.5, 0.5

    for i in range(1, N):
        x1[i] = np.mod(x1[i-1] + a1, 1.0)
        x2[i] = np.mod(x2[i-1] + a2, 1.0)
        x3[i] = np.mod(x3[i-1] + a3, 1.0)
    
    return x1, x2, x3

@njit(cache=True)
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

@njit
def bessi0(x):
    ax = abs(x)
    if ax < 3.75:
        y=x/3.75
        y*=y
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
            +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
    else:
        y=3.75/ax
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
            +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
            +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
            +y*0.392377e-2))))))))
    return ans

@njit
def bessi1(x):
    ax = abs(x)
    if ax < 3.75:
        y=x/3.75
        y*=y
        ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
            +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))))
    else:
        y=3.75/ax
        ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
            -y*0.420059e-2))
        ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
            +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))))
        ans *= (exp(ax)/sqrt(ax))
    if x < 0.0:
        ans = -ans
    return ans

@njit 
def bessj0(x):
    ax = abs(x)
    if ax < 8.0:
        y=x*x
        ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
            +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))))
        ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
            +y*(59272.64853+y*(267.8532712+y*1.0))))
        ans=ans1/ans2
    else:
        z=8.0/ax
        y=z*z
        xx=ax-0.785398164
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
            +y*(-0.2073370639e-5+y*0.2093887211e-6)))
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
            +y*(-0.6911147651e-5+y*(0.7621095161e-6
            -y*0.934935152e-7)))
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2)
    return ans

@njit
def bessj1(x):

    ax = abs(x)
    if ax < 8.0:
        y=x*x
        ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
            +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))))
        ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
            +y*(99447.43394+y*(376.9991397+y*1.0))))
        ans=ans1/ans2
    else:
        z=8.0/ax
        y=z*z
        xx=ax-2.356194491
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
            +y*(0.2457520174e-5+y*(-0.240337019e-6))))
        ans2=0.04687499995+y*(-0.2002690873e-3
            +y*(0.8449199096e-5+y*(-0.88228987e-6
            +y*0.105787412e-6)))
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2)
        if x < 0.0:
             ans = -ans
    return ans

@njit
def bessk0(x):
    if x <= 2.0:
        y=x*x/4.0
        ans=(-log(x/2.0)*bessi0(x))+(-0.57721566+y*(0.42278420
            +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
            +y*(0.10750e-3+y*0.74e-5))))))
    else:
        y=2.0/x
        ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
            +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
            +y*(-0.251540e-2+y*0.53208e-3))))))
    return ans

@njit
def bessk1(x):
    if x <= 2.0:
        y=x*x/4.0
        ans=(log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144
            +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
            +y*(-0.110404e-2+y*(-0.4686e-4)))))))
    else:
        y=2.0/x
        ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
            +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
            +y*(0.325614e-2+y*(-0.68245e-3)))))))
    return ans



