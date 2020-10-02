import numpy as np
from numba import njit

@njit
def _halo_density(pos, M, a):
    """
    The value of the halo density.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the density.
    M : `float`
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    Returns
    -------
    rho : `~numpy.ndarray` of shape `(N)`
        The value of the halo density at pos.
    """
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    rho = (M/(2.*np.pi)) * (a/r) * (r+a)**(-3.)
    return rho

def _halo_potential_derivative_R(pos, M, a, u):
    """
    The value of the partial derivative in R direction of the halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential derivative.
    M : `float`
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot_R : `~numpy.ndarray` of shape `(N)`
        The value of the partial derivative in R direction of the halo potential at pos.
    """
    r = np.linalg.norm(pos, axis=1)
    R = np.linalg.norm(pos[:,:2], axis=1)
    pot_R = u.G * M
    pot_R /= np.square(r + a)
    pot_R *= R/r

    return pot_R

@njit
def _halo_mass_enclosed(r, M, a, u):
    """
    The mass enclosed within a certain radius of a Hernquist halo.
    Parameters
    ----------
    r : `~numpy.ndarray` of shape `(N)` or `float`
        Radii at which to compute the enclosed mass.
    M : `float`
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    Menc : `~numpy.ndarray` of shape `(N)` or `float`
        The enclosed mass at each radii.
    """
    x = r / a
    Menc = M * (x/(1.+x))**2.
    return Menc