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

def _halo_potential(pos, M, a, u):
    """
    The value of the halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    M : `float`
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot : `~numpy.ndarray` of shape `(N)`
        The value of the halo potential at pos.
    """
    r = np.linalg.norm(pos, axis=1)
    pot = - u.G * M
    pot /= r + a
    return pot

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

def _halo_potential_derivative_z(pos, M, a, u):
    """
    The value of the partial derivative in z direction of the halo potential.
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
    pot_z : `~numpy.ndarray` of shape `(N)`
        The value of the partial derivative in z direction of the halo potential at pos.
    """
    r = np.linalg.norm(pos, axis=1)
    z = pos[:,2]
    pot_z = u.G * M
    pot_z /= np.square(r + a)
    pot_z *= z/r

    return pot_z
