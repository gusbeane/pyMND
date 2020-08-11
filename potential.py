import numpy as np
from .halo import halo_potential, halo_potential_derivative_R
from .gas_halo import gas_halo_potential, gas_halo_potential_derivative_R

def potential(pos, M, a, MG, u):
    """
    The value of the potential from all components.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    M : float
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    MG : float
        Total mass of the gaseous hot halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot : `~numpy.ndarray` of shape `(N)`
        The value of the potential at pos.
    """
    pot = halo_potential(pos, M, a, u)
    pot += gas_halo_potential(pos, MG, a, u)
    return pot
    
def potential_derivative_R(pos, M, a, MG, u):
    """
    The value of the partial derivative of the potential in R direction from all components.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    M : float
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    MG : float
        Total mass of the gaseous hot halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot_R : `~numpy.ndarray` of shape `(N)`
        The value of the partial derivative of the potential in the R direction at pos.
    """
    pot_R = halo_potential_derivative_R(pos, M, a, u)
    pot_R += gas_halo_potential_derivative_R(pos, MG, a, u)
    return pot_R
