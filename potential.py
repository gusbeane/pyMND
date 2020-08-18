import numpy as np
from .halo import _halo_potential, _halo_potential_derivative_R
from .gas_halo import _gas_halo_potential, _gas_halo_potential_derivative_R

def potential(pos, p, u):
    """
    The value of the potential from all components.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    p : `~pyMND.param.pyMND_param`
        pyMND param class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot : `~numpy.ndarray` of shape `(N)`
        The value of the potential at pos.
    """
    pot = _halo_potential(pos, p.M_HALO, p.RH, u)
    pot += _gas_halo_potential(pos, p.M_GASHALO, p.RH, u)
    return pot
    
def potential_derivative_R(pos, p, u):
    """
    The value of the partial derivative of the potential in R direction from all components.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    p : `~pyMND.param.pyMND_param`
        pyMND param class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot_R : `~numpy.ndarray` of shape `(N)`
        The value of the partial derivative of the potential in the R direction at pos.
    """
    pot_R = _halo_potential_derivative_R(pos, p.M_HALO, p.RH, u)
    pot_R += _gas_halo_potential_derivative_R(pos, p.M_GASHALO, p.RH, u)
    return pot_R
