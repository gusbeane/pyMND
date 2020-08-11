import numpy as np
from .halo import *
from .util import *
from numba import njit

def gas_halo_temperature(r, M, a, u):
    """
    Compute the equilibrium temperature at a given distance.

    Temperature is computed assuming hydrostatic equilibirum and no rotation for a Hernquist profile. 
    The correction for rotation is usually small (<1%), except for very highly rotating systems. 
    See, e.g., Equation 1 from Kaufmann+2007. Note: the value of u.TFLOOR is NOT applied in this function.
    Parameters
    ----------
    r : `~numpy.ndarray` of shape `(N)`
        Distances at which to compute the halo temperature.
    M : float
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    T : `~numpy.ndarray` of shape `(N)`
        Equilibrium temperature at the given r.
    """
    # TODO: check for if T < 1e4 K, maybe set T floor at 1e4 K?
    meanweight = 4 / (8 - 5 * (1 - u.HYDROGEN_MASSFRAC)) # assume full ionization, in units of proton mass
    mu = meanweight * u.PROTONMASS / u.UnitMass_in_g
    kB = u.BOLTZMANN / u.UnitEnergy_in_cgs
    G = u.G

    x = r / a
    term1 = np.log(1 + 1./x)
    term2 = (25. + 52.*x + 42.*x**2. + 12*x**3.) / (12 * (1+x)**4.)

    T = x * (1+x)**3. * (term1 - term2)

    T *= (mu / kB) * G * M / a
    return T

def gas_halo_thermal_energy(pos, M, a, u):
    """
    Compute the equilibrium thermal energy at given positions.

    See :func:`gas_halo_temperature` for more info on how the temperature is computed.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the halo thermal energy.
    M : float
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    energy : `~numpy.ndarray` of shape `(N)`
        Equilibrium thermal energy at the given positions.
    """
    r = np.linalg.norm(pos, axis=1)
    T = gas_halo_temperature(r, M, a, u)

    T[T < u.TFLOOR] = u.TFLOOR

    meanweight = 4 / (8 - 5 * (1 - u.HYDROGEN_MASSFRAC))
    energy = 1 / meanweight * (1.0 / u.GAMMA_MINUS1) * (u.BOLTZMANN / u.PROTONMASS) * T
    energy *= u.UnitMass_in_g / u.UnitEnergy_in_cgs
    return energy

def gas_halo_potential(pos, M, a, u):
    """
    The value of the gas halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    M : float
        Total mass of the gas halo.
    a : float
        Scale length of the gas halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot : `~numpy.ndarray` of shape `(N)`
        The value of the gas halo potential at pos.
    """
    return halo_potential(pos, M, a, u)

def gas_halo_potential_derivative_R(pos, M, a, u):
    """
    The value of the partial derivative in R direction of the gas halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential derivative.
    M : float
        Total mass of the gas halo.
    a : float
        Scale length of the gas halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pot_R : `~numpy.ndarray` of shape `(N)`
        The value of the partial derivative in R direction of the gas halo potential at pos.
    """
    return halo_potential_derivative_R(pos, M, a, u)

@njit
def gas_halo_density(pos, M, a):
    """
    The value of the gas halo density.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the density.
    M : float
        Total mass of the gas halo.
    a : float
        Scale length of the gas halo.
    Returns
    -------
    rho : `~numpy.ndarray` of shape `(N)`
        The value of the gas halo density at pos.
    """
    return halo_density(pos, M, a)

def draw_gas_halo_pos(N, M, a, Rmax):
    """
    Evenly draw positions from a Hernquist profile.

    Evenly draws 3D values for a Hernquist profile of a given scale length. Distances are
    evenly drawn and then angles are drawn from a 3D golden spiral to evenly space gas
    cells.
    Parameters
    ----------
    N : `int`
        Number of positions to draw.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    M : `int`
        The actual number of gas cells drawn.
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Evenly drawn positions from a Hernquist profile.
    mass : `~numpy.ndarray` of shape `(N)`
        The masses of each cell.
    """
    # We have a target gas cell mass, in units of M:
    mg = M/N

    u, v = R2_method(N)
    phi = 2.*np.pi * u
    theta = np.arccos(2.*v - 1)

    f = np.arange(0.5, N)/N
    sqrtf = np.sqrt(f)
    r = a * sqrtf / (1. - sqrtf)

    pos = np.transpose([r * np.cos(phi) * np.sin(theta),
                                 r * np.sin(phi) * np.sin(theta),
                                 r * np.cos(theta)])
    
    key = np.where(r < Rmax)[0]
    M = len(key)
    pos = pos[key]
    mass = np.full(M, mg)
    
    return M, pos, mass

def draw_gas_halo_vel(pos, vcsq, halo_spinfactor, GasHaloSpinFraction):
    """
    Assign gas halo velocities.

    Assigns 3D velocities assuming the gas halo has a fraction of the dark matter halo spin.
    Only the azimuthal velocity is assigned.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to assign gas halo velocities.
    vcsq : `~numpy.ndarray` of shape `(N)`
        Circular velocity squared at each pos.
    halo_spinfactor : `float`
        Halo spin factor LAMBDA.
    GasHaloSpinFraction : `float`
        The fraction of the dark matter halo spin assigned to the gas halo.
    Returns
    -------
    vel : `~numpy.ndarray` of shape `(N, 3)`
        Velocities drawn from a Hernquist profile.
    """
    vphi = halo_spinfactor * GasHaloSpinFraction * np.sqrt(vcsq)

    R = np.linalg.norm(pos[:, :2], axis=1)

    vel = np.zeros(np.shape(pos))
    vel[:,0] = - vphi * pos[:,1] / R
    vel[:,1] = vphi * pos[:,0] / R

    return vel
