import numpy as np
from scipy.interpolate import RectBivariateSpline

from .potential import circular_velocity_squared
from .util import *

def draw_gas_halo_pos(p):
    """
    Draws positions from a Hernquist profile.

    Cuts off particles at R200.
    Parameters
    ----------
    p : `~pyMND.param.pyMND_param`
        pyMND param class.
    Returns
    -------
    N : `int`
        The actual number of gas cells drawn.
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Evenly drawn positions from a Hernquist profile.
    mass : `~numpy.ndarray` of shape `(N)`
        The masses of each cell.
    """
    # We have a target gas cell mass, in units of M:
    N = p.N_GAS
    M = p.M_GASHALO
    a = p.RH
    Rmax = p.R200

    mg = M/N

    u, v = np.random.rand(2, N)
    phi = 2.*np.pi * u
    theta = np.arccos(2.*v - 1)

    f = np.random.rand(N)
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

def draw_gas_halo_vel(pos, p, u, force_grid):
    """
    Assign gas halo velocities.

    Assigns 3D velocities assuming the gas halo has a fraction of the dark matter halo spin.
    Only the azimuthal velocity is assigned.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to assign gas halo velocities.
    p : `~pyMND.param.pyMND_param`
        pyMND params class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    vel : `~numpy.ndarray` of shape `(N, 3)`
        Velocities drawn from a Hernquist profile.
    """
    halo_spinfactor, GasHaloSpinFraction = p.halo_spinfactor, p.GasHaloSpinFraction

    R = np.linalg.norm(pos[:,:2], axis=1)
    z = np.abs(pos[:,2])

    VCircSq_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelVc2'])
    vcsq = VCircSq_spline(R, z, grid=False)

    vphi = halo_spinfactor * GasHaloSpinFraction * np.sqrt(vcsq)

    vel = np.zeros(np.shape(pos))
    vel[:,0] = - vphi * pos[:,1] / R
    vel[:,1] = vphi * pos[:,0] / R

    return vel