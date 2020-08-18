
import numpy as np

from .potential import circular_velocity_squared

def _halo_draw_r(N, p):
    """
    Randomly draw distances from a Hernquist profile.

    Randomly draws r values for a Hernquist profile of a given scale length. Poisson
    draws are done.
    Parameters
    ----------
    N : `int`
        Number of distances to draw.
    a : float
        Scale length of the dark matter halo.
    Returns
    -------
    r : `~numpy.ndarray` of shape `(N)`
        Randomly drawn distances from a Hernquist profile.
    """
    a = p.RH

    f = np.random.rand(N)
    sqrtf = np.sqrt(f)
    rt = np.divide(sqrtf, np.subtract(1., sqrtf))
    
    return np.multiply(rt, a)

def draw_halo_pos(p, u):
    """
    Randomly draw positions from a Hernquist profile.

    Randomly draws 3D values for a Hernquist profile of a given scale length. Poisson
    draws on the distances are first done, and then particles are given random
    orientations.
    Parameters
    ----------
    p : `~pyMND.param.pyMND_param`
        pyMND params class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Randomly drawn positions from a Hernquist profile.
    """
    N = p.N_HALO
    a = p.RH

    r = _halo_draw_r(N, p)
    phi = np.multiply(2.*u.PI, np.random.rand(N))
    theta = np.arccos(np.random.rand(N) * 2. - 1.)

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    xp_halo = np.multiply(np.multiply(r, stheta), np.cos(phi))
    yp_halo = np.multiply(np.multiply(r, stheta), np.sin(phi))
    zp_halo = np.multiply(r, ctheta)
    pos = np.transpose([xp_halo, yp_halo, zp_halo])
    return pos

def compute_velocity_ellipsoid_halo(pos, p, u):
    """
    Compute the velocity ellipsoid for the halo component from input positions.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Number of distances to draw.
    p : `~pyMND.param.pyMND_param`
        pyMND param class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    vcirc_squared : `~numpy.ndarray` of shape `(N)`
        Circular velocity squared of each position.
    halo_spinfactor : float
        Halo spin factor LAMBDA.
    Returns
    -------
    ave_R : `~numpy.ndarray` of shape `(N)`
        Average velocity in the R direction.
    ave_z : `~numpy.ndarray` of shape `(N)`
        Average velocity in the z direction.
    ave_phi : `~numpy.ndarray` of shape `(N)`
        Average velocity in the phi direction.
    sigma_R : `~numpy.ndarray` of shape `(N)`
        Velocity standard deviation in the R direction.
    sigma_z : `~numpy.ndarray` of shape `(N)`
        Velocity standard deviation in the z direction.
    sigma_phi : `~numpy.ndarray` of shape `(N)`
        Velocity standard deviation in the phi direction.
    """
    M = p.M_HALO + p.M_GASHALO
    a = p.RH 
    halo_spinfactor = p.halo_spinfactor

    vcsq = circular_velocity_squared(pos, p, u)

    prefactor = u.G * M / a
    r = np.linalg.norm(pos, axis=1)

    # compute < vR^2>, < vz^2>
    prefactor *= r * np.power(r+a, 3.) / a**4.

    rplusa = r + a

    term1 = np.log((rplusa)/r)
    term2 = 3*a**4 + 4*a**3*rplusa + 6*a**2*rplusa**2 + 12*a*rplusa**3
    term2 /= 12 * rplusa**4

    vRz_squared = prefactor * (term1 - term2)

    # as r -> inf, this formula becomes numerically unstable, so fix
    vRz_squared[vRz_squared < 0] = 0

    # compute < vphi^2>
    R = np.linalg.norm(pos[:,:2], axis=1)
    prefactor = u.G * M**2 / (2 * u.PI * a**4)

    term1 = - a * R / (r**2 * rplusa)
    term2 = (R / r) * a * np.power(rplusa, -5)
    term2 *= a**3 + a**2 * rplusa + a * rplusa**2 + rplusa**3

    partial_rhosigma = prefactor * (term1 + term2)

    rho = (M / (2*u.PI)) * (a/r) * np.power(rplusa, -3.)

    vphi_squared = vRz_squared + (R/rho) * partial_rhosigma + vcsq
    ave_phi = halo_spinfactor * np.sqrt(vcsq)

    sigma_phi = vphi_squared - np.square(ave_phi)
    # unstable as r -> inf
    sigma_phi[sigma_phi < 0] = 0
    sigma_phi = np.sqrt(sigma_phi)

    sigma_R = np.sqrt(vRz_squared)
    sigma_z = np.copy(sigma_R)
    ave_R = np.zeros(len(sigma_R))
    ave_z = np.zeros(len(sigma_z))

    return ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi

def draw_halo_vel(pos, p, u):
    """
    Randomly draw velocities from a Hernquist profile.

    Randomly draws 3D velocities for given positions assuming a Hernquist profile. This
    assumes the velocity distribution function is Gaussian. First the velocity ellipsoid
    is calculated and then draws are made from normal gaussians.
    orientations.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to draw velocities.
    p : `~pyMND.param.pyMND_param`
        pyMND params class.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Randomly drawn positions from a Hernquist profile.
    """
    N = p.N_HALO

    ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi = \
            compute_velocity_ellipsoid_halo(pos, p, u)

    vR = np.random.normal(size=N)
    vz = np.random.normal(size=N)
    vphi = np.random.normal(size=N)

    vR *= sigma_R
    vz *= sigma_z
    vphi *= sigma_phi

    vR += ave_R
    vz += ave_z
    vphi += ave_phi

    R = np.linalg.norm(pos[:, :2], axis=1)

    vx = vR * pos[:,0] / R - vphi * pos[:,1] / R
    vy = vR * pos[:,1] / R + vphi * pos[:,0] / R

    return np.transpose([vx, vy, vz])