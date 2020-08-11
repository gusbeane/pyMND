import numpy as np
from numba import njit

@njit
def halo_density(pos, M, a):
    """
    The value of the halo density.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the density.
    M : float
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

def halo_potential(pos, M, a, u):
    """
    The value of the halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential.
    M : float
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

def halo_potential_derivative_R(pos, M, a, u):
    """
    The value of the partial derivative in R direction of the halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential derivative.
    M : float
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

def halo_potential_derivative_z(pos, M, a, u):
    """
    The value of the partial derivative in z direction of the halo potential.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Positions at which to compute the value of the potential derivative.
    M : float
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

def halo_draw_r(N, a):
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
    f = np.random.rand(N)
    sqrtf = np.sqrt(f)
    rt = np.divide(sqrtf, np.subtract(1., sqrtf))
    
    return np.multiply(rt, a)

def draw_halo_pos(N, a, u):
    """
    Randomly draw positions from a Hernquist profile.

    Randomly draws 3D values for a Hernquist profile of a given scale length. Poisson
    draws on the distances are first done, and then particles are given random
    orientations.
    Parameters
    ----------
    N : `int`
        Number of distances to draw.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Randomly drawn positions from a Hernquist profile.
    """
    r = halo_draw_r(N, a)
    phi = np.multiply(2.*u.PI, np.random.rand(N))
    theta = np.arccos(np.random.rand(N) * 2. - 1.)

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    xp_halo = np.multiply(np.multiply(r, stheta), np.cos(phi))
    yp_halo = np.multiply(np.multiply(r, stheta), np.sin(phi))
    zp_halo = np.multiply(r, ctheta)
    pos = np.transpose([xp_halo, yp_halo, zp_halo])
    return pos

def compute_velocity_ellipsoid_halo(pos, M, a, u, vcirc_squared, halo_spinfactor):
    """
    Compute the velocity ellipsoid for the halo component from input positions.
    Parameters
    ----------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Number of distances to draw.
    M : float
        Total mass of the dark matter halo.
    a : float
        Scale length of the dark matter halo.
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

    vphi_squared = vRz_squared + (R/rho) * partial_rhosigma + vcirc_squared
    ave_phi = halo_spinfactor * np.sqrt(vcirc_squared)

    sigma_phi = vphi_squared - np.square(ave_phi)
    # unstable as r -> inf
    sigma_phi[sigma_phi < 0] = 0
    sigma_phi = np.sqrt(sigma_phi)

    sigma_R = np.sqrt(vRz_squared)
    sigma_z = np.copy(sigma_R)
    ave_R = np.zeros(len(sigma_R))
    ave_z = np.zeros(len(sigma_z))

    return ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi

def draw_halo_vel(pos, vcsq, N, M, a, lam, u):
    """
    Randomly draw velocities from a Hernquist profile.

    Randomly draws 3D velocities for given positions assuming a Hernquist profile. This
    assumes the velocity distribution function is Gaussian. First the velocity ellipsoid
    is calculated and then draws are made from normal gaussians.
    orientations.
    Parameters
    ----------
    pos : `int`
        Number of distances to draw.
    a : float
        Scale length of the dark matter halo.
    u : `~pyMND.units.pyMND_units`
        pyMND units class.
    Returns
    -------
    pos : `~numpy.ndarray` of shape `(N, 3)`
        Randomly drawn positions from a Hernquist profile.
    """
    ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi = \
            compute_velocity_ellipsoid_halo(pos, M, a, u, vcsq, lam)

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
