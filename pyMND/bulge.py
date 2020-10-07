import numpy as np
from numba import njit

from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline

def _bulge_draw_r(N, A):
    """
    Randomly draw distances from a Hernquist profile.

    Randomly draws r values for a Hernquist profile of a given scale length. Poisson
    draws are done.
    Parameters
    ----------
    N : `int`
        Number of distances to draw.
    A : float
        Scale length of the dark matter bulge.
    Returns
    -------
    r : `~numpy.ndarray` of shape `(N)`
        Randomly drawn distances from a Hernquist profile.
    """

    f = np.random.rand(N)
    sqrtf = np.sqrt(f)
    rt = np.divide(sqrtf, np.subtract(1., sqrtf))
    
    return np.multiply(rt, A)

def draw_bulge_pos(p, u):
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
    N = p.N_BULGE
    A = p.A

    r = _bulge_draw_r(N, A)
    phi = np.multiply(2.*u.PI, np.random.rand(N))
    theta = np.arccos(np.random.rand(N) * 2. - 1.)

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    xp_bulge = np.multiply(np.multiply(r, stheta), np.cos(phi))
    yp_bulge = np.multiply(np.multiply(r, stheta), np.sin(phi))
    zp_bulge = np.multiply(r, ctheta)
    pos = np.transpose([xp_bulge, yp_bulge, zp_bulge])
    return pos

def compute_velocity_dispersions_bulge(force_grid, p, u):
    R, RplusdR, z, = force_grid['R'], force_grid['RplusdR'], force_grid['z']
    z_list = force_grid['z_list']
    Dphi_R, Dphi_z, Dphi_z_dR = force_grid['Dphi_R'], force_grid['Dphi_z'], force_grid['Dphi_z_dR']
    VelVc2, epi_gamma2 = force_grid['VelVc2'], force_grid['epi_gamma2']

    rho = compute_rho_bulge(R, z, p, u)

    # Compute velocity dispersion in R/z directions
    # NOTE: We are computing rho * vel dispersion, since this makes computing phi vel dispersion
    # easier. Later, we will divide by rho.
    VelDispRz_bulge = cumtrapz(rho * Dphi_z, z_list, initial=0, axis=1)
    VelDispRz_bulge = np.transpose((VelDispRz_bulge[:,-1] - np.transpose(VelDispRz_bulge)))
    VelDispRz_bulge[np.isnan(VelDispRz_bulge)] = 0.0

    # Now compute derivative of velocity dispersion in R/z direction wrt R
    rho_dR = compute_rho_bulge(RplusdR, z, p, u)

    VelDispRz_dR_bulge = cumtrapz(rho_dR * Dphi_z_dR, z_list, initial=0, axis=1)
    VelDispRz_dR_bulge = np.transpose((VelDispRz_dR_bulge[:,-1] - np.transpose(VelDispRz_dR_bulge)))
    VelDispRz_dR_bulge[np.isnan(VelDispRz_dR_bulge)] = 0.0

    dVDispRz_R = (VelDispRz_dR_bulge - VelDispRz_bulge) / (RplusdR - R)
    dVDispRz_R[0,:] = 0.0

    # Now compute velocity dispersion in phi direction, first just the deriv term
    # Recall that dVDispRz_R is actually the derivative of rho * vel disp
    VelDispPhi_bulge = (R / rho) * dVDispRz_R
    VelDispPhi_bulge[np.isnan(VelDispPhi_bulge)] = 0.0
    VelDispPhi_bulge[np.isinf(VelDispPhi_bulge)] = 0.0

    # Divide by rho for the vel disp RZ
    VelDispRz_bulge /= rho
    VelDispRz_bulge[np.isnan(VelDispRz_bulge)] = 0.0
    VelDispRz_bulge[np.isinf(VelDispRz_bulge)] = 0.0

    # Add other terms for velocity dispersion in phi direction
    VelDispPhi_bulge += VelDispRz_bulge + VelVc2

    # Set streaming velocity and then convert from avg(vphi^2) to sigma(vphi)^2
    VelStreamPhi_bulge = np.zeros(np.shape(VelDispPhi_bulge))

    VelDispPhi_bulge = VelDispPhi_bulge - np.square(VelStreamPhi_bulge)
    VelDispPhi_bulge[VelDispPhi_bulge < 0.0] = 0.0

    VelDispRz_bulge[0,:] = 0.0
    VelDispPhi_bulge[0,:] = 0.0

    VelDispRz_bulge[np.logical_or(VelDispRz_bulge < 0.0, np.isnan(VelDispRz_bulge))] = 0.0
    VelDispPhi_bulge[np.logical_or(VelDispPhi_bulge < 0.0, np.isnan(VelDispPhi_bulge))] = 0.0

    # Now put into a nice dict
    force_grid['VelDispRz_bulge'] = VelDispRz_bulge
    force_grid['VelDispPhi_bulge'] = VelDispPhi_bulge
    force_grid['VelStreamPhi_bulge'] = VelStreamPhi_bulge

    return force_grid

def draw_bulge_vel(pos, force_grid, p, u):
    # Setup bivariate splines
    VelDispRz_bulge_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispRz_bulge'])
    VelDispPhi_bulge_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispPhi_bulge'])
    VelStreamPhi_bulge_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelStreamPhi_bulge'])

    R = np.linalg.norm(pos[:,:2], axis=1)
    z = np.abs(pos[:,2])

    # Interpolate the correct sigmas
    VelDispR = p.RadialDispersionFactor * VelDispRz_bulge_spline(R, z, grid=False)
    VelDispz = VelDispRz_bulge_spline(R, z, grid=False)
    VelDispPhi = VelDispPhi_bulge_spline(R, z, grid=False)
    VelStreamPhi = VelStreamPhi_bulge_spline(R, z, grid=False)

    # Now draw from gaussians
    vR = np.random.normal(size=p.N_BULGE)
    vz = np.random.normal(size=p.N_BULGE)
    vphi = np.random.normal(size=p.N_BULGE)

    # TODO: track down the real issue so this ad hoc step doesnt have to be taken
    VelDispPhi[VelDispPhi < 0.0] = 0.0

    vR *= np.sqrt(VelDispR)
    vz *= np.sqrt(VelDispz)
    vphi *= np.sqrt(VelDispPhi)

    vphi += VelStreamPhi

    # Convert to cartesian
    vx = vR * pos[:,0] / R - vphi * pos[:,1] / R
    vy = vR * pos[:,1] / R + vphi * pos[:,0] / R

    return np.transpose([vx, vy, vz])

def compute_rho_bulge(R, z, p, u):
    M = p.M_BULGE
    a = p.A

    r = np.sqrt(R*R + z*z)

    ans = (M / (2. * u.PI)) * a / (r * (r + a)**3.) 

    return ans