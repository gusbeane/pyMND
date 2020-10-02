import numpy as np
from numba import njit

from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline

def compute_velocity_dispersions_halo(force_grid, p, u):
    R, RplusdR, z, = force_grid['R'], force_grid['RplusdR'], force_grid['z']
    z_list = force_grid['z_list']
    Dphi_R, Dphi_z, Dphi_z_dR = force_grid['Dphi_R'], force_grid['Dphi_z'], force_grid['Dphi_z_dR']
    VelVc2, epi_gamma2 = force_grid['VelVc2'], force_grid['epi_gamma2']

    rho = compute_rho_halo(R, z, p, u)

    # Compute velocity dispersion in R/z directions
    # NOTE: We are computing rho * vel dispersion, since this makes computing phi vel dispersion
    # easier. Later, we will divide by rho.
    VelDispRz_halo = cumtrapz(rho * Dphi_z, z_list, initial=0, axis=1)
    VelDispRz_halo = np.transpose((VelDispRz_halo[:,-1] - np.transpose(VelDispRz_halo)))
    VelDispRz_halo[np.isnan(VelDispRz_halo)] = 0.0

    # Now compute derivative of velocity dispersion in R/z direction wrt R
    rho_dR = compute_rho_halo(RplusdR, z, p, u)

    VelDispRz_dR_halo = cumtrapz(rho_dR * Dphi_z_dR, z_list, initial=0, axis=1)
    VelDispRz_dR_halo = np.transpose((VelDispRz_dR_halo[:,-1] - np.transpose(VelDispRz_dR_halo)))
    VelDispRz_dR_halo[np.isnan(VelDispRz_dR_halo)] = 0.0

    dVDispRz_R = (VelDispRz_dR_halo - VelDispRz_halo) / (RplusdR - R)
    dVDispRz_R[0,:] = 0.0

    # Now compute velocity dispersion in phi direction, first just the deriv term
    # Recall that dVDispRz_R is actually the derivative of rho * vel disp
    VelDispPhi_halo = (R / rho) * dVDispRz_R
    VelDispPhi_halo[np.isnan(VelDispPhi_halo)] = 0.0
    VelDispPhi_halo[np.isinf(VelDispPhi_halo)] = 0.0

    # Divide by rho for the vel disp RZ
    VelDispRz_halo /= rho
    VelDispRz_halo[np.isnan(VelDispRz_halo)] = 0.0
    VelDispRz_halo[np.isinf(VelDispRz_halo)] = 0.0

    # Add other terms for velocity dispersion in phi direction
    VelDispPhi_halo += VelDispRz_halo + VelVc2

    # Set streaming velocity and then convert from avg(vphi^2) to sigma(vphi)^2
    VelStreamPhi_halo = p.halo_spinfactor * np.sqrt(VelVc2)

    VelDispPhi_halo = VelDispPhi_halo - np.square(VelStreamPhi_halo)
    VelDispPhi_halo[VelDispPhi_halo < 0.0] = 0.0

    VelDispRz_halo[0,:] = 0.0
    VelDispPhi_halo[0,:] = 0.0

    VelDispRz_halo[np.logical_or(VelDispRz_halo < 0.0, np.isnan(VelDispRz_halo))] = 0.0
    VelDispPhi_halo[np.logical_or(VelDispPhi_halo < 0.0, np.isnan(VelDispPhi_halo))] = 0.0

    # Now put into a nice dict
    force_grid['VelDispRz_halo'] = VelDispRz_halo
    force_grid['VelDispPhi_halo'] = VelDispPhi_halo
    force_grid['VelStreamPhi_halo'] = VelStreamPhi_halo

    return force_grid

def draw_halo_vel(pos, force_grid, p, u):
    # Setup bivariate splines
    VelDispRz_halo_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispRz_halo'])
    VelDispPhi_halo_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispPhi_halo'])
    VelStreamPhi_halo_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelStreamPhi_halo'])

    R = np.linalg.norm(pos[:,:2], axis=1)
    z = np.abs(pos[:,2])

    # Interpolate the correct sigmas
    VelDispR = p.RadialDispersionFactor * VelDispRz_halo_spline(R, z, grid=False)
    VelDispz = VelDispRz_halo_spline(R, z, grid=False)
    VelDispPhi = VelDispPhi_halo_spline(R, z, grid=False)
    VelStreamPhi = VelStreamPhi_halo_spline(R, z, grid=False)

    # Now draw from gaussians
    vR = np.random.normal(size=p.N_HALO)
    vz = np.random.normal(size=p.N_HALO)
    vphi = np.random.normal(size=p.N_HALO)

    vR *= np.sqrt(VelDispR)
    vz *= np.sqrt(VelDispz)
    vphi *= np.sqrt(VelDispPhi)

    vphi += VelStreamPhi

    # Convert to cartesian
    vx = vR * pos[:,0] / R - vphi * pos[:,1] / R
    vy = vR * pos[:,1] / R + vphi * pos[:,0] / R

    return np.transpose([vx, vy, vz])

def compute_rho_halo(R, z, p, u):
    M = p.M_HALO
    a = p.RH

    r = np.sqrt(R*R + z*z)

    ans = (M / (2. * u.PI)) * a / (r * (r + a)**3.) 

    return ans


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