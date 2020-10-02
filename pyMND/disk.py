import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline

def compute_velocity_dispersions_disk(force_grid, p, u):
    R, RplusdR, z, = force_grid['R'], force_grid['RplusdR'], force_grid['z']
    z_list = force_grid['z_list']
    Dphi_R, Dphi_z, Dphi_z_dR = force_grid['Dphi_R'], force_grid['Dphi_z'], force_grid['Dphi_z_dR']
    VelVc2, epi_gamma2 = force_grid['VelVc2'], force_grid['epi_gamma2']

    RadialDispersionFactor = p.RadialDispersionFactor
    
    rho = compute_rho_disk(R, z, p, u)

    # Compute velocity dispersion in R/z directions
    # NOTE: We are computing rho * vel dispersion, since this makes computing phi vel dispersion
    # easier. Later, we will divide by rho.
    VelDispRz_disk = cumtrapz(rho * Dphi_z, z_list, initial=0, axis=1)
    VelDispRz_disk = np.transpose((VelDispRz_disk[:,-1] - np.transpose(VelDispRz_disk)))
    VelDispRz_disk[np.isnan(VelDispRz_disk)] = 0.0

    # Now compute derivative of velocity dispersion in R/z direction wrt R
    rho_dR = compute_rho_disk(RplusdR, z, p, u)

    VelDispRz_dR_disk = cumtrapz(rho_dR * Dphi_z_dR, z_list, initial=0, axis=1)
    VelDispRz_dR_disk = np.transpose((VelDispRz_dR_disk[:,-1] - np.transpose(VelDispRz_dR_disk)))
    VelDispRz_dR_disk[np.isnan(VelDispRz_dR_disk)] = 0.0

    dVDispRz_R = (VelDispRz_dR_disk - VelDispRz_disk) / (RplusdR - R)
    dVDispRz_R[0,:] = 0.0

    # Now compute velocity dispersion in phi direction, first just the deriv term
    # Recall that dVDispRz_R is actually the derivative of rho * vel disp
    VelDispPhi_disk = RadialDispersionFactor * (R / rho) * dVDispRz_R
    VelDispPhi_disk[np.isnan(VelDispPhi_disk)] = 0.0
    VelDispPhi_disk[np.isinf(VelDispPhi_disk)] = 0.0

    # Divide by rho for the vel disp RZ
    VelDispRz_disk /= rho
    VelDispRz_disk[np.isnan(VelDispRz_disk)] = 0.0
    VelDispRz_disk[np.isinf(VelDispRz_disk)] = 0.0

    # Add other terms for velocity dispersion in phi direction
    VelDispPhi_disk += VelDispRz_disk + VelVc2

    # Set streaming velocity and then convert from avg(vphi^2) to sigma(vphi)^2
    VelStreamPhi_disk = VelDispPhi_disk - RadialDispersionFactor * VelDispRz_disk/epi_gamma2
    VelStreamPhi_disk[VelStreamPhi_disk < 0] = 0.0
    VelStreamPhi_disk = np.sqrt(VelStreamPhi_disk)

    VelDispPhi_disk = RadialDispersionFactor * VelDispRz_disk
    VelDispPhi_disk = np.transpose(np.transpose(VelDispPhi_disk)/epi_gamma2)

    VelDispRz_disk[0,:] = 0.0
    VelDispPhi_disk[0,:] = 0.0

    VelDispRz_disk[np.logical_or(VelDispRz_disk < 0.0, np.isnan(VelDispRz_disk))] = 0.0
    VelDispPhi_disk[np.logical_or(VelDispPhi_disk < 0.0, np.isnan(VelDispPhi_disk))] = 0.0

    # Now put into a nice dict
    force_grid['VelDispRz_disk'] = VelDispRz_disk
    force_grid['VelDispPhi_disk'] = VelDispPhi_disk
    force_grid['VelStreamPhi_disk'] = VelStreamPhi_disk

    return force_grid

def draw_disk_vel(pos, force_grid, p, u):
    VelDispRz_disk_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispRz_disk'])
    VelDispPhi_disk_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelDispPhi_disk'])
    VelStreamPhi_disk_spline = RectBivariateSpline(force_grid['R_list'], force_grid['z_list'], force_grid['VelStreamPhi_disk'])

    R = np.linalg.norm(pos[:,:2], axis=1)
    z = np.abs(pos[:,2])

    print(np.shape(R), np.shape(z))
    print(np.max(R))
    print(np.max(force_grid['R_list']))

    VelDispR = p.RadialDispersionFactor * VelDispRz_disk_spline(R, z, grid=False)
    VelDispz = VelDispRz_disk_spline(R, z, grid=False)
    VelDispPhi = VelDispPhi_disk_spline(R, z, grid=False)
    VelStreamPhi = VelStreamPhi_disk_spline(R, z, grid=False)

    vR = np.random.normal(size=p.N_DISK)
    vz = np.random.normal(size=p.N_DISK)
    vphi = np.random.normal(size=p.N_DISK)

    print('R', VelDispR[:10])
    print('z', VelDispz[:10])
    print('phi disp', VelDispPhi[:10])
    print('phi steam', VelStreamPhi[:10])

    vR *= np.sqrt(VelDispR)
    vz *= np.sqrt(VelDispz)
    vphi *= np.sqrt(VelDispPhi)

    vphi += VelStreamPhi

    vx = vR * pos[:,0] / R - vphi * pos[:,1] / R
    vy = vR * pos[:,1] / R + vphi * pos[:,0] / R

    return np.transpose([vx, vy, vz])

def compute_rho_disk(R, z, p, u):
    M = p.M_DISK
    H = p.H
    Z0 = p.Z0
    
    ans = M / (4. * u.PI * H * H * Z0) * np.exp(-R / H) * np.power(2 / (np.exp(z / Z0) + np.exp(-z / Z0)), 2.)
    return ans
