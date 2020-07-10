import numpy as np

def halo_potential(pos, M, a, u):
    r = np.linalg.norm(pos, axis=1)
    ans = - u.G * M
    ans /= r + a
    return ans

def halo_potential_derivative_R(pos, M, a, u):
    r = np.linalg.norm(pos, axis=1)
    R = np.linalg.norm(pos[:,:2], axis=1)
    ans = u.G * M
    ans /= np.square(r + a)
    ans *= R/r

    return ans

def halo_potential_derivative_z(pos, M, a, u):
    r = np.linalg.norm(pos, axis=1)
    z = pos[:,2]
    ans = u.G * M
    ans /= np.square(r + a)
    ans *= z/r

    return ans


def halo_draw_r(N, a):
    f = np.random.rand(N)
    sqrtf = np.sqrt(f)
    rt = np.divide(sqrtf, np.subtract(1., sqrtf))
    
    return np.multiply(rt, a)

def compute_velocity_ellipsoid_halo(pos, M, a, u, vcirc_squared, halo_spinfactor):
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

    print('rho=', rho, 'M=', M, 'a=', a, 'rplusa=', rplusa)
    print('R/rho*deriv=', (R/rho)*partial_rhosigma, R, rho)
    print('deriv=', partial_rhosigma)
    print('vc_sq=', vcirc_squared)

    vphi_squared = vRz_squared + (R/rho) * partial_rhosigma + vcirc_squared
    ave_phi = halo_spinfactor * np.sqrt(vcirc_squared)

    sigma_phi = np.sqrt(vphi_squared - np.square(ave_phi))
    sigma_R = np.sqrt(vRz_squared)
    sigma_z = np.copy(sigma_R)
    ave_R = np.zeros(len(sigma_R))
    ave_z = np.zeros(len(sigma_z))

    return ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi
