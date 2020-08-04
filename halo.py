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

def draw_halo_pos(N, a, u):
        r = halo_draw_r(N, a)
        phi = np.multiply(2.*u.PI, np.random.rand(N))
        theta = np.arccos(np.random.rand(N) * 2. - 1.)

        stheta = np.sin(theta)
        ctheta = np.cos(theta)

        xp_halo = np.multiply(np.multiply(r, stheta), np.cos(phi))
        yp_halo = np.multiply(np.multiply(r, stheta), np.sin(phi))
        zp_halo = np.multiply(r, ctheta)
        halo_pos = np.transpose([xp_halo, yp_halo, zp_halo])
        return halo_pos

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

def _compute_vel_disp_halo(self):
        ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi = \
            compute_velocity_ellipsoid_halo(self.halo_pos, self.M_HALO, self.RH,
                                            self.u, vcirc_squared, self.halo_spinfactor)
        
        self.ave_R, self.ave_z, self.ave_phi = ave_R, ave_z, ave_phi
        self.sigma_R, self.sigma_z, self.sigma_phi = sigma_R, sigma_z, sigma_phi

def draw_halo_vel(pos, vcsq, N, M, a, lam, u):
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
