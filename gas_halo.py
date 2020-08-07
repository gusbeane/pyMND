import numpy as np
from halo import *
from util import *
from numba import njit

def gas_halo_temperature(r, M, a, u):
    # TODO: check for if T < 1e4 K, maybe set T floor at 1e4 K?
    meanweight = 4 / (8 - 5 * (1 - u.HYDROGEN_MASSFRAC)) # assume full ionization, in units of proton mass
    mu = meanweight * u.PROTONMASS / u.UnitMass_in_g
    kB = u.BOLTZMANN / u.UnitEnergy_in_cgs
    G = u.G

    x = r / a
    term1 = np.log(1 + 1./x)
    term2 = (25. + 52.*x + 42.*x**2. + 12*x**3.) / (12 * (1+x)**4.)

    ans = x * (1+x)**3. * (term1 - term2)

    ans *= (mu / kB) * G * M / a
    return ans

def gas_halo_thermal_energy(pos, M, a, u):
    # TODO: fix for if the T is < 1e4 K
    r = np.linalg.norm(pos, axis=1)
    T = gas_halo_temperature(r, M, a, u)

    T[T < u.TFLOOR] = u.TFLOOR

    meanweight = 4 / (8 - 5 * (1 - u.HYDROGEN_MASSFRAC))
    energy = 1 / meanweight * (1.0 / u.GAMMA_MINUS1) * (u.BOLTZMANN / u.PROTONMASS) * T
    energy *= u.UnitMass_in_g / u.UnitEnergy_in_cgs

    return energy

def gas_halo_potential(pos, M, a, u):
    return halo_potential(pos, M, a, u)

def gas_halo_potential_derivative_R(pos, M, a, u):
    return halo_potential_derivative_R(pos, M, a, u)

@njit
def gas_halo_density(pos, M, a):
    return halo_density(pos, M, a)

@njit
def goodness(Rn1, mg, Nn):
    X = mg*Nn + Rn1**2 / (1+Rn1)**2
    mass_rn = (Rn1*(X-1) + X + np.sqrt(X)) / (2*(1-X))

    # d = np.sqrt(4 - np.sin(np.pi*Nn/6 * (Nn-2))**(-2.))
    d = 3.091 * Nn**(-0.5)
    size_rn = Rn1 * d / (1-d)

    return np.abs(mass_rn - size_rn), mass_rn, size_rn

def draw_gas_halo_pos(N, M, a, Rmax):
    # We have a target gas cell mass, in units of M:
    mg = M/N
    print('mg=', mg, 'N=', N)

    u, v = R2_method(N)
    phi = 2.*np.pi * u
    theta = np.arccos(2.*v - 1)

    f = np.arange(0.5, N)/N
    sqrtf = np.sqrt(f)
    r = a * sqrtf / (1. - sqrtf)

    gas_halo_pos = np.transpose([r * np.cos(phi) * np.sin(theta),
                                 r * np.sin(phi) * np.sin(theta),
                                 r * np.cos(theta)])
    
    key = np.where(r < Rmax)[0]
    N = len(key)
    gas_halo_pos = gas_halo_pos[key]
    gas_halo_mass = np.full(N, mg)
    
    return N, gas_halo_pos, gas_halo_mass

def draw_gas_halo_vel(pos, vcsq, halo_spinfactor, GasHaloSpinFraction):
    vphi = halo_spinfactor * GasHaloSpinFraction * np.sqrt(vcsq)

    R = np.linalg.norm(pos[:, :2], axis=1)

    vel = np.zeros(np.shape(pos))
    vel[:,0] = - vphi * pos[:,1] / R
    vel[:,1] = vphi * pos[:,0] / R

    return vel
