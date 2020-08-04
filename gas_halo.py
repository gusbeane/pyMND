import numpy as np
from halo import *

def gas_halo_temperature(r, M, a, u):
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

def gas_halo_potential(pos, M, a, u):
    return halo_potential(pos, M, a, u)

def gas_halo_potential_derivative_R(pos, M, a, u):
    return halo_potential_derivative_R(pos, M, a, u)

def draw_gas_halo_pos(N, a, u):
    return draw_halo_pos(N, a, u)
