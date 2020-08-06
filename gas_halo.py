import numpy as np
from halo import *
from util import *
from numba import njit

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

@njit
def draw_gas_halo_pos(N, M, a, Rmax):
    # We have a target gas cell mass, in units of M:
    mg = M/N
    print('mg=', mg, 'N=', N)

    # Initialize gas halo positions.
    gas_halo_pos = np.zeros((N, 3))
    gas_halo_mass = np.zeros(N)
    i = 0

    # We first place a cell at the origin, and then determine its radius based on the enclosed
    # mass matching the target mass.
    i += 1
    r0 = (2. * mg + 2. * np.sqrt(mg)) / (2 * (1. - mg))

    # Offset the position of the cell by 1e-9 just so we dont have to deal with undefined behavior for r
    # close to 0.
    phi = np.multiply(2.*np.pi, np.random.rand())
    theta = np.arccos(np.random.rand() * 2. - 1.)
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    gas_halo_pos[0] = 1e-9 * np.array([x, y, z])

    # To set the density of the first cell, just use the density of a sphere of radius r0
    gas_halo_mass[0] = mg / ((4.*np.pi/3.) * r0**3.)

    Rn1 = r0
    # Now we iteratively place shells of N gas cells such that each cell has on average the correct
    # mass and the cells are roughly evenly shaped.
    ct = 1
    print('ct=', 0, 'Nn=', 1, 'rn=', r0, 'i=', i, 'N=', N, 'Rn1=', Rn1)
    while i < N:
        Nn = 10
        old_good, old_rn, tmp = goodness(Rn1, mg, Nn)
        Nn += 1
        for _ in range(1000000):
            good, rn, tmp = goodness(Rn1, mg, Nn)
            if good < old_good:
                # print('good=', good, 'old_good=', old_good, 'Nn=', Nn)
                old_good = good
                old_rn = rn
                Nn += 1
            else:
                good = old_good
                rn = old_rn
                Nn -= 1
                break
        
        if i+Nn > N:
            print('i+Nn > N, something seems wrong in gas halo')
            break
        if Rn1 + rn > Rmax:
            # expected break condition
            break

        # now assign this sphere's positions
        gas_halo_pos[i:i+Nn] = (Rn1 + rn) * draw_golden_spiral(Nn, random_orientation=False)
        gas_halo_mass[i:i+Nn] = gas_halo_density(gas_halo_pos[i:i+Nn], M, a)

        Rn1 += 2*rn
        ct += 1
        i += Nn
    
    
    return i, gas_halo_pos[:i], gas_halo_mass[:i]

def draw_gas_halo_vel(pos, vcsq, halo_spinfactor, GasHaloSpinFraction):
    vphi = halo_spinfactor * GasHaloSpinFraction * np.sqrt(vcsq)

    R = np.linalg.norm(pos[:, :2], axis=1)

    vel = np.zeros(np.shape(pos))
    vel[:,0] = - vphi * pos[:,1] / R
    vel[:,1] = vphi * pos[:,0] / R

    return vel
