import numpy as np
from halo import halo_potential, halo_potential_derivative_R
from gas_halo import gas_halo_potential, gas_halo_potential_derivative_R

def potential(pos, M, a, MG, u):
    pot = halo_potential(pos, M, a, u)
    pot += gas_halo_potential(pos, MG, a, u)
    return pot
    
def potential_derivative_R(pos, M, a, MG, u):
    pot_R = halo_potential_derivative_R(pos, M, a, u)
    pot_R += gas_halo_potential_derivative_R(pos, MG, a, u)
    return pot_R