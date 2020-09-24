import numpy as np
from math import sqrt, exp, log

from .forcetree import force_treeevaluate
# from .hernquist import _hernquist_potential_derivative_z#, _hernquist_potential_derivative_R
from .force import compute_Dphi_z

from numba import njit

def _compute_forces_on_grid(R_list, RplusdR_list, z_list, p, u, disk_tree, gas_tree=None):

    Dphi_z = compute_Dphi_z(R_list, z_list, p.RSIZE, p.ZSIZE, p.M_HALO, p.RH, u.G, disk_tree)

    return Dphi_z
