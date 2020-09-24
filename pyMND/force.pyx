from .hernquist import _hernquist_potential_derivative_z
cimport forcetree
from .forcetree import force_treeevaluate, TREE

from math import exp, log, sqrt
import numpy as np

cpdef _generate_force_grid(int RSIZE, int ZSIZE, double H, double R200):
    if H > 0.0:
        Baselen = 0.001 * H
    else:
        # Approximately equal to the above for a MW disk
        Baselen = 1.7E-5 * R200
    LL = 10.0 * R200

    cdef double[:] R_list
    cdef double[:] RplusdR_list
    cdef double[:] z_list
    cdef int i, j

    R_list = np.zeros(RSIZE)
    RplusdR_list = np.zeros(RSIZE)
    z_list = np.zeros(RSIZE)

    for i in range(RSIZE):
        R_list[i] = exp(log(Baselen) + i * (log(LL) - log(Baselen)) / (RSIZE - 1))
        RplusdR_list[i] = exp(log(Baselen) + (i + 0.01) * (log(LL) - log(Baselen)) / (RSIZE - 1))
    
    for i in range(ZSIZE):
        z_list[i] = exp(log(Baselen) + i * (log(LL) - log(Baselen)) / (ZSIZE - 1))
    
    return R_list, RplusdR_list, z_list

cpdef compute_Dphi_z(double[:] R_list, double[:] z_list, int RSIZE, int ZSIZE,
                     double MHALO, double RH, double G, forcetree.TREE disk_tree):
    cdef double[:,:] Dphi_z
    cdef int i, j
    cdef double R, z
    Dphi_z = np.zeros((RSIZE, ZSIZE))

    for i in range(RSIZE):
        R = R_list[i]
        for j in range(RSIZE):
            z = z_list[j]
            # Halo
            Dphi_z[i][j] = _hernquist_potential_derivative_z(R, z, MHALO, RH, G)

            if i==99 and j==99:
                print('halo=', Dphi_z[i][j])

            # Bulge
            #TODO

            # Disk
            pos = np.array([R, 0, z])
            frc = force_treeevaluate(pos, disk_tree)
            Dphi_z[i][j] += - G * frc[2]

            if i==99 and j==99:
                print('tree=', -G * frc[2])
                print('R=', R, 'z=', z)

    
    return Dphi_z