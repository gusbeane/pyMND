from .hernquist import _hernquist_potential_derivative_z
cimport forcetree
from .forcetree import force_treeevaluate, TREE

from math import exp, log, sqrt
import numpy as np
cimport numpy as np

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

    R_list = np.zeros(RSIZE+1)
    RplusdR_list = np.zeros(RSIZE+1)
    z_list = np.zeros(RSIZE+1)

    for i in range(1, RSIZE+1):
        R_list[i] = exp(log(Baselen) + i * (log(LL) - log(Baselen)) / (RSIZE - 1))
        RplusdR_list[i] = exp(log(Baselen) + (i + 0.01) * (log(LL) - log(Baselen)) / (RSIZE - 1))
    
    for i in range(1, ZSIZE+1):
        z_list[i] = exp(log(Baselen) + i * (log(LL) - log(Baselen)) / (ZSIZE - 1))
    
    return R_list, RplusdR_list, z_list

cpdef compute_Dphi_z(double[:] R_list, double[:] z_list, int RSIZE, int ZSIZE,
                     double MHALO, double RH, double G, forcetree.TREE disk_tree):
    cdef double[:,:] Dphi_z
    cdef double[:,:] 
    cdef int i, j
    cdef double R, z
    # cdef np.ndarray Dphi_z = np.zeros([RSIZE, ZSIZE], dtype=np.float)

    Dphi_z = np.zeros((RSIZE, ZSIZE))

    for i in range(RSIZE+1):
        R = R_list[i]
        for j in range(RSIZE+1):
            if j==0:
                Dphi_z[i][j] = 0.0

            else:
                z = z_list[j]
                # Halo
                Dphi_z[i][j] = _hernquist_potential_derivative_z(R, z, MHALO, RH, G)

                # Bulge
                #TODO

                # Disk
                pos = np.array([R, 0, z])
                frc = force_treeevaluate(pos, disk_tree)
                Dphi_z[i][j] += - G * frc[2]

    return Dphi_z


cpdef compute_Dphi_R(double[:] R_list, double[:] z_list, int RSIZE, int ZSIZE,
                     double MHALO, double RH, double G, forcetree.TREE disk_tree):
    cdef double[:,:] Dphi_R
    cdef int i, j
    cdef double R, z
    # cdef np.ndarray Dphi_z = np.zeros([RSIZE, ZSIZE], dtype=np.float)

    Dphi_R = np.zeros((RSIZE, ZSIZE))

    for i in range(RSIZE):
        R = R_list[i]
        for j in range(RSIZE):
            z = z_list[j]
            
            # Halo
            Dphi_R[i][j] = _hernquist_potential_derivative_R(R, z, MHALO, RH, G)

            # Bulge
            #TODO

            # Disk
            pos = np.array([R, 0, z])
            frc = force_treeevaluate(pos, disk_tree)
            Dphi_R[i][j] += - G * frc[0]

    return Dphi_R, epi_gamma2, epi_kappa2
