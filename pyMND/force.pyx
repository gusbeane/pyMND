from .hernquist import _hernquist_potential_derivative_z, _hernquist_potential_derivative_R
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
        R_list[i] = exp(log(Baselen) + (i - 1) * (log(LL) - log(Baselen)) / (RSIZE - 1))
        RplusdR_list[i] = exp(log(Baselen) + (i - 1 + 0.01) * (log(LL) - log(Baselen)) / (RSIZE - 1))
    
    for i in range(1, ZSIZE+1):
        z_list[i] = exp(log(Baselen) + (i - 1) * (log(LL) - log(Baselen)) / (ZSIZE - 1))
    
    return R_list, RplusdR_list, z_list

def generate_force_grid(RSIZE, ZSIZE, H, R200):
    R_list, RplusdR_list, z_list = _generate_force_grid(RSIZE, ZSIZE, H, R200)

    force_grid = {}
    force_grid['R_list'] = R_list
    force_grid['RplusdR_list'] = RplusdR_list
    force_grid['z_list'] = z_list

    force_grid['R'], _ = np.meshgrid(R_list, z_list, indexing='ij')
    force_grid['RplusdR'], force_grid['z'] = np.meshgrid(RplusdR_list, z_list, indexing='ij')

    return force_grid

cpdef _compute_forces(double[:] R_list, double[:] z_list, double[:] R_dR_list, int RSIZE, int ZSIZE,
                     double MHALO, double RH, double MBULGE, double A, double G, forcetree.TREE disk_tree,
                     forcetree.TREE gas_tree):
    cdef double[:,:] Dphi_R
    cdef double[:,:] Dphi_z
    cdef double[:,:] Dphi_z_dR
    cdef double[:,:] VelVc2
    cdef double[:] epi_gamma2
    cdef double[:] epi_kappa2
    cdef int i, j
    cdef double R, z
    # cdef np.ndarray Dphi_z = np.zeros([RSIZE, ZSIZE], dtype=np.float)

    Dphi_R = np.zeros((RSIZE+1, ZSIZE+1))
    Dphi_z = np.zeros((RSIZE+1, ZSIZE+1))
    Dphi_z_dR = np.zeros((RSIZE+1, ZSIZE+1))
    VelVc2 = np.zeros((RSIZE+1, ZSIZE+1))
    epi_gamma2 = np.zeros(RSIZE+1)
    epi_kappa2 = np.zeros(RSIZE+1)

    for i in range(RSIZE+1):
        R = R_list[i]
        RdR = R_dR_list[i]
        for j in range(RSIZE+1):
            z = z_list[j]

            if i==0:
                Dphi_R[i][j] = 0.0
                VelVc2[i][j] = 0.0
            else:
                Dphi_R[i][j] = comp_Dphi_R(R, z, MHALO, RH, MBULGE, A, G, disk_tree, gas_tree)
                VelVc2[i][j] = R * Dphi_R[i][j]
            
            if j==0:
                Dphi_z[i][j] = 0.0
                Dphi_z_dR[i][j] = 0.0
            else:
                Dphi_z[i][j] = comp_Dphi_z(R, z, MHALO, RH, MBULGE, A, G, disk_tree, gas_tree)
                Dphi_z_dR[i][j] = comp_Dphi_z(RdR, z, MHALO, RH, MBULGE, A, G, disk_tree, gas_tree)
    
    epi_gamma2[0] = 0.0
    for i in range(1, RSIZE+1):
        RdR = R_dR_list[i]
        dphi_R_dr = comp_Dphi_R(RdR, 0, MHALO, RH, MBULGE, A, G, disk_tree, gas_tree)

        k2 = 3. / R_list[i] * Dphi_R[i][0] + (dphi_R_dr - Dphi_R[i][0]) / (R_dR_list[i] - R_list[i])

        epi_gamma2[i] = 4 / R_list[i] * Dphi_R[i][0] / k2
        epi_kappa2[i] = k2
    
    epi_kappa2[0] = epi_kappa2[1]

    return Dphi_R, Dphi_z, Dphi_z_dR, VelVc2, epi_gamma2, epi_kappa2

def compute_forces(force_grid, p, u, disk_tree, gas_tree):
    R_list, z_list, RplusdR_list = force_grid['R_list'], force_grid['z_list'], force_grid['RplusdR_list']

    Dphi_R, Dphi_z, Dphi_z_dR, VelVc2, epi_gamma2, epi_kappa2 = _compute_forces(R_list, z_list, 
                            RplusdR_list, p.RSIZE, p.ZSIZE, p.M_HALO, p.RH, p.M_BULGE, p.A, u.G, disk_tree, gas_tree)

    force_grid['Dphi_R'] = Dphi_R
    force_grid['Dphi_z'] = Dphi_z
    force_grid['Dphi_z_dR'] = Dphi_z_dR
    force_grid['VelVc2'] = VelVc2
    force_grid['epi_gamma2'] = epi_gamma2
    force_grid['epi_kappa2'] = epi_kappa2
    
    return force_grid

cpdef _compute_vertical_forces(double[:] R_list, double[:] z_list, double[:] R_dR_list, int RSIZE, int ZSIZE,
                     double MHALO, double RH, double MBULGE, double A, double G, forcetree.TREE disk_tree,
                     forcetree.TREE gas_tree):
    cdef double[:,:] Dphi_z
    cdef int i, j
    cdef double R, z
    # cdef np.ndarray Dphi_z = np.zeros([RSIZE, ZSIZE], dtype=np.float)

    Dphi_z = np.zeros((RSIZE+1, ZSIZE+1))

    for i in range(RSIZE+1):
        R = R_list[i]
        RdR = R_dR_list[i]
        for j in range(RSIZE+1):
            z = z_list[j]

            if j==0:
                Dphi_z[i][j] = 0.0
            else:
                Dphi_z[i][j] = comp_Dphi_z(R, z, MHALO, RH, MBULGE, A, G, disk_tree, gas_tree)
    
    return Dphi_z

def compute_vertical_forces(force_grid, p, u, disk_tree, gas_tree):
    R_list, z_list, RplusdR_list = force_grid['R_list'], force_grid['z_list'], force_grid['RplusdR_list']

    Dphi_z = _compute_vertical_forces(R_list, z_list, 
                            RplusdR_list, p.RSIZE, p.ZSIZE, p.M_HALO, p.RH, p.M_BULGE, p.A, u.G, disk_tree, gas_tree)

    force_grid['Dphi_z'] = Dphi_z
    
    return force_grid

cpdef comp_Dphi_z(double R, double z, double MHALO, double RH, double MBULGE, double A, double G, forcetree.TREE disk_tree, forcetree.TREE gas_tree):
    cdef double ans
    cdef double[:] pos
    
    pos = np.array([R, 0, z])
    frc_disk = force_treeevaluate(pos, disk_tree)
    frc_gas = force_treeevaluate(pos, gas_tree)
    
    ans = _hernquist_potential_derivative_z(R, z, MHALO, RH, G)
    ans += _hernquist_potential_derivative_z(R, z, MBULGE, A, G)
    ans += -G * frc_disk[2]
    ans += -G * frc_gas[2]

    return ans

cpdef comp_Dphi_R(double R, double z, double MHALO, double RH, double MBULGE, double A, double G, forcetree.TREE disk_tree, forcetree.TREE gas_tree):
    cdef double ans
    cdef double[:] pos
    
    pos = np.array([R, 0, z])
    frc_disk = force_treeevaluate(pos, disk_tree)
    frc_gas = force_treeevaluate(pos, gas_tree)
    
    ans = _hernquist_potential_derivative_R(R, z, MHALO, RH, G)
    ans += _hernquist_potential_derivative_R(R, z, MBULGE, A, G)
    ans += -G * frc_disk[0]
    ans += -G * frc_gas[0]

    return ans
