import numpy as np
from math import exp
from scipy.integrate import cumtrapz
from numba import njit

import sys

from .forcetree import construct_tree, construct_empty_tree
from .force import compute_vertical_forces

global ctr 
ctr = 0

def init_gas_field(force_grid, disk_tree, p, u):
    # RhoGas 
    RhoGas = np.zeros((p.RSIZE, p.ZSIZE))

    # Set the initial gas central density assuming an isothermal sheet. This is just a guess.
    RhoGas = surface_density_gasdisk(force_grid['R'], p, u)
    RhoGas *= (1./(2*p.H*p.H)) * np.power(np.cosh(force_grid['z']/(2. * p.Z0)), -2.0)

    # print('initial RhoGas[24][0]=', RhoGas[24][0])
    # RhoGas[24][0] = 0.207637
    # print(RhoGas[256][:])
    # np.save('RhoGas_init.npy', RhoGas[24])
    # np.save('z_list.npy', force_grid['z_list'])
    
    for _ in range(5):
        FracEnclosed = integrate_RhoGas(RhoGas, force_grid, p)

        # Now, draw the dummy positions.
        dummy_pos, dummy_mass = draw_gas_disk_dummy_pos(FracEnclosed, force_grid['R_list'], force_grid['z_list'], p, u)

        gas_tree = construct_tree(dummy_pos, dummy_mass, p.Theta, 0.01 * p.H)
        force_grid = compute_vertical_forces(force_grid, p, u, disk_tree, gas_tree)

        # Okay, we have the potential, time to adjust the density so that it is in equilibrium with
        # the potential.
        for _ in range(5):
            RhoGas = step_gas_equilibrium(RhoGas, np.array(force_grid['R_list']), np.array(force_grid['z_list']), force_grid['Dphi_z'], p, u, _)
            # print(RhoGas[256][:])
            # np.save('RhoGas_'+str(_)+'.npy', RhoGas2)
        
        # sys.exit()

    force_grid['RhoGas'] = RhoGas

    FracEnclosed = integrate_RhoGas(RhoGas, force_grid, p)

    # Now, draw the dummy positions.
    dummy_pos, dummy_mass = draw_gas_disk_dummy_pos(FracEnclosed, force_grid['R_list'], force_grid['z_list'], p, u)

    gas_tree = construct_tree(dummy_pos, dummy_mass, p.Theta, 0.01 * p.H)

    return force_grid, gas_tree

@njit
def step_gas_equilibrium(RhoGas, R_list, z_list, Dphi_z, p, u, ctr):
    for i in range(p.RSIZE):
        # First, set all the densities to be zero above the central density
        for j in range(1, p.ZSIZE):
            RhoGas[i][j] = 0.0
        
        # Make sure we are within the extent of the gas disk.
        if R_list[i]/p.H > p.NumGasScaleLengths:
            RhoGas[i][0] = 0.0
            continue
    
        # Now, go through and update
        drho = 0.0
        rho0 = RhoGas[i][0]
        for j in range(1, p.ZSIZE):
            rho = RhoGas[i][j-1]
            rhoold = rho
            dz = z_list[j] - z_list[j-1]

            rho += 0.5 * dz * drho
            P = eos_gas(rho, p)
            # if P <1e-7:
            #     P=1.0
            # P1 = eos_gas(1.01*rho, p)
            # gam = np.log(P1/P) / np.log(1.01)
            # print(gam)
            gam = 1.0

            # if i==24:
            # if i==24:
                # print("Dphi_z[i][j]=", Dphi_z[i][j], "Dphi_z[i][j+1]=", Dphi_z[i][j+1], "rho=", rho, "P=", P, "gam=", gam)


            drho = -0.5 * (Dphi_z[i][j] + Dphi_z[i][j+1]) * rho * rho / P / gam
            
            # if i==24:
                # print("drho=", drho, "Dphi_z[i][j]=", Dphi_z[i][j], "Dphi_z[i][j+1]=", Dphi_z[i][j+1], "rho=", rho, "P=", P, "gam=", gam)

            rho = rhoold + drho * dz

            if rho < 0.0:
                rho = 0.0

            RhoGas[i][j] = rho
            if rho < 0.00001 * rho0 or rho == 0.0:
                break
    
    # np.save('RhoGas_'+ctr+'.npy', RhoGas[24])

    # Now, we integrate vertically and adjust the 
    surf_target = surface_density_gasdisk(R_list, p, u)
    for i in range(p.RSIZE):
        if R_list[i]/p.H > p.NumGasScaleLengths:
            continue

        surf_equib = 2.0 * np.trapz(RhoGas[i,:], z_list)
        # factor of two accounts for other side
        # if i==24:
            # print('factor to multiply by is: ', surf_target[i]/surf_equib, surf_target[i], surf_equib)
        fac = surf_target[i]/surf_equib
        # if fac > 2.:
            # fac = 2.
        # if fac < 0:
            # fac = 2.
        for j in range(p.ZSIZE):
            RhoGas[i][j] *= fac
    
    return RhoGas

def integrate_RhoGas(RhoGas, force_grid, p):
    # integrating RhoGas along the z-axis to get the fraction enclosed
    # as a function of z in order to be able to make position draws
    FracEnclosed = cumtrapz(RhoGas, force_grid['z_list'], initial=0, axis=1)

    # Divide by the maximal value in each row, setting to zero expected inf's
    FracEnclosed = np.transpose(np.divide(np.transpose(FracEnclosed), FracEnclosed[:,-1]))
    FracEnclosed[force_grid['R']/p.H > p.NumGasScaleLengths] = 0.0 

    return FracEnclosed

@njit
def draw_gas_disk_dummy_pos(FracEnclosed, R_list, z_list, p, u):
    N = p.RMASSBINS * p.ZMASSBINS * p.PHIMASSBINS
    pos = np.zeros((N, 3))

    it = 0
    for i in range(p.RMASSBINS):
        q = (i + 0.5) / p.RMASSBINS
        R = draw_R_gas_disk(q, p)

        for j in range(p.ZMASSBINS):
            q = (j + 0.5) / p.ZMASSBINS

            q = 2. * q - 1
            sign_q = np.sign(q)
            q = np.abs(q)

            z = sign_q * draw_z_gas_disk(R, q, R_list, z_list, FracEnclosed, p)

            for k in range(p.PHIMASSBINS):
                phi = 2. * u.PI * (k + 0.5) / p.PHIMASSBINS

                pos[it][0] = R * np.cos(phi)
                pos[it][1] = R * np.sin(phi)
                pos[it][2] = z
                it += 1
    
    mass = np.full(N, p.M_GAS/N)
    
    return pos, mass

@njit
def surface_density_gasdisk(R, p, u):
    M_GAS = p.M_GAS
    H = p.H


    fN = 1.0 - (1.0 + p.NumGasScaleLengths) * np.exp(-p.NumGasScaleLengths)

    ans = (M_GAS / (2. * u.PI * H * H)) * np.exp(-R / H) / fN

    key = np.where(R/H > p.NumGasScaleLengths)[0]
    ans[key] = 0.0

    return ans

# R is assumed to be in units of scale lengths
@njit
def fraction_enclosed_gasdisk(R, p):
    NumGasScaleLengths = p.NumGasScaleLengths

    if(R > NumGasScaleLengths):
        return 1.0

# ifndef GAS_CORE
    ans = 1 - (1 + R) * np.exp(-R)
    ans /= 1 - (1 + NumGasScaleLengths) * exp(-NumGasScaleLengths)
#else

#   if(R <= Rcore)
    # ans = 0.5 * R*R * exp(-Rcore);
#   else
#   {
    # ans = 0.5 * Rcore*Rcore *exp(-Rcore);
    # ans += (1+Rcore)*exp(-Rcore);
    # ans -= (1 + R)*exp(-R);
#   }
#   ans /= Mtot_gas; /* divide by total mass */
#endif

    return ans

# // R is assumed to be in units of scale lengths
@njit
def fraction_enclosed_gasdisk_derivative(R, p):
    NumGasScaleLengths = p.NumGasScaleLengths

    if(R > NumGasScaleLengths):
        return 0.0
  
#ifndef GAS_CORE
    ans = R * np.exp(-R)
    ans /= 1 - (1 + NumGasScaleLengths) * exp(-NumGasScaleLengths)
#else

#   if(R <= Rcore)
    # ans = R * exp(-Rcore);
#   else
    # ans = R * exp(-R);

#   ans /= Mtot_gas; /* divide by total mass */

#endif
    return ans

@njit
def draw_R_gas_disk(q, p):
#ifndef GAS_CORE
    R = 1.0
#else
#   R = Rcore;
#endif
    itr = 0
    Rold = 0.0
    while np.abs(R - Rold)/R > 1e-7:
        f = fraction_enclosed_gasdisk(R, p) - q
        f_ = fraction_enclosed_gasdisk_derivative(R, p)

        Rold = R
        R = R - f / f_

        itr+=1
  
    if(np.abs(fraction_enclosed_gasdisk(R, p) - q) > 1e-5):
        print("WARNING: R draw of gas particle does not seem to be converged")

    return R * p.H

@njit
def draw_z_gas_disk(R, q, R_list, z_list, FracEnclosed, p):
    # First, find the R index
    for ui in range(p.RSIZE-1):
        if R > R_list[ui] and R < R_list[ui+1]:
            break
    
    # Now find the linear interpolated value between the two
    # If R1 > Rmax for the gas disk, just use the 
    # Note that R0 cannot be bigger than Rmax since R cannot be bigger than Rmax
    R0 = R_list[ui]
    R1 = R_list[ui+1]
    if R1/p.H < p.NumGasScaleLengths:
        t = (R - R0)/(R1 - R0)
    else:
        t = 0.0

    # Now, interpolate the FracEnclosed
    Fenc_atR = FracEnclosed[ui,:] * (1-t) + FracEnclosed[ui+1,:]*t

    # Now, find the z index
    for uj in range(p.ZSIZE-1):
        if q > Fenc_atR[uj] and q < Fenc_atR[uj+1]:
            break
    
    # Find the linear interpolated value
    Fenc0 = Fenc_atR[uj]
    Fenc1 = Fenc_atR[uj+1]
    u = (q - Fenc0) / (Fenc1 - Fenc0)

    # Now we can write down z
    z = z_list[uj] * (1-u) + z_list[uj+1] * u

    return z

@njit
def eos_gas(rho, p):
    return rho * p.P4_factor
