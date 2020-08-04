import numpy as np
import arepo
from math import log, sqrt, exp

from units import pyMND_units
from halo import *
from gas_halo import *
from util import *
from potential import *

class pyMND(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, 
                 MGH,
                 HubbleParam,
                 OutputDir, OutputFile):

        self.CC = CC
        self.V200 = V200
        self.LAMBDA = LAMBDA
        self.N_HALO = N_HALO
        self.N_GAS = N_GAS

        self.MGH = MGH

        self.OutputDir = OutputDir
        self.OutputFile = OutputFile
        self.HubbleParam = HubbleParam

        self.u = pyMND_units(self.HubbleParam)

        # initialize structural constants
        self._structure()

        # draw positions
        self._draw_pos()

        # compute velocity dispersions
        self._compute_vel_disp_halo()

        # draw velocities
        self._draw_halo_vel()

        # output to file
        self._output_ics_file()

    def _structure(self):
        self.M200 = self.V200**3. / (10 * self.u.G * self.u.H0)
        self.R200 = self.V200 / (10 * self.u.H0)

        self.RS = self.R200 / self.CC
        self.RHO_0 = self.M200 / (4. * self.u.PI * (log(1+self.CC) - self.CC/(1.+self.CC)) * self.RS**3.)

        self.M_DISK = 0.
        self.M_BULGE = 0.
        self.M_GASHALO = self.MGH * self.M200

        self.M_HALO = self.M200 - self.M_DISK - self.M_BULGE - self.M_GASHALO

        self.RH = self.RS * sqrt(2. * (log(1+self.CC) - self.CC / (1. + self.CC)))

        self.jhalo = self.LAMBDA * sqrt(self.u.G) * self.M200**(1.5) * sqrt(2 * self.R200 / fc(self.CC))
    
        self.halo_spinfactor = 1.5 * self.LAMBDA * sqrt(2 * self.CC / fc(self.CC))
        self.halo_spinfactor *= pow(log(1 + self.CC) - self.CC / (1 + self.CC), 1.5) / gc(self.CC)
    
    def circular_velocity_squared(self, pos):
        R = np.linalg.norm(pos[:,:2], axis=1)
        partial_phi = potential_derivative_R(pos, self.M_HALO, self.RH, self.M_GASHALO, self.u)
        return R * partial_phi
    
    def _draw_pos(self):
        self.halo_pos = draw_halo_pos(self.N_HALO, self.RH, self.u)
        if self.M_GASHALO > 0.0:
            self.gashalo_pos = draw_gas_halo_pos(self.N_GAS, self.RH, self.u)

    def _compute_vel_disp_halo(self):
        vcirc_squared = self.circular_velocity_squared(self.halo_pos)
        
        ave_R, ave_z, ave_phi, sigma_R, sigma_z, sigma_phi = \
            compute_velocity_ellipsoid_halo(self.halo_pos, self.M_HALO, self.RH,
                                            self.u, vcirc_squared, self.halo_spinfactor)
        
        self.ave_R, self.ave_z, self.ave_phi = ave_R, ave_z, ave_phi
        self.sigma_R, self.sigma_z, self.sigma_phi = sigma_R, sigma_z, sigma_phi

    def _draw_halo_vel(self):
        vR = np.random.normal(size=N_HALO)
        vz = np.random.normal(size=N_HALO)
        vphi = np.random.normal(size=N_HALO)

        vR *= self.sigma_R
        vz *= self.sigma_z
        vphi *= self.sigma_phi

        vR += self.ave_R
        vz += self.ave_z
        vphi += self.ave_phi

        R = np.linalg.norm(self.halo_pos[:, :2], axis=1)

        vx = vR * self.halo_pos[:,0] / R - vphi * self.halo_pos[:,1] / R
        vy = vR * self.halo_pos[:,1] / R + vphi * self.halo_pos[:,0] / R

        self.halo_vel = np.transpose([vx, vy, vz])
    
    def _output_ics_file(self):
        npart = [0, self.N_HALO, 0, 0, 0, 0]
        masses = [0, self.M_HALO/self.N_HALO, 0, 0, 0, 0]

        out_file = self.OutputDir + '/' + self.OutputFile
        if out_file[5:] != '.hdf5' and out_file[3:] != '.h5':
            out_file += '.hdf5'
        
        ics = arepo.ICs(out_file, npart, masses=masses)

        ics.part1.pos[:] = self.halo_pos
        ics.part1.vel[:] = self.halo_vel
        ics.part1.id[:] = np.arange(1, self.N_HALO+1)

        ics.write()



if __name__ == '__main__':
    CC = 11.0
    V200 = 163.
    LAMBDA = 0.035
    N_GAS = 39606
    N_HALO = 396060 - N_GAS
    MGH = 0.1
    HubbleParam = 1.0
    OutputDir='./'
    OutputFile='MW_ICs'
    t = pyMND(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, HubbleParam, OutputDir, OutputFile)
    