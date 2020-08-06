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

        # draw velocities
        self._draw_vel()

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
            self.N_GAS, self.gashalo_pos, self.gashalo_mass = draw_gas_halo_pos(self.N_GAS, self.M_GASHALO, self.RH, self.R200)

    def _draw_vel(self):
        vcsq = self.circular_velocity_squared(self.halo_pos)
        self.halo_vel = draw_halo_vel(self.halo_pos, vcsq, self.N_HALO, self.M_HALO, self.RH, self.halo_spinfactor, self.u)
    
    def _output_ics_file(self):
        npart = [self.N_GAS, self.N_HALO, 0, 0, 0, 0]
        masses = [0, self.M_HALO/self.N_HALO, 0, 0, 0, 0]

        out_file = self.OutputDir + '/' + self.OutputFile
        if out_file[5:] != '.hdf5' and out_file[3:] != '.h5':
            out_file += '.hdf5'
        
        ics = arepo.ICs(out_file, npart, masses=masses)
        id0 = 1
        if self.M_GASHALO > 0.0:
            ics.part0.pos[:] = self.gashalo_pos
            ics.part0.mass[:] = self.gashalo_mass
            # ics.part0.vel[:] = self.gashalo_vel
            ics.part0.id[:] = np.arange(id0, id0 + self.N_GAS)
            id0 += self.N_GAS

        ics.part1.pos[:] = self.halo_pos
        ics.part1.vel[:] = self.halo_vel
        ics.part1.id[:] = np.arange(id0, id0 + self.N_HALO)
        id0 += self.N_HALO

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
    