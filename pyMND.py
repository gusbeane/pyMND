import numpy as np
import arepo
from math import log, sqrt, exp
from scipy.spatial import ConvexHull

from units import pyMND_units
from halo import *
from gas_halo import *
from util import *
from potential import *

from tqdm import tqdm
import time

class pyMND(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, 
                 MGH, GasHaloSpinFraction,
                 HubbleParam, BoxSize, AddBackgroundGrid,
                 OutputDir, OutputFile):

        self.CC = CC
        self.V200 = V200
        self.LAMBDA = LAMBDA
        self.N_HALO = N_HALO
        self.N_GAS = N_GAS

        self.MGH = MGH

        self.GasHaloSpinFraction = GasHaloSpinFraction

        self.OutputDir = OutputDir
        self.OutputFile = OutputFile
        self.HubbleParam = HubbleParam
        self.BoxSize = BoxSize
        self.AddBackgroundGrid = AddBackgroundGrid

        self.u = pyMND_units(self.HubbleParam)

        # initialize structural constants
        self._structure()

        # draw positions
        self._draw_pos()

        # draw velocities
        self._draw_vel()

        # get temperature of gas
        self._get_gas_thermal_energy()

        # add background grid of cells
        self._add_background_grid()

        # output to file
        self._output_ics_file()

        print('R200=', self.R200, 'M200=', self.M200, 'R200/a=', self.R200/self.RH)
        print('WARNING: gas masses are actually densities, you must enable MESHRELAX_DENSITY_IN_INPUT in arepo!!')

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

        if self.M_GASHALO > 0.0:
            vcsq = self.circular_velocity_squared(self.gashalo_pos)
            self.gashalo_vel = draw_gas_halo_vel(self.gashalo_pos, vcsq, self.halo_spinfactor, self.GasHaloSpinFraction)
    
    def _get_gas_thermal_energy(self):
        if self.M_GASHALO > 0.0:
            self.gashalo_u = gas_halo_thermal_energy(self.gashalo_pos, self.M_GASHALO, self.RH, self.u)
    
    def _add_background_grid(self):
        if self.AddBackgroundGrid == 0:
            return
        
        # first shift the gas distribution
        self.gashalo_pos += np.array([self.BoxSize/2.0, self.BoxSize/2.0, self.BoxSize/2.0])

        # next find the convex hull of the gas distribution
        hull = ConvexHull(self.gashalo_pos)

        # now iterate and insert background points if they are not in the convex hull
        bg_1d_pos = self.BoxSize * (np.arange(0, self.AddBackgroundGrid) + 0.5) / self.AddBackgroundGrid
        bg_points = gen_3D_grid(bg_1d_pos)

        outside_hull = np.where([point_in_hull(bg_p, hull) for bg_p in bg_points])[0]
        bg_points = bg_points[outside_hull]

        # construct arrays
        Nbgpoints = len(bg_points)
        self.gashalo_pos = np.concatenate((self.gashalo_pos, bg_points))
        self.gashalo_vel = np.concatenate((self.gashalo_vel, np.zeros((Nbgpoints, 3))))
        self.gashalo_u = np.concatenate((self.gashalo_u, np.zeros(Nbgpoints)))
        self.gashalo_mass = np.concatenate((self.gashalo_mass, np.zeros(Nbgpoints)))
        self.N_GAS += Nbgpoints

        # some sanity checks
        assert len(self.gashalo_pos) == self.N_GAS
        assert len(self.gashalo_vel) == self.N_GAS
        assert len(self.gashalo_u) == self.N_GAS
        assert len(self.gashalo_mass) == self.N_GAS
        return

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
            ics.part0.vel[:] = self.gashalo_vel
            ics.part0.u[:] = self.gashalo_u
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
    N_GAS = 64*208333
    N_HALO = 64*375000
    MGH = 0.1
    GasHaloSpinFraction = 1.0
    HubbleParam = 1.0
    BoxSize=1200.0
    AddBackgroundGrid = 16
    OutputDir='./'
    OutputFile='MW_ICs'
    t = pyMND(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction, HubbleParam, BoxSize, 
              AddBackgroundGrid, OutputDir, OutputFile)
    
