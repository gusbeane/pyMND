import numpy as np
import arepo
from math import log, sqrt, exp
from scipy.spatial import ConvexHull

from .units import pyMND_units
from .halo import *
from .gas_halo import *
from .util import *
from .param import gen_pyMND_param
from .potential import *

from tqdm import tqdm
import time

class pyMND(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, 
                 MGH, GasHaloSpinFraction,
                 HubbleParam, BoxSize, AddBackgroundGrid,
                 OutputDir, OutputFile):

        np.random.seed(160)

        self.data = {}

        self.u = pyMND_units(HubbleParam)
        self.p = gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction,
                                 HubbleParam, BoxSize, AddBackgroundGrid, OutputDir, OutputFile, self.u)

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

        print('R200=', self.p.R200, 'M200=', self.p.M200, 'R200/a=', self.p.R200/self.p.RH)
    
    def circular_velocity_squared(self, pos):
        R = np.linalg.norm(pos[:,:2], axis=1)
        partial_phi = potential_derivative_R(pos, self.p, self.u)
        return R * partial_phi
    
    def _draw_pos(self):
        self.data['part1'] = {}
        self.data['part1']['pos'] = draw_halo_pos(self.p, self.u)
        if self.p.M_GASHALO > 0.0:
            self.data['part0'] = {}
            self.p.N_GAS, self.data['part0']['pos'], self.data['part0']['mass'] = draw_gas_halo_pos(self.p)

    def _draw_vel(self):
        vcsq = self.circular_velocity_squared(self.data['part1']['pos'])
        self.halo_vel = draw_halo_vel(self.data['part1']['pos'], vcsq, self.p, self.u)

        if self.p.M_GASHALO > 0.0:
            vcsq = self.circular_velocity_squared(self.data['part0']['pos'])
            self.data['part0']['vel'] = draw_gas_halo_vel(self.data['part0']['pos'], vcsq, self.p)
    
    def _get_gas_thermal_energy(self):
        if self.p.M_GASHALO > 0.0:
            self.data['part0']['u'] = gas_halo_thermal_energy(self.data['part0']['pos'], self.p, self.u)
    
    def _add_background_grid(self):
        if self.p.AddBackgroundGrid == 0:
            return
        
        # first shift the gas distribution
        for k in self.data.keys():
            self.data[k]['pos'] += np.array([self.p.BoxSize/2.0, self.p.BoxSize/2.0, self.p.BoxSize/2.0])

        # next find the convex hull of the gas distribution
        hull = ConvexHull(self.data['part0']['pos'])

        # now iterate and insert background points if they are not in the convex hull
        bg_1d_pos = self.p.BoxSize * (np.arange(0, self.p.AddBackgroundGrid) + 0.5) / self.p.AddBackgroundGrid
        bg_points = gen_3D_grid(bg_1d_pos)

        outside_hull = np.where([point_in_hull(bg_p, hull) for bg_p in bg_points])[0]
        bg_points = bg_points[outside_hull]

        # construct arrays
        Nbgpoints = len(bg_points)
        self.data['part0']['pos']  = np.concatenate((self.data['part0']['pos'], bg_points))
        self.data['part0']['vel']  = np.concatenate((self.data['part0']['vel'], np.zeros((Nbgpoints, 3))))
        self.data['part0']['u']    = np.concatenate((self.data['part0']['u'], np.zeros(Nbgpoints)))
        self.data['part0']['mass'] = np.concatenate((self.data['part0']['mass'], np.zeros(Nbgpoints)))
        self.p.N_GAS += Nbgpoints

        # some sanity checks
        assert len(self.data['part0']['pos'] ) == self.p.N_GAS
        assert len(self.data['part0']['vel'] ) == self.p.N_GAS
        assert len(self.data['part0']['u']   ) == self.p.N_GAS
        assert len(self.data['part0']['mass']) == self.p.N_GAS

        return

    def _output_ics_file(self):
        npart = [self.p.N_GAS, self.p.N_HALO, 0, 0, 0, 0]
        masses = [0, self.p.M_HALO/self.p.N_HALO, 0, 0, 0, 0]

        out_file = self.p.OutputDir + '/' + self.p.OutputFile
        if out_file[5:] != '.hdf5' and out_file[3:] != '.h5':
            out_file += '.hdf5'
        
        ics = arepo.ICs(out_file, npart, masses=masses)
        id0 = 1
        for part in self.data.keys():
            for key in self.data[part].keys():
                getattr(getattr(ics, part), key)[:] = self.data[part][key]
            N = len(self.data[part]['pos'])
            getattr(getattr(ics, part), 'id')[:] = np.arange(id0, id0+N)
            id0 += N

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
    
