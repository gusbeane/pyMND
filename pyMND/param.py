from math import log, sqrt

from numba.types import float64, int64, unicode_type
from numba.experimental import jitclass

from .util import fc, gc
from .units import pyMND_units

spec_param = [('CC', float64),
              ('V200', float64),
              ('LAMBDA', float64),
              ('N_HALO', int64),
              ('N_GAS', int64),
              ('MGH', float64),
              ('GasHaloSpinFraction', float64),
              ('HubbleParam', float64),
              ('BoxSize', float64),
              ('AddBackgroundGrid', int64),
              ('OutputDir', unicode_type),
              ('OutputFile', unicode_type),
              ('u', pyMND_units.class_type.instance_type),
              # Structural parameters.
              ('M200', float64),
              ('R200', float64),
              ('RS', float64),
              ('RHO_0', float64),
              ('M_DISK', float64),
              ('M_BULGE', float64),
              ('M_GASHALO', float64),
              ('M_HALO', float64),
              ('RH', float64),
              ('jhalo', float64),
              ('halo_spinfactor', float64)]

@jitclass(spec_param)
class pyMND_param(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, MGH,
                 GasHaloSpinFraction, HubbleParam, BoxSize, AddBackgroundGrid,
                 OutputDir, OutputFile, Units):
        self.CC = CC
        self.V200 = V200
        self.LAMBDA = LAMBDA
        self.N_HALO = N_HALO
        self.N_GAS = N_GAS
        self.MGH = MGH
        self.GasHaloSpinFraction = GasHaloSpinFraction
        self.HubbleParam = HubbleParam
        self.BoxSize = BoxSize
        self.AddBackgroundGrid = AddBackgroundGrid
        self.OutputDir = OutputDir
        self.OutputFile = OutputFile
        self.u = Units

        # Derive structural parameters.
        self._structure()

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

        # self.jhalo = self.LAMBDA * sqrt(self.u.G) * self.M200**(1.5) * sqrt(2 * self.R200 / fc(self.CC))
    
        self.halo_spinfactor = 1.5 * self.LAMBDA * sqrt(2 * self.CC / fc(self.CC))
        fac = pow(log(1 + self.CC) - self.CC / (1 + self.CC), 1.5) / gc(self.CC)
        self.halo_spinfactor = self.halo_spinfactor * fac

def gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction, HubbleParam, 
                    BoxSize, AddBackgroundGrid, OutputDir, OutputFile, Units):
    
    return pyMND_param(CC,
                       V200,
                       LAMBDA,
                       N_HALO,
                       N_GAS,
                       MGH,
                       GasHaloSpinFraction,
                       HubbleParam,
                       BoxSize,
                       AddBackgroundGrid,
                       OutputDir,
                       OutputFile,
                       Units)

if __name__ == '__main__':
    CC = 11.0
    V200 = 163.
    LAMBDA = 0.035
    N_GAS = 208333
    N_HALO = 375000
    MGH = 0.1
    GasHaloSpinFraction = 1.0
    HubbleParam = 1.0
    BoxSize=1200.0
    AddBackgroundGrid = 16
    OutputDir = './'
    OutputFile = 'MW_ICs'

    u = pyMND_units(1.0)

    p = gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction, HubbleParam, BoxSize, 
                        AddBackgroundGrid, OutputDir, OutputFile, u)

