from math import log, sqrt, exp
import numpy as np

from numba.types import float64, int64, unicode_type
from numba.experimental import jitclass

from .util import fc, gc
from .util import bessi0, bessk0, bessi1, bessk1
from .units import pyMND_units
from .halo import _halo_mass_enclosed

spec_param = [('CC', float64),
              ('V200', float64),
              ('LAMBDA', float64),
              ('N_HALO', int64),
              ('N_GAS', int64),
              ('N_DISK', int64),
              ('MD', float64),
              ('JD', float64),
              ('MGH', float64),
              ('DiskHeight', float64),
              ('GasHaloSpinFraction', float64),
              ('RadialDispersionFactor', float64),
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
              ('jdisk', float64),
              ('H', float64),
              ('Z0', float64),
              ('A', float64),
              ('halo_spinfactor', float64),
              # Auxiliary paramters.
              ('Theta', float64),
              ('RMASSBINS', int64),
              ('ZMASSBINS', int64),
              ('PHIMASSBINS', int64),
              ('RSIZE', int64),
              ('ZSIZE', int64)]

@jitclass(spec_param)
class pyMND_param(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, N_DISK, MD, JD, MGH,
                 DiskHeight,
                 GasHaloSpinFraction, RadialDispersionFactor, HubbleParam, BoxSize, AddBackgroundGrid,
                 OutputDir, OutputFile, Units):
        self.CC = CC
        self.V200 = V200
        self.LAMBDA = LAMBDA
        self.N_HALO = N_HALO
        self.N_GAS = N_GAS
        self.N_DISK = N_DISK
        self.MD = MD
        self.JD = JD
        self.MGH = MGH
        self.DiskHeight = DiskHeight
        self.GasHaloSpinFraction = GasHaloSpinFraction
        self.RadialDispersionFactor = RadialDispersionFactor
        self.HubbleParam = HubbleParam
        self.BoxSize = BoxSize
        self.AddBackgroundGrid = AddBackgroundGrid
        self.OutputDir = OutputDir
        self.OutputFile = OutputFile
        self.u = Units

        # Derive structural parameters.
        self._structure()

        # Set some auxilliary parameters.
        self._auxilliary()

    def _structure(self):
        self.M200 = self.V200**3. / (10 * self.u.G * self.u.H0)
        self.R200 = self.V200 / (10 * self.u.H0)

        self.RS = self.R200 / self.CC
        self.RHO_0 = self.M200 / (4. * self.u.PI * (log(1+self.CC) - self.CC/(1.+self.CC)) * self.RS**3.)

        self.M_DISK = self.MD * self.M200
        self.M_BULGE = 0.
        self.M_GASHALO = self.MGH * self.M200

        self.M_HALO = self.M200 - self.M_DISK - self.M_BULGE - self.M_GASHALO

        self.RH = self.RS * sqrt(2. * (log(1+self.CC) - self.CC / (1. + self.CC)))

        self.jhalo = self.LAMBDA * sqrt(self.u.G) * self.M200**(1.5) * sqrt(2 * self.R200 / fc(self.CC))
        self.jdisk = self.JD * self.jhalo
    
        self.halo_spinfactor = 1.5 * self.LAMBDA * sqrt(2 * self.CC / fc(self.CC))
        fac = pow(log(1 + self.CC) - self.CC / (1 + self.CC), 1.5) / gc(self.CC)
        self.halo_spinfactor = self.halo_spinfactor * fac

        if self.M_DISK > 0.0:
            self._determine_disk_scalelength()
        print(self.H)
        print('M_HALO=', self.M_HALO, 'LAMBDA=', self.LAMBDA, 'M_DISK=', self.M_DISK)
    
    def _auxilliary(self):
        self.RMASSBINS = 2048
        self.ZMASSBINS = 32
        self.PHIMASSBINS = 64

        self.RSIZE = 512
        self.ZSIZE = 512

        self.Theta = 0.35
    
    def _determine_disk_scalelength(self):
        self.H = sqrt(2.0) / 2.0 * self.LAMBDA / fc(self.CC) * self.R200 #/* first guess for disk scale length */

        # print("first guess for disk scale length H= ", self.H)

        self.Z0 = self.DiskHeight * self.H		#/* sets disk thickness for stars */
        # self.A = self.BulgeSize * self.H		#/* sets bulge size */

        dh = self.H

        while(abs(dh)/self.H > 1e-5):
            jd = self._disk_angmomentum() #	/* computes disk momentum */
      
            hnew = (self.jdisk / jd) * self.H
      
            dh = hnew - self.H
      
            if(abs(dh) > 0.5 * self.H):
                dh = 0.5 * self.H * dh / abs(dh)
            else:
                dh = dh * 0.1
      
            self.H = self.H + dh

            # print("Jd/J=", jd/self.jhalo,  "hnew: ", self.H)
      
        self.Z0 = self.DiskHeight * self.H #;	/* sets disk thickness */
        # A = BulgeSize * self.H	#/* sets bulge size */

    def _disk_angmomentum(self, tol=1.48e-08, rtol=1.48e-08):
        xmax = min(30 * self.H, self.R200)
        n = 10
        xlist = np.linspace(0, xmax, n)
        int0 = self._jdisk_int(xlist)
        int0 = np.trapz(self._jdisk_int(xlist), xlist)
        n *= 2
        xlist = np.linspace(0, xmax, n)
        int1 = np.trapz(self._jdisk_int(xlist), xlist)
        while abs(int1-int0) > tol or abs((int1-int0)/int1) > rtol:
            int0 = int1
            n *= 2
            xlist = np.linspace(0, xmax, n)
            int1 = np.trapz(self._jdisk_int(xlist), xlist)
        
        return self.M_DISK * int1

    def _jdisk_int(self, x):

        vc2 = np.zeros(len(x))
        vc = np.zeros(len(x))

        Sigma0 = (self.M_DISK) / (2 * self.u.PI * self.H * self.H)
        for i in range(len(x)):
            if(x[i] > 1e-20):
                # TODO: add bulge
                # vc2 = self.u.G * (_halo_mass_enclosed(x, self.M_HALO, self.RH, self.u) + mass_cumulative_bulge(x)) / x
                menc = _halo_mass_enclosed(x[i], self.M_HALO + self.M_GASHALO, self.RH, self.u)
                vc2[i] = self.u.G * menc / x[i]

            y = x[i] / (2 * self.H)

            if(y > 1e-4):
                vc2[i] = vc2[i] + x[i] * 2 * self.u.PI * self.u.G * Sigma0 * y * (bessi0(y) * bessk0(y) - bessi1(y) * bessk1(y))

            vc[i] = sqrt(vc2[i])

        return np.power(x / self.H, 2) * vc * np.exp(-x / self.H)


def gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, N_DISK, MD, JD, MGH, DiskHeight, GasHaloSpinFraction, 
                    RadialDispersionFactor, HubbleParam, 
                    BoxSize, AddBackgroundGrid, OutputDir, OutputFile, Units):
    
    return pyMND_param(CC,
                       V200,
                       LAMBDA,
                       N_HALO,
                       N_GAS,
                       N_DISK,
                       MD,
                       JD,
                       MGH,
                       DiskHeight,
                       GasHaloSpinFraction,
                       RadialDispersionFactor,
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
    N_DISK = 100000
    MD = 0.048
    JD = 0.052
    MGH = 0.1
    DiskHeight = 0.12
    GasHaloSpinFraction = 1.0
    RadialDispersionFactor = 1.0
    HubbleParam = 1.0
    BoxSize=1200.0
    AddBackgroundGrid = 16
    OutputDir = './'
    OutputFile = 'MW_ICs'

    u = pyMND_units(1.0)

    p = gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, N_DISK, MD, JD, MGH, DiskHeight,
                        GasHaloSpinFraction, RadialDispersionFactor, HubbleParam, BoxSize, 
                        AddBackgroundGrid, OutputDir, OutputFile, u)

