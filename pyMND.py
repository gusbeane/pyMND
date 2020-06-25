from units import pyMND_units
from math import log, sqrt, pow

from scipy.integrate import romberg

class pyMND(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, 
                 HubbleParam,
                 OutputDir, OutputFile):

        self.CC = CC
        self.V200 = V200
        self.LAMBDA = LAMBDA
        self.N_HALO = N_HALO
        self.OutputDir = OutputDir
        self.OutputFile = OutputFile

        self.HubbleParam = HubbleParam

        self.u = pyMND_units(self.HubbleParam)

        self._structure()

    def _structure(self):
        self.M200 = self.V200**3. / (10 * self.u.G * self.u.H0)
        self.R200 = self.V200 / (10 * self.u.H0)

        self.RS = self.R200 / self.CC
        self.RHO_0 = self.M200 / (4. * self.u.PI * (log(1+self.CC) - self.CC/(1.+self.CC)) * self.RS**3.)

        self.M_DISK = 0.
        self.M_BULGE = 0.

        self.M_HALO = self.M200 - self.M_DISK - self.M_BULGE

        self.RH = self.RS * sqrt(2. * (log(1+self.CC) - self.CC / (1. + self.CC)))

        self.jhalo = self.LAMBDA * sqrt(self.u.G) * self.M200**(1.5) * sqrt(2 * self.R200 / self.fc(self.CC))
    
        self.halo_spinfactor = 1.5 * self.LAMBDA * sqrt(2 * self.CC / self.fc(self.CC))
        self.halo_spinfactor *= pow(log(1 + self.CC) - self.CC / (1 + self.CC), 1.5) / self.gc(self.CC)

    def fc(self, c):
        return c * (0.5 - 0.5 / pow(1 + c, 2) - log(1 + c) / (1 + c)) / pow(log(1 + c) - c / (1 + c), 2)
    
    def gc(self, c):
        return romberg(self.gc_int, 0, c)

    def gc_int(self, x):
        return pow(log(1 + x) - x / (1 + x), 0.5) * pow(x, 1.5) / pow(1 + x, 2)

if __name__ == '__main__':
    CC = 11.0
    V200 = 163.
    LAMBDA = 0.035
    N_HALO = 396060
    HubbleParam = 1.0
    OutputDir='./'
    OutputFile='MW_ICs'
    t = pyMND(CC, V200, LAMBDA, N_HALO, HubbleParam, OutputDir, OutputFile)
    