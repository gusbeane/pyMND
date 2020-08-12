from numba.types import float64, int64, unicode_type
from numba.experimental import jitclass

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
              ('OutputFile', unicode_type)]

@jitclass(spec_param)
class pyMND_param(object):
    def __init__(self, CC, V200, LAMBDA, N_HALO, N_GAS, MGH,
                 GasHaloSpinFraction, HubbleParam, BoxSize, AddBackgroundGrid,
                 OutputDir, OutputFile):
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

def gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction, HubbleParam, 
                    BoxSize, AddBackgroundGrid, OutputDir, OutputFile):
    
    print(N_GAS)
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
                       OutputFile)

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

    p = gen_pyMND_param(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, GasHaloSpinFraction, HubbleParam, BoxSize, 
                        AddBackgroundGrid, OutputDir, OutputFile)

