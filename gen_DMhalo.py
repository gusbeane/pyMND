from pyMND import pyMND

CC = 11.0
V200 = 163.
LAMBDA = 0.035
N_GAS = 0
N_HALO = 396060
MGH = 0
HubbleParam = 1.0
OutputDir='./'
OutputFile='MW_ICs'
t = pyMND(CC, V200, LAMBDA, N_HALO, N_GAS, MGH, HubbleParam, OutputDir, OutputFile)