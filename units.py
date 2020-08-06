class pyMND_units(object):
    def __init__(self, HubbleParam,
                       UnitLength_in_cm=3.085678e21, 
                       UnitMass_in_g=1.989e43, 
                       UnitVelocity_in_cm_per_s=1e5):
        # Set physical constants.
        self._set_physical_constants()

        # Set user-specified units.
        self.UnitLength_in_cm = UnitLength_in_cm
        self.UnitMass_in_g = UnitMass_in_g
        self.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s

        # Set derived units.
        self.UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        self.UnitTime_in_Megayears = self.UnitTime_in_s / self.SEC_PER_MEGAYEAR

        self.UnitDensity_in_cgs = self.UnitMass_in_g / pow(self.UnitLength_in_cm, 3)
        self.UnitPressure_in_cgs = self.UnitMass_in_g / self.UnitLength_in_cm / pow(self.UnitTime_in_s, 2)
        self.UnitCoolingRate_in_cgs = self.UnitPressure_in_cgs / self.UnitTime_in_s
        self.UnitEnergy_in_cgs = self.UnitMass_in_g * pow(self.UnitLength_in_cm, 2) / pow(self.UnitTime_in_s, 2)

        self.G = self.GRAVITY * self.UnitLength_in_cm**(-3.) * \
                           self.UnitMass_in_g * self.UnitTime_in_s**(2.)
        
        self.HubbleParam = HubbleParam
        self.H0 = self.HubbleParam * 100 * 1e5 / self.CM_PER_MPC / self.UnitVelocity_in_cm_per_s * self.UnitLength_in_cm

        self.TFLOOR = 1e4 # Temperature floor in kelvin
    
    def _set_physical_constants(self):
        self.GAMMA = 5./3.
        self.GAMMA_MINUS1 = self.GAMMA-1.
        self.PI = 3.1415926
        self.GRAVITY = 6.672e-8
        self.SOLAR_MASS = 1.989e33
        self.SOLAR_LUM = 3.826e33
        self.RAD_CONST = 7.565e-15
        self.AVOGADRO = 6.0222e23
        self.BOLTZMANN = 1.3806e-16
        self.GAS_CONST = 8.31425e7
        self.C = 2.9979e10
        self.PLANCK = 6.6262e-27
        self.CM_PER_MPC = 3.085678e24
        self.PROTONMASS = 1.6726e-24
        self.ELECTRONMASS = 9.10953e-28
        self.THOMPSON = 6.65245e-25
        self.ELECTRONCHARGE = 4.8032e-10
        self.SEC_PER_MEGAYEAR = 3.155e13
        self.SEC_PER_YEAR = 3.155e7
        self.HYDROGEN_MASSFRAC = 0.76
        self.SEC_PER_GIGAYEAR = 3.15576e16
        self.SEC_PER_MEGAYEAR = 3.15576e13
        self.SEC_PER_YEAR     = 3.15576e7
        self.GRAVITY = 6.6738e-8

if __name__ == '__main__':
    u = pyMND_units(1.0)
