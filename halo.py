import numpy as np
import arepo

class Hernquist(object):
    def __init__(self, M, a,
                 UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e43, UnitVelocity_in_cm_per_s=1e5):
        self._init_units_(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s)

        self.M = M
        self.a = a

        self.density_prefactor = self.M / (2. * np.pi * self.a**3)

    def _init_units_(self, UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s):
        self.UnitLength_in_cm = UnitLength_in_cm
        self.UnitMass_in_g = UnitMass_in_g
        self.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s

        self._SEC_PER_GIGAYEAR_ = 3.15576e16
        self._SEC_PER_MEGAYEAR_ = 3.15576e13
        self._SEC_PER_YEAR_     = 3.15576e7

        self.UnitTime_in_s      = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        self.UnitTime_in_Megayears = self.UnitTime_in_s / self._SEC_PER_MEGAYEAR_

        self._GRAVITY_ = 6.6738e-8

        self.G = self._GRAVITY_ * UnitLength_in_cm**(-3.) * UnitMass_in_g * self.UnitTime_in_s**(2.)

    def density(self, r):
        rt = np.divide(r, self.a)
        ans = np.multiply(np.power(rt, -1.), np.power(np.add(rt, 1.), -3.))
        ans = np.multiply(ans, self.density_prefactor)
        return ans

    def mass_enclosed(self, r):
        rt = np.divide(r, self.a)
        ans = np.multiply(np.power(rt, 2.), np.add(rt, 1.), -2.)
        ans = np.multiply(ans, self.M)
        return ans

    def draw_radii(self, N):
        f = np.random.rand(N)
        sqrtf = np.sqrt(f)
        
        #f = fenclosed
        #r/a = sqrt(f) / (1-sqrt(f))
        
        rt = np.divide(sqrtf, np.subtract(1., sqrtf))
        return np.multiply(rt, self.a)

    def draw_coordinates(self, N):
        r = self.draw_radii(N)

        theta = np.arccos(np.subtract(1., np.multiply(2., np.random.rand(N))))
        phi = np.multiply(np.random.rand(N), 2.*np.pi)

        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        stheta = np.sin(theta)
        x = np.multiply(np.multiply(r, stheta), np.cos(phi))
        y = np.multiply(np.multiply(r, stheta), np.sin(phi))
        z =             np.multiply(r, np.cos(theta))

        return np.transpose([x, y, z])

    def _sigmasq_(self, r):
        prefactor = self.G * self.M / (12. * self.a)
        rt = np.divide(r, self.a)
        oneplusrt = np.add(1.0, rt)

        # ans1 = 12 * rt * (1+rt)**3 * np.log(1+1./rt)
        ans1 = np.log(np.add(1., np.divide(1., rt)))
        np.multiply(ans1, np.power(oneplusrt, 3.0), out=ans1)
        np.multiply(ans1, np.multiply(12.0, rt), out=ans1)

        # ans2 = (rt/(1+rt)) * (25 + 52*rt + 42*rt**2 + 12*rt**3)
        ans2 = np.add(25.0, np.multiply(52.0, rt))
        np.add(ans2, np.multiply(42.0, np.square(rt)), out=ans2)
        np.add(ans2, np.multiply(12.0, np.power(rt, 3.0)), out=ans2)
        np.multiply(ans2, np.divide(rt, oneplusrt), out=ans2)

        ans1 = np.multiply(prefactor, np.subtract(ans1, ans2))

        # check if there are any nans, which just might indicate r=0
        keys = np.where(np.isnan(ans1))
        for i in keys:
            if 0 <= rt[i] < 1e-12:
                ans1[i] = 0.0

        # if rt > 300, revert to the analytical form for sigma sq because the numerical eqn breaks down
        keys = np.where(rt > 300.0)[0]
        r_gtr = np.multiply(rt[keys], self.a)
        ans1[keys] = np.divide(self.G*self.M/5.0, r_gtr)

        return ans1

    def _vesc_sq_(self, r):
        ans = 2. * self.G * self.M
        return np.divide(ans, np.add(r, self.a))

    def draw_velocities(self, pos):
        r = np.linalg.norm(pos, axis=1)

        N = len(r)
        mean = np.zeros(N)
        sigma = np.sqrt(self._sigmasq_(r))

        vr = np.random.normal(mean, sigma)
        vphi = np.random.normal(mean, sigma)
        vtheta = np.random.normal(mean, sigma)

        # redraw any which exceed 0.95 * vesc
        velsq = np.add(np.add(np.square(vr), np.square(vphi)), np.square(vtheta))
        vesc_sq = self._vesc_sq_(r)
        keys = np.where(velsq > np.multiply(0.95, vesc_sq))[0]
        for i in keys:
            vmagsq = vesc_sq[i]
            while(vmagsq > 0.95 * vesc_sq[i]):
                vr_, vphi_, vtheta_ = np.random.normal(mean[i], sigma[i], size=3)
                vmagsq = vr_**2 + vphi_**2 + vtheta_**2
            vr[i] = vr_
            vphi[i] = vphi_
            vtheta[i] = vtheta_

        # in the future, when rotation is added it will be necessary to actually convert
        # from spherical to cartesian, but it is not clear to me what the correct decisions
        # that need to be made are
        # so for now, since isotropic, can just substitute vx=vr, vy=...
        return np.transpose([vr, vphi, vtheta])

    def gen_ics(self, N, fname):
        ics = arepo.ICs(fname, [0, N, 0, 0, 0, 0], masses=[0, self.M/N, 0, 0, 0, 0])

        pos = self.draw_coordinates(N)
        vel = self.draw_velocities(pos)

        ics.part1.pos[:] = pos
        ics.part1.vel[:] = vel
        ics.part1.id[:] = np.arange(N) + 1
        ics.write()

if __name__ == '__main__':
    a = 25 # kpc
    M = 1E12/1E10 # 1E10 Msun
    pot = Hernquist(M, a)
    
    # rlist = pot.a * np.linspace(0, 10, 1000)
    # sigmasq = pot._sigmasq_(rlist)

    N = int(1E6)
    pot.gen_ics(N, 'ics.hdf5')

    # N = int(1E4)
    # pos = pot.draw_coordinates(N)

    # import matplotlib.pyplot as plt 
    # plt.scatter(pos[:,0], pos[:,1], s=1)
    # plt.xlim(-80, 80)
    # plt.ylim(-80, 80)
    # plt.show()

    # r = np.linalg.norm(pos, axis=1)
    # rbins = np.linspace(0, 100, 20)
    # Menclosed = []
    # for this_r in rbins:
    #   Menclosed.append(len(np.where(r < this_r)[0]))
    # Menclosed = np.array(Menclosed)/N

    # plt.plot(rbins, Menclosed)
    # plt.plot(rbins, rbins**2 / (rbins**2 + a**2))
# 
