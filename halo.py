import numpy as np
import arepo
from util import rejection_sample
from tqdm import tqdm

class Hernquist(object):
    def __init__(self, M, a,
                 UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e43, UnitVelocity_in_cm_per_s=1e5):
        self._init_units_(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s)

        self.M = M
        self.a = a

        self.density_prefactor = self.M / (2. * np.pi * self.a**3)
        self.vg = (self.G * self.M / self.a)**(0.5)
        self.f_prefactor = self.M / (8. * np.sqrt(2) * np.pi**3 * self.a**3 * self.vg**3)
        self.g_prefactor = (2. * np.sqrt(2) * np.pi**2 * self.a**3 * self.vg) / (3.)
        self.phi_of_0 = - self.G * self.M / self.a

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

    def potential(self, r):
        ans = np.add(r, self.a)
        ans = np.divide(-self.G * self.M, ans)
        return ans

    def mass_enclosed(self, r):
        rt = np.divide(r, self.a)
        ans = np.multiply(np.power(rt, 2.), np.add(rt, 1.), -2.)
        ans = np.multiply(ans, self.M)
        return ans

    def _f_of_q_close_to_1_(self, q):
        # appears to be a lot wrong in this eqn in Hernquist 1990
        prefactor = 3 * self.M / (16 * np.sqrt(2) * np.pi**2 * self.a**3 * self.vg**3)
        a = np.power(np.subtract(1., np.square(q)), 5./2.)
        ans = np.multiply(32./(5.*np.pi), a)
        ans = np.subtract(1., ans)
        ans = np.divide(ans, a)
        return np.multiply(ans, prefactor)

    def f_of_q(self, q):
        # Equation 17 of Hernquist 1990
        qsquared = np.square(q)
        oneminusqsquared = np.subtract(1., qsquared)

        # 3 * arcsin(q)
        term1 = np.multiply(3., np.arcsin(q))

        # q * sqrt(1-q**2) * (1 - 2q**2) * (8*q**4 - 8q**2 -3)
        term2 = np.multiply(8., np.subtract(np.square(qsquared), qsquared))
        np.subtract(term2, 3., out=term2)
        np.multiply(np.sqrt(oneminusqsquared), term2, out=term2)
        np.multiply(q, term2, out=term2)
        np.multiply(np.subtract(oneminusqsquared, qsquared), term2, out=term2)

        np.add(term1, term2, out=term1)

        np.multiply(term1, np.power(oneminusqsquared, -5./2.), out=term1)

        ans = np.multiply(term1, self.f_prefactor)

        return ans

    def f_of_E(self, E):
        keys = np.where(np.logical_and(E>0, E<1e-8))[0]
        keys_ = np.where(E > 1e-2)[0]
        if len(keys_) > 0:
            print(E[keys_])

        E[keys] = -1e-8
        q = self.q_of_E(E)
        return self.f_of_q(q)
    
    def f_of_vr(self, v, r=0.0):
        pot = self.potential(r)
        kin = np.multiply(0.5, np.square(v))
        return self.f_of_E(np.add(pot, kin))

    def my_f_of_vr(self, v, r=0.0):
        return np.multiply(self.f_of_vr(v, r), np.square(v))

    def g_of_q(self, q):
        # Equation 23 of Hernquist 1990
        qsquared = np.square(q)

        # term1 = 3 * (8 * q**4 - 4 * q**2 + 1) * np.arccos(q)
        term1 = np.multiply(8., np.square(qsquared))
        np.subtract(term1, np.multiply(4., qsquared), out=term1)
        np.add(term1, 1., out=term1)
        np.multiply(term1, np.arccos(q), out=term1)
        np.multiply(term1, 3., out=term1)


        # term2 = q * (1 - q**2)**(1/2) *(4*q**2 - 1) * (2*q**2 + 3)
        term2 = np.sqrt(np.subtract(1., qsquared))
        np.multiply(term2, np.subtract(np.multiply(4., qsquared), 1.), out=term2)
        np.multiply(term2, np.add(np.multiply(2., qsquared), 3.), out=term2)
        np.multiply(term2, q, out=term2)

        np.subtract(term1, term2, out=term1)
        np.multiply(term1, np.power(q, -5.), out=term1)

        return np.multiply(term1, self.g_prefactor)

    def _dMdE_close_to_1_(self, q):
        # appears to be a factor of 0.5 wrong in Hernquist 1990
        prefactor = 0.5 * (32./35.) * (self.M / self.vg**2)
        return np.multiply(prefactor, np.subtract(1., np.square(q)))

    def _dMdE_close_to_0_(self, q):
        prefactor = (16./5.) * (self.M/self.vg**2)
        ans = np.multiply(18./7., np.square(q))
        ans = np.subtract(1., ans)
        return np.multiply(prefactor, ans)

    def dMdE(self, E, convert_to_q=True):
        # TODO: redo this because I think it's not working the way I want
        # but got draw_energies to give reasonable samples
        if convert_to_q:
            q = self.q_of_E(E)
        else:
            q = E

        ans = np.multiply(self.f_of_q(q), self.g_of_q(q))

        # now revert to Taylor series for q close to 1
        keys = np.where(np.longdouble(np.subtract(1., q)) < 1e-4)[0]
        ans[keys] = self._dMdE_close_to_1_(q[keys])

        # now revert to Taylor series for q close to 0
        keys = np.where(q < 1e-3)[0]
        ans[keys] = self._dMdE_close_to_0_(q[keys])

        return ans

    def E_of_q(self, q):
        return np.multiply(self.phi_of_0, np.square(q))

    def q_of_E(self, E):
        return np.sqrt(np.divide(E, self.phi_of_0))

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

    def draw_energies(self, r):
        energies = []
        pot_list = self.potential(r)
        maxval_list = self.f_of_E(pot_list)
        for pot, maxval in zip(tqdm(pot_list), maxval_list):
            sample = rejection_sample(self.f_of_E, maxval, 1, xrng=[pot, 0])
            energies.append(float(sample))

        return np.array(energies)

    def draw_speeds(self, r):
        speeds = []
        pot_list = self.potential(r)
        vmax_list = np.sqrt(np.multiply(2., np.abs(pot_list)))

        maxval_list = self.my_f_of_vr(vmax_list/2., r) * 10 

        for this_r, vmax in zip(tqdm(r), vmax_list):
            vlist = np.linspace(0, vmax, 100)
            maxval = np.nanmax(self.my_f_of_vr(vlist, this_r)) * 2
            sample = rejection_sample(self.my_f_of_vr, maxval, 1, xrng=[0, vmax], fn_args={'r': this_r})
            speeds.append(float(sample))

        return np.array(speeds)

    def old_draw_velocities(self, pos):
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

    def draw_velocities(self, pos):
        r = np.linalg.norm(pos, axis=1)
        N = len(r)

        speeds = self.draw_speeds(r)

        theta = np.arccos(np.subtract(1., np.multiply(2., np.random.rand(N))))
        phi = np.multiply(np.random.rand(N), 2.*np.pi)

        # vx = v * sin(theta) * cos(phi)
        # vy = v * sin(theta) * sin(phi)
        # vz = v * cos(theta)
        stheta = np.sin(theta)
        vx = np.multiply(np.multiply(speeds, stheta), np.cos(phi))
        vy = np.multiply(np.multiply(speeds, stheta), np.sin(phi))
        vz =             np.multiply(speeds, np.cos(theta))
        
        return np.transpose([vx, vy, vz])

    def gen_ics(self, N, fname):
        ics = arepo.ICs(fname, [0, N, 0, 0, 0, 0], masses=[0, self.M/N, 0, 0, 0, 0])

        pos = self.draw_coordinates(N)
        vel = self.draw_velocities(pos)

        ics.part1.pos[:] = pos
        ics.part1.vel[:] = vel
        ics.part1.id[:] = np.arange(N) + 1
        ics.write()

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    a = 25 # kpc
    M = 1E12/1E10 # 1E10 Msun
    pot = Hernquist(M, a)
    
    # rlist = pot.a * np.linspace(0, 10, 1000)
    # sigmasq = pot._sigmasq_(rlist)

    Elist = np.linspace(0.9999999, 0.000000001, 10000, dtype=np.longdouble) * pot.phi_of_0
    qlist = pot.q_of_E(Elist)
    g = pot.g_of_q(qlist)
    f = pot.f_of_q(qlist)
    
    dMdE = pot.dMdE(Elist)

    r = pot.a
    this_pot = pot.potential(r)
    vmax = np.sqrt(-2 * this_pot)
    vlist = np.linspace(0, vmax, 1000)
    myf = pot.my_f_of_vr(vlist, r)
    plt.plot(vlist, myf)

    x = Elist/(np.abs(pot.phi_of_0))
    # plt.plot(x, dMdE * np.abs(pot.phi_of_0)/pot.M)
    # plt.axhline(np.abs(pot.phi_of_0) * 16. / (5. * pot.vg**2))
    # # plt.plot(x, np.abs(pot.phi_of_0) * 16. / (5. * pot.vg**2) * np.exp(2*x))
    # # plt.plot(np.abs(Elist/pot.phi_of_0), f * (pot.G*pot.M*pot.a)**(3/2)/M)
    # # plt.plot(Elist/np.abs(pot.phi_of_0), g / (pot.a**2 * np.sqrt(pot.G*pot.M*pot.a)))
    # plt.yscale('log')
    # plt.xlim(-1, 0)
    # # plt.ylim(1E-7, 1E4)
    # plt.ylim(1E-3, 1E1)
    # plt.show()

    # energies = pot.draw_energies(int(1E7))
    # # plt.hist(energies/np.abs(pot.phi_of_0), bins=100)
    # plt.hist(energies, bins=100)
    # plt.yscale('log')
    # plt.show()

    N = int(1E5)
    # pot.gen_ics(N, 'ics.hdf5')
    # pos = pot.draw_coordinates(N)
    # r = np.linalg.norm(pos, axis=1)
    # speeds = pot.draw_speeds(r)
    # energies = pot.draw_energies(r)
    # speeds, vel = pot.draw_velocities(pos)
    # vmag = np.linalg.norm(vel, axis=1)

    # N = int(1E4)
    # pos = pot.draw_coordinates(N)

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
# 
