import numpy as np
import arepo
from util import rejection_sample
from tqdm import tqdm
from units import pyMND_units

u = pyMND_units()

def Hernquist_density(r, M, a):
    density_prefactor = M / (2. * n.pi * a**3)
    rt = np.divide(r, a)
    ans = np.multiply(np.power(rt, -1.), np.power(np.add(rt, 1.), -3.))
    ans = np.multiply(ans, density_prefactor)
    return ans

def Hernquist_potential(r, M, a):
    ans = np.add(r, a)
    ans = np.divide(-u.G * M, ans)
    return ans


class Hernquist(object):
    def __init__(self, M, a,
                 UnitLength_in_cm=3.085678e21, UnitMass_in_g=1.989e43, UnitVelocity_in_cm_per_s=1e5):
        self._init_units_(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s)

        self.M = M
        self.a = a

        self.density_prefactor = self.M / (2. * np.pi * self.a**3)
        self.vg = (u.G * self.M / self.a)**(0.5)
        self.f_prefactor = self.M / (8. * np.sqrt(2) * np.pi**3 * self.a**3 * self.vg**3)
        self.g_prefactor = (2. * np.sqrt(2) * np.pi**2 * self.a**3 * self.vg) / (3.)
        self.phi_of_0 = - u.G * self.M / self.a

    

    def mass_enclosed(self, r):
        rt = np.divide(r, self.a)
        ans = np.multiply(np.power(rt, 2.), np.add(rt, 1.), -2.)
        ans = np.multiply(ans, self.M)
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

    def _vesc_sq_(self, r):
        ans = 2. * u.G * self.M
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
    pass