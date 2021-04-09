import numpy as np
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from personal_utilities.nufft_interpolation import nufft_interpolation1d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, BoundaryFunction
from ipde.embedded_function import EmbeddedFunction
from fast_interp import interp1d

class FE_Advector(object):
    def __init__(self, ebdyc, u, v):
        self.ebdyc = ebdyc
        self.u = u
        self.v = v
        self.ux, self.uy = self.u.gradient()
        self.vx, self.vy = self.v.gradient()

        self.aap = ebdyc.pnar
        if hasattr(ebdyc, 'danger_zone_list'):
            self.AP_key = ebdyc.register_points(self.aap.x, self.aap.y, dzl=ebdyc.danger_zone_list, gil=ebdyc.guess_ind_list)
        else:
            self.AP_key = ebdyc.register_points(self.aap.x, self.aap.y)

    def generate(self, dt):
        ebdyc = self.ebdyc
        aap = self.aap
        u, v = self.u, self.v
        ux, uy, vx, vy = self.ux, self.uy, self.vx, self.vy

        # extract physical / not annular region
        uxh = ux.extract_pnar()
        uyh = uy.extract_pnar()
        vxh = vx.extract_pnar()
        vyh = vy.extract_pnar()
        uh  = u.extract_pnar()
        vh  = v.extract_pnar()

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[self.AP_key]

        # get departure points
        xd_all = np.zeros(aap.N)
        yd_all = np.zeros(aap.N)

        c1n, c2n, c3n = AEP.get_Ns()
        # category 1 and 2
        c1_2n = c1n + c2n
        c1_2 = AEP.zone1_or_2
        SLM = np.zeros([c1_2n,] + [2,2], dtype=float)
        SLR = np.zeros([c1_2n,] + [2,], dtype=float)
        SLM[:,0,0] = 1 + dt*uxh[c1_2]
        SLM[:,0,1] = dt*uyh[c1_2]
        SLM[:,1,0] = dt*vxh[c1_2]
        SLM[:,1,1] = 1 + dt*vyh[c1_2]
        SLR[:,0] = dt*uh[c1_2]
        SLR[:,1] = dt*vh[c1_2]
        OUT = np.linalg.solve(SLM, SLR)
        xdt, ydt = OUT[:,0], OUT[:,1]
        xd, yd = aap.x[c1_2] - xdt, aap.y[c1_2] - ydt
        xd_all[c1_2] = xd
        yd_all[c1_2] = yd
        # categroy 3 is nonphysical --- with stationary boundary
        # there should be no points in category 3
        if c3n > 0:
            raise Exception('You have a stationary boundary but non-physical points onto which you are trying to solve the advection problem.... something has gone wrong')

        self.xd_all = xd_all
        self.yd_all = yd_all

    def __call__(self, f):
        ebdyc = self.ebdyc

        # create holding ground
        f_new = EmbeddedFunction(ebdyc, zero=True)
        # semi-lagrangian interpolation
        if hasattr(ebdyc, 'danger_zone_list'):
            fh = ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True, dzl=ebdyc.danger_zone_list, gil=ebdyc.guess_ind_list)
        else:
            fh = ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True)
        f_new['grid'][ebdyc.phys_not_in_annulus[ebdyc.phys]] = fh[:ebdyc.grid_pna_num]
        # set the radial values (needs to be upgraded to a loop!)
        f_new[0][:] = fh[ebdyc.grid_pna_num:].reshape(ebdyc[0].radial_shape)
        # overwrite under grid under annulus by radial grid
        ebdyc.update_radial_to_grid2(f_new)

        return f_new



