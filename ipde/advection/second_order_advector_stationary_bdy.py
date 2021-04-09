import numpy as np
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from personal_utilities.nufft_interpolation import nufft_interpolation1d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, BoundaryFunction
from ipde.embedded_function import EmbeddedFunction
from fast_interp import interp1d

class SecondOrder_Advector(object):
    """
    General class for semi-lagrangian advection
    """
    def __init__(self, ebdyc, u, v, uo, vo):
        self.ebdyc = ebdyc
        self.u = u
        self.v = v
        self.uo = uo
        self.vo = vo
        self.ux, self.uy = u.gradient()
        self.vx, self.vy = v.gradient()
        self.uxo, self.uyo = uo.gradient()
        self.vxo, self.vyo = vo.gradient()

        self.aap = ebdyc.pnar
        if hasattr(ebdyc, 'danger_zone_list'):
            self.AP_key = ebdyc.register_points(self.aap.x, self.aap.y, dzl=ebdyc.danger_zone_list, gil=ebdyc.guess_ind_list)
        else:
            self.AP_key = ebdyc.register_points(self.aap.x, self.aap.y)
        self.OAP_key = self.AP_key

    def generate(self, dt, fixed_grid=False):
        ebdyc = self.ebdyc
        aap = self.aap
        AP_key = self.AP_key
        OAP_key = self.OAP_key
        u, v = self.u, self.v
        uo, vo = self.uo, self.vo
        ux, uy, vx, vy = self.ux, self.uy, self.vx, self.vy
        uxo, uyo, vxo, vyo = self.uxo, self.uyo, self.vxo, self.vyo

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[AP_key]
        OAEP = ebdyc.registered_partitions[OAP_key]

        # get departure points
        xd_all = np.zeros(aap.N)
        yd_all = np.zeros(aap.N)
        xD_all = np.zeros(aap.N)
        yD_all = np.zeros(aap.N)

        # advect those in the annulus
        c1n, c2n, c3n = AEP.get_Ns()
        oc1n, oc2n, oc3n = OAEP.get_Ns()
        # category 1 and 2
        c1_2 = AEP.zone1_or_2
        oc1_2 = OAEP.zone1_or_2
        fc12 = np.logical_and(c1_2, oc1_2)
        fc12n = np.sum(fc12)

        # extract physical / not annular region
        uxh = ux.extract_pnar()
        uyh = uy.extract_pnar()
        vxh = vx.extract_pnar()
        vyh = vy.extract_pnar()
        uh  = u.extract_pnar()
        vh  = v.extract_pnar()

        uxoh = uxo.extract_pnar()
        uyoh = uyo.extract_pnar()
        vxoh = vxo.extract_pnar()
        vyoh = vyo.extract_pnar()
        uoh  = uo.extract_pnar()
        voh  = vo.extract_pnar()

        SLM = np.zeros([fc12n,] + [4,4], dtype=float)
        SLR = np.zeros([fc12n,] + [4,], dtype=float)

        # solve for departure points
        SLM[:,0,0] = uxh[fc12]
        SLM[:,0,1] = uyh[fc12]
        SLM[:,0,2] = 0.5/dt
        SLM[:,0,3] = 0.0
        SLM[:,1,0] = vxh[fc12]
        SLM[:,1,1] = vyh[fc12]
        SLM[:,1,2] = 0.0
        SLM[:,1,3] = 0.5/dt
        SLM[:,2,0] = 2.0/dt + 3*uxh[fc12]
        SLM[:,2,1] = 3*uyh[fc12]
        SLM[:,2,2] = -uxoh[fc12]
        SLM[:,2,3] = -uyoh[fc12]
        SLM[:,3,0] = 3*vxh[fc12]
        SLM[:,3,1] = 2.0/dt + 3*vyh[fc12]
        SLM[:,3,2] = -vxoh[fc12]
        SLM[:,3,3] = -vyoh[fc12]
        SLR[:,0] = uh[fc12]
        SLR[:,1] = vh[fc12]
        SLR[:,2] = 3*uh[fc12] - uoh[fc12]
        SLR[:,3] = 3*vh[fc12] - voh[fc12]
        OUT = np.linalg.solve(SLM, SLR)
        dx, dy, Dx, Dy = OUT[:,0], OUT[:,1], OUT[:,2], OUT[:,3]
        xd, yd = aap.x[fc12] - dx, aap.y[fc12] - dy
        xD, yD = aap.x[fc12] - Dx, aap.y[fc12] - Dy
        xd_all[fc12] = xd
        yd_all[fc12] = yd
        xD_all[fc12] = xD
        yD_all[fc12] = yD

        # categroy 3 is nonphysical --- with stationary boundary
        # there should be no points in category 3
        fc3n = aap.N - fc12n
        if fc3n > 0:
            raise Exception('You have a stationary boundary but non-physical points onto which you are trying to solve the advection problem.... something has gone wrong')

        self.xd_all = xd_all
        self.yd_all = yd_all
        self.xD_all = xD_all
        self.yD_all = yD_all

    def __call__(self, f, fo):
        ebdyc = self.ebdyc
        # create holding ground
        f_new = EmbeddedFunction(ebdyc, zero=True)
        # semi-lagrangian interpolation
        if hasattr(ebdyc, 'danger_zone_list'):
            fh1 = ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True, dzl=ebdyc.danger_zone_list, gil=ebdyc.guess_ind_list)
            fh2 = ebdyc.interpolate_to_points(fo, self.xD_all, self.yD_all, fix_r=True, dzl=ebdyc.danger_zone_list, gil=ebdyc.guess_ind_list)
        else:
            fh1 = ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True)
            fh2 = ebdyc.interpolate_to_points(fo, self.xD_all, self.yD_all, fix_r=True)
        fh = fh1 + fh2
        f_new['grid'][ebdyc.phys_not_in_annulus[ebdyc.phys]] = fh[:ebdyc.grid_pna_num]
        # set the radial values (needs to be upgraded to a loop!)
        f_new[0][:] = fh[ebdyc.grid_pna_num:].reshape(ebdyc[0].radial_shape)
        # overwrite under grid under annulus by radial grid
        ebdyc.update_radial_to_grid2(f_new)

        return f_new
