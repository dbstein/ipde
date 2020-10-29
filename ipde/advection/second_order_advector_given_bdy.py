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
    def __init__(self, ebdyc, u, v, old_advector, filter_fraction=0.9):
        self.ebdyc = ebdyc
        self.u = u
        self.v = v
        self.ebdyc_old = old_advector.ebdyc
        ### CHECK: DO ALL OF THESE COPIES REALLY NEED TO BE MADE?
        ### NOW THAT I'VE FIXED THE LEAK, SHOULD CHECK...
        self.uo = old_advector.u.copy()
        self.vo = old_advector.v.copy()
        self.ux, self.uy = self.ebdyc.gradient(self.u)
        self.vx, self.vy = self.ebdyc.gradient(self.v)
        self.uxo = old_advector.ux.copy()
        self.uyo = old_advector.uy.copy()
        self.vxo = old_advector.vx.copy()
        self.vyo = old_advector.vy.copy()
        self.filter_fraction = filter_fraction
        del old_advector
    def generate(self, bxs, bys, dt, fixed_grid=False):
        ebdyc = self.ebdyc
        ebdyc_old = self.ebdyc_old
        u, v = self.u, self.v
        uo, vo = self.uo, self.vo
        ux, uy, vx, vy = self.ux, self.uy, self.vx, self.vy
        uxo, uyo, vxo, vyo = self.uxo, self.uyo, self.vxo, self.vyo

        # interpolate the velocity
        ubs = ebdyc.interpolate_radial_to_boundary(u)
        vbs = ebdyc.interpolate_radial_to_boundary(v)

        # move all boundarys; generate new embedded boundaries
        new_ebdys = []
        for ind, ebdy in enumerate(self.ebdyc):
            # generate the new embedded boundary
            bx, by, new_t = arc_length_parameterize(bxs[ind], bys[ind], filter_fraction=self.filter_fraction, return_t=True)
            new_ebdy = ebdy.regenerate(bx, by)
            new_ebdys.append(new_ebdy)


            # # interpolate the velocity
            # ub = ubs.bdy_value_list[ind]
            # vb = vbs.bdy_value_list[ind]
            # ubo_new_parm = self.ubos[ind]
            # vbo_new_parm = self.vbos[ind]
            # # move the boundary with Forward Euler
            # bx = ebdy.bdy.x + 0.5*dt*(3*ub - ubo_new_parm)
            # by = ebdy.bdy.y + 0.5*dt*(3*vb - vbo_new_parm)
            # # repararmetrize the boundary
            # bx, by, new_t = arc_length_parameterize(bx, by, filter_fraction=self.filter_fraction, return_t=True)
            # # move these boundary values for velocity to the new parametrization
            # self.reparmed_ubs.append(nufft_interpolation1d(new_t, np.fft.fft(ub)))
            # self.reparmed_vbs.append(nufft_interpolation1d(new_t, np.fft.fft(vb)))
            # # generate the new embedded boundary
            # new_ebdy = ebdy.regenerate(bx, by)
            # new_ebdys.append(new_ebdy)

        new_ebdyc = EmbeddedBoundaryCollection(new_ebdys)
        # get dnager zone distance
        umax = np.sqrt(u*u + v*v).max()
        ddd = 2*umax*dt
        # raise an exception if danger zone thicker than radial width
        if ddd > new_ebdyc[0].radial_width:
            raise Exception('Velocity is so fast that one timestep oversteps safety zones; reduce timestep.')
        # register the grid...
        if fixed_grid:
            new_ebdyc.register_grid(ebdyc.grid, danger_zone_distance=ddd)
        else:
            new_ebdyc.generate_grid(danger_zone_distance=ddd)

        # let's get the points that need to be interpolated to
        aap = new_ebdyc.pnar
        AP_key  = ebdyc.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        OAP_key = ebdyc_old.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[AP_key]
        OAEP = ebdyc_old.registered_partitions[OAP_key]

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

        # category 1 and 2
        # NOTE:  THESE INTERPOLATIONS CAN BE MADE FASTER BY EXPLOITING SHARED
        #        GRIDPOINTS IF THAT IS ENFORCED IN GRID GENERATION
        #        THIS IS NOT EXPLOITED, FOR THE TIME BEING
        uxh = ebdyc.interpolate_to_points(ux, aap.x, aap.y)
        uyh = ebdyc.interpolate_to_points(uy, aap.x, aap.y)
        vxh = ebdyc.interpolate_to_points(vx, aap.x, aap.y)
        vyh = ebdyc.interpolate_to_points(vy, aap.x, aap.y)
        uh =  ebdyc.interpolate_to_points(u, aap.x, aap.y)
        vh =  ebdyc.interpolate_to_points(v, aap.x, aap.y)

        uxoh = ebdyc_old.interpolate_to_points(uxo, aap.x, aap.y)
        uyoh = ebdyc_old.interpolate_to_points(uyo, aap.x, aap.y)
        vxoh = ebdyc_old.interpolate_to_points(vxo, aap.x, aap.y)
        vyoh = ebdyc_old.interpolate_to_points(vyo, aap.x, aap.y)
        uoh =  ebdyc_old.interpolate_to_points(uo,  aap.x, aap.y)
        voh =  ebdyc_old.interpolate_to_points(vo,  aap.x, aap.y)

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

        # categroy 3... this is the tricky one
        fc3n = aap.N - fc12n
        print('Number of points in category 3 is:', fc3n)
        if fc3n > 0:
            for ind, (ebdy, ebdy_old) in enumerate(zip(ebdyc, ebdyc_old)):
                ub = ubs[ind]
                vb = vbs[ind]

                c3l = AEP.zone3l[ind]
                oc3l = OAEP.zone3l[ind]
                fc3l = np.unique(np.concatenate([c3l, oc3l]))
                th = ebdy.bdy.dt
                tk = ebdy.bdy.k
                def d1_der(f):
                    return np.fft.ifft(np.fft.fft(f)*tk*1j).real
                interp = lambda f: interp1d(0, 2*np.pi, th, f, k=3, p=True)
                bx_interp  = interp(ebdy.bdy.x)
                by_interp  = interp(ebdy.bdy.y)
                bxs_interp = interp(d1_der(ebdy.bdy.x))
                bys_interp = interp(d1_der(ebdy.bdy.y))
                nx_interp  = interp(ebdy.bdy.normal_x)
                ny_interp  = interp(ebdy.bdy.normal_y)
                nxs_interp = interp(d1_der(ebdy.bdy.normal_x))
                nys_interp = interp(d1_der(ebdy.bdy.normal_y))
                urb = ebdy.interpolate_radial_to_boundary_normal_derivative(u[0])
                vrb = ebdy.interpolate_radial_to_boundary_normal_derivative(v[0])
                urrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(u[0])
                vrrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(v[0])
                ub_interp   = interp(ub)
                vb_interp   = interp(vb)
                urb_interp  = interp(urb)
                vrb_interp  = interp(vrb)
                urrb_interp  = interp(urrb)
                vrrb_interp  = interp(vrrb)
                ubs_interp  = interp(d1_der(ub))
                vbs_interp  = interp(d1_der(vb))
                urbs_interp = interp(d1_der(urb))
                vrbs_interp = interp(d1_der(vrb))
                urrbs_interp = interp(d1_der(urrb))
                vrrbs_interp = interp(d1_der(vrrb))

                old_bx_interp  = interp(ebdy_old.bdy.x)
                old_by_interp  = interp(ebdy_old.bdy.y)
                old_bxs_interp = interp(d1_der(ebdy_old.bdy.x))
                old_bys_interp = interp(d1_der(ebdy_old.bdy.y))
                old_nx_interp  = interp(ebdy_old.bdy.normal_x)
                old_ny_interp  = interp(ebdy_old.bdy.normal_y)
                old_nxs_interp = interp(d1_der(ebdy_old.bdy.normal_x))
                old_nys_interp = interp(d1_der(ebdy_old.bdy.normal_y))
                old_ub = ebdy_old.interpolate_radial_to_boundary(uo[0])
                old_vb = ebdy_old.interpolate_radial_to_boundary(vo[0])
                old_urb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(uo[0])
                old_vrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(vo[0])
                old_urrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(uo[0])
                old_vrrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(vo[0])
                # i think the old parm is right, but should think about
                old_ub_interp   = interp(old_ub)
                old_vb_interp   = interp(old_vb)
                old_urb_interp  = interp(old_urb)
                old_vrb_interp  = interp(old_vrb)
                old_urrb_interp  = interp(old_urrb)
                old_vrrb_interp  = interp(old_vrrb)
                old_ubs_interp  = interp(d1_der(old_ub))
                old_vbs_interp  = interp(d1_der(old_vb))
                old_urbs_interp = interp(d1_der(old_urb))
                old_vrbs_interp = interp(d1_der(old_vrb))
                old_urrbs_interp = interp(d1_der(old_urrb))
                old_vrrbs_interp = interp(d1_der(old_vrrb))

                xx = aap.x[fc3l]
                yy = aap.y[fc3l]
                def objective(s, r, so, ro):
                    f = np.empty([s.size, 4])
                    f[:,0] = old_bx_interp(so) + ro*old_nx_interp(so) + 2*dt*ub_interp(s) + 2*dt*r*urb_interp(s) + dt*r**2*urrb_interp(s) - xx
                    f[:,1] = old_by_interp(so) + ro*old_ny_interp(so) + 2*dt*vb_interp(s) + 2*dt*r*vrb_interp(s) + dt*r**2*vrrb_interp(s) - yy
                    f[:,2] = bx_interp(s) + r*nx_interp(s) + 1.5*dt*ub_interp(s) + 1.5*dt*r*urb_interp(s) + 0.75*dt*r**2*urrb_interp(s) - 0.5*dt*old_ub_interp(so) - 0.5*dt*ro*old_urb_interp(so) - 0.25*dt*ro**2*old_urrb_interp(so) - xx
                    f[:,3] = by_interp(s) + r*ny_interp(s) + 1.5*dt*vb_interp(s) + 1.5*dt*r*vrb_interp(s) + 0.75*dt*r**2*vrrb_interp(s) - 0.5*dt*old_vb_interp(so) - 0.5*dt*ro*old_vrb_interp(so) - 0.25*dt*ro**2*old_vrrb_interp(so) - yy
                    return f
                def Jac(s, r, so, ro):
                    J = np.empty([s.size, 4, 4])
                    # derivative with respect to s
                    J[:,0,0] = 2*dt*ubs_interp(s) + 2*dt*r*urbs_interp(s) + dt*r**2*urrbs_interp(s)
                    J[:,1,0] = 2*dt*vbs_interp(s) + 2*dt*r*vrbs_interp(s) + dt*r**2*vrrbs_interp(s)
                    J[:,2,0] = bxs_interp(s) + r*nxs_interp(s) + 1.5*dt*ubs_interp(s) + 1.5*dt*r*urbs_interp(s) + 0.75*dt*r**2*urrbs_interp(s)
                    J[:,3,0] = bys_interp(s) + r*nys_interp(s) + 1.5*dt*vbs_interp(s) + 1.5*dt*r*vrbs_interp(s) + 0.75*dt*r**2*vrrbs_interp(s)
                    # derivative with respect to r
                    J[:,0,1] = 2*dt*urb_interp(s) + 2*dt*r*urrb_interp(s)
                    J[:,1,1] = 2*dt*vrb_interp(s) + 2*dt*r*vrrb_interp(s)
                    J[:,2,1] = nx_interp(s) + 1.5*dt*urb_interp(s) + 1.5*dt*r*urrb_interp(s)
                    J[:,3,1] = ny_interp(s) + 1.5*dt*vrb_interp(s) + 1.5*dt*r*vrrb_interp(s)
                    # derivative with respect to so
                    J[:,0,2] = old_bxs_interp(so) + ro*old_nxs_interp(so)
                    J[:,1,2] = old_bys_interp(so) + ro*old_nys_interp(so)
                    J[:,2,2] = -0.5*dt*old_ubs_interp(so) - 0.5*dt*ro*old_urbs_interp(so) - 0.25*dt*ro**2*old_urrbs_interp(so)
                    J[:,3,2] = -0.5*dt*old_vbs_interp(so) - 0.5*dt*ro*old_vrbs_interp(so) - 0.25*dt*ro**2*old_vrrbs_interp(so)
                    # derivative with respect to ro
                    J[:,0,3] = old_nx_interp(so)
                    J[:,1,3] = old_ny_interp(so)
                    J[:,2,3] = -0.5*dt*old_urb_interp(so) - 0.5*dt*ro*old_urrb_interp(so)
                    J[:,3,3] = -0.5*dt*old_vrb_interp(so) - 0.5*dt*ro*old_vrrb_interp(so)
                    return J
                # take as guess inds our s, r
                s =   AEP.full_t[fc3l]
                r =   AEP.full_r[fc3l]
                so = OAEP.full_t[fc3l]
                ro = OAEP.full_r[fc3l]
                # now solve for sd, rd
                res = objective(s, r, so, ro)
                mres1 = np.hypot(res[:,0], res[:,1]).max()
                mres2 = np.hypot(res[:,2], res[:,3]).max()
                mres = max(mres1, mres2)
                tol = 1e-12
                while mres > tol:
                    J = Jac(s, r, so, ro)
                    d = np.linalg.solve(J, res)
                    s  -= d[:,0]
                    r  -= d[:,1]
                    so -= d[:,2]
                    ro -= d[:,3]
                    res = objective(s, r, so, ro)
                    mres1 = np.hypot(res[:,0], res[:,1]).max()
                    mres2 = np.hypot(res[:,2], res[:,3]).max()
                    mres = max(mres1, mres2)
                r_fail_1 = r.max() > 0.0
                r_fail_2 = r.min() < -ebdy.radial_width
                ro_fail_1 = ro.max() > 0.0
                ro_fail_2 = ro.min() < -ebdy_old.radial_width
                r_fail = r_fail_1 or r_fail_2
                ro_fail = ro_fail_1 or ro_fail_2
                fail = r_fail or ro_fail
                fail_amount = 0.0
                if fail:
                    if r_fail_1:
                        fail_amount = max(fail_amount, r.max())
                        r[r > 0.0] = 0.0
                    if r_fail_2:
                        fail_amount = max(fail_amount, (-r-ebdy.radial_width).max())
                        r[r < -ebdy.radial_width] = -ebdy.radial_width
                    if ro_fail_1:
                        fail_amount = max(fail_amount, ro.max())
                        ro[ro > 0.0] = 0.0
                    if ro_fail_2:
                        fail_amount = max(fail_amount, (-ro-ebdy_old.radial_width).max())
                        ro[ro < -ebdy_old.radial_width] = -ebdy_old.radial_width

                # get the departure points
                xd = bx_interp(s) + nx_interp(s)*r
                yd = by_interp(s) + ny_interp(s)*r
                xD = old_bx_interp(so) + old_nx_interp(so)*ro
                yD = old_by_interp(so) + old_ny_interp(so)*ro
                xd_all[fc3l] = xd
                yd_all[fc3l] = yd
                xD_all[fc3l] = xD
                yD_all[fc3l] = yD

        self.new_ebdyc = new_ebdyc
        self.xd_all = xd_all
        self.yd_all = yd_all
        self.xD_all = xD_all
        self.yD_all = yD_all

        return self.new_ebdyc, new_t

    def __call__(self, f, fo):
        new_ebdyc = self.new_ebdyc
        # create holding ground
        f_new = EmbeddedFunction(new_ebdyc)
        f_new.zero()
        # semi-lagrangian interpolation
        fh1 = self.ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        fh2 = self.ebdyc_old.interpolate_to_points(fo, self.xD_all, self.yD_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        fh = fh1 + fh2
        # set the grid values
        # f_new.grid_value[new_ebdyc.phys_not_in_annulus] = fh[:new_ebdyc.grid_pna_num]
        # this is pretty hacky here!
        # f_new.grid_value[new_ebdyc.phys_not_in_annulus[new_ebdyc.phys]] = fh[:new_ebdyc.grid_pna_num]
        f_new['grid'][new_ebdyc.phys_not_in_annulus[new_ebdyc.phys]] = fh[:new_ebdyc.grid_pna_num]
        # set the radial values (needs to be upgraded to a loop!)
        f_new[0][:] = fh[new_ebdyc.grid_pna_num:].reshape(new_ebdyc[0].radial_shape)
        # overwrite under grid under annulus by radial grid
        new_ebdyc.update_radial_to_grid2(f_new)

        return f_new

    def remove_refs(self):
        del self.ebdyc_old
