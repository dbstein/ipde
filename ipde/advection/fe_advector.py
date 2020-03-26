import numpy as np
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from fast_interp import interp1d

class FE_Advector(object):
    """
    General class for semi-lagrangian advection
    """
    def __init__(self, ebdyc, u, v, filter_function=None):
        self.ebdyc = ebdyc
        self.u = u
        self.v = v
        self.ux, self.uy = self.ebdyc.gradient(u)
        self.vx, self.vy = self.ebdyc.gradient(v)
        self.filter_function = filter_function
    def generate(self, dt):
        ebdyc = self.ebdyc
        u, v = self.u, self.v
        ux, uy, vx, vy = self.ux, self.uy, self.vx, self.vy
        # interpolate the velocity
        ubs = ebdyc.interpolate_radial_to_boundary(u)
        vbs = ebdyc.interpolate_radial_to_boundary(v)

        # move all boundarys; generate new embedded boundaries
        new_ebdys = []
        self.reparmed_ubs = []
        self.reparmed_vbs = []
        for ind, ebdy in enumerate(self.ebdyc):
            # interpolate the velocity
            ub = ubs.bdy_value_list[ind]
            vb = vbs.bdy_value_list[ind]
            # move the boundary with Forward Euler
            bx = ebdy.bdy.x + dt*ub
            by = ebdy.bdy.y + dt*vb
            # repararmetrize the boundary
            bx, by, new_t = arc_length_parameterize(bx, by, return_t=True, filter_function=self.filter_function)
            # move these boundary values to the new parametrization
            # This is not necessary for this timestepper, but is used by other
            # timesteppers which use this as a startup!
            # SHOULD I SWITCH THIS TO NUFFT WHEN THAT IS BEING USED?
            bu_interp = interp1d(0, 2*np.pi, ebdy.bdy.dt, ub, p=True)
            bv_interp = interp1d(0, 2*np.pi, ebdy.bdy.dt, vb, p=True)
            self.reparmed_ubs.append(bu_interp(new_t))
            self.reparmed_vbs.append(bv_interp(new_t))
            # generate the new embedded boundary
            new_ebdy = ebdy.regenerate(bx, by)
            new_ebdys.append(new_ebdy)
        new_ebdyc = EmbeddedBoundaryCollection(new_ebdys)
        # get dnager zone distance
        umax = np.sqrt(u*u + v*v).max()
        ddd = 2*umax*dt
        # raise an exception if danger zone thicker than radial width
        if ddd > new_ebdyc[0].radial_width:
            raise Exception('Velocity is so fast that one timestep oversteps safety zones; reduce timestep.')
        new_ebdyc.register_grid(ebdyc.grid, danger_zone_distance=ddd)

        # let's get the points that need to be interpolated to
        aap = new_ebdyc.pnar
        AP_key  = ebdyc.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[AP_key]

        # get departure points
        xd_all = np.zeros(aap.N)
        yd_all = np.zeros(aap.N)

        c1n, c2n, c3n = AEP.get_Ns()
        # category 1 and 2
        c1_2n = c1n + c2n
        c1_2 = AEP.zone1_or_2
        uxh = ebdyc.interpolate_to_points(ux, aap.x, aap.y)
        uyh = ebdyc.interpolate_to_points(uy, aap.x, aap.y)
        vxh = ebdyc.interpolate_to_points(vx, aap.x, aap.y)
        vyh = ebdyc.interpolate_to_points(vy, aap.x, aap.y)
        uh = ebdyc.interpolate_to_points(u, aap.x, aap.y)
        vh = ebdyc.interpolate_to_points(v, aap.x, aap.y)
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
        # categroy 3... this is the tricky one
        if c3n > 0:
            for ind, ebdy in enumerate(ebdyc):
                c3l = AEP.zone3l[ind]
                th = ebdy.bdy.dt
                # th = 2*np.pi/nb
                # tk = np.fft.fftfreq(nb, th/(2*np.pi))
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
                urb = ebdy.interpolate_radial_to_boundary_normal_derivative(u[ind])
                vrb = ebdy.interpolate_radial_to_boundary_normal_derivative(v[ind])
                ub_interp   = interp(ub)
                vb_interp   = interp(vb)
                urb_interp  = interp(urb)
                vrb_interp  = interp(vrb)
                ubs_interp  = interp(d1_der(ub))
                vbs_interp  = interp(d1_der(vb))
                urbs_interp = interp(d1_der(urb))
                vrbs_interp = interp(d1_der(vrb))
                xo = aap.x[c3l]
                yo = aap.y[c3l]
                def objective(s, r):
                    f = np.empty([s.size, 2])
                    f[:,0] = bx_interp(s) + r*nx_interp(s) + dt*ub_interp(s) + dt*r*urb_interp(s) - xo
                    f[:,1] = by_interp(s) + r*ny_interp(s) + dt*vb_interp(s) + dt*r*vrb_interp(s) - yo
                    return f
                def Jac(s, r):
                    J = np.empty([s.size, 2, 2])
                    J[:,0,0] = bxs_interp(s) + r*nxs_interp(s) + dt*ubs_interp(s) + dt*r*urbs_interp(s)
                    J[:,1,0] = bys_interp(s) + r*nys_interp(s) + dt*vbs_interp(s) + dt*r*vrbs_interp(s)
                    J[:,0,1] = nx_interp(s) + dt*urb_interp(s)
                    J[:,1,1] = ny_interp(s) + dt*vrb_interp(s)
                    return J
                # take as guess inds our s, r
                s = AEP.zone3t[ind]
                r = AEP.zone3r[ind]
                # now solve for sd, rd
                res = objective(s, r)
                mres = np.hypot(res[:,0], res[:,1]).max()
                tol = 1e-12
                while mres > tol:
                    J = Jac(s, r)
                    d = np.linalg.solve(J, res)
                    s -= d[:,0]
                    r -= d[:,1]
                    res = objective(s, r)
                    mres = np.hypot(res[:,0], res[:,1]).max()
                # get the departure points
                xd = bx_interp(s) + nx_interp(s)*r
                yd = by_interp(s) + ny_interp(s)*r
                xd_all[c3l] = xd
                yd_all[c3l] = yd

        self.new_ebdyc = new_ebdyc
        self.xd_all = xd_all
        self.yd_all = yd_all

        return self.new_ebdyc

    def __call__(self, f):
        new_ebdyc = self.new_ebdyc
        # create holding ground
        f_new = EmbeddedFunction(new_ebdyc)
        f_new.zero()

        # semi-lagrangian interpolation
        fh = self.ebdyc.interpolate_to_points(f, self.xd_all, self.yd_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        # set the grid values
        f_new.grid_value[new_ebdyc.phys_not_in_annulus] = fh[:new_ebdyc.grid_pna_num]
        # set the radial values (needs to be upgraded to a loop!)
        f_new[0][:] = fh[new_ebdyc.grid_pna_num:].reshape(new_ebdyc[0].radial_shape)
        # overwrite under grid under annulus by radial grid
        new_ebdyc.update_radial_to_grid(f_new)

        return f_new

