import numpy as np
import pybie2d
import fast_interp
import finufftpy
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
from near_finder.points_near_curve import gridpoints_near_curve_update
from ipde.heavisides import SlepianMollifier
from ipde.utilities import affine_transformation, get_chebyshev_nodes, SimpleFourierFilter
from qfs.two_d_qfs import QFS_Boundary

def setit(name, dictionary, default):
    return dictionary[name] if name in dictionary else default

# quad weights for the chebyshev nodes i'm using
def fejer_1(n):
    points = -np.cos(np.pi * (np.arange(n) + 0.5) / n)
    N = np.arange(1, n, 2)
    length = len(N)
    m = n - length
    K = np.arange(m)
    v0 = np.concatenate(
        [
            2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K ** 2),
            np.zeros(length + 1),
        ]
    )
    v1 = v0[:-1] + np.conjugate(v0[:0:-1])
    w = np.fft.ifft(v1)
    weights = w.real
    return points, weights

class EmbeddedBoundary(object):
    def __init__(self, bdy, interior, M, h, **kwargs):
        """
        Instantiate an EmbeddedBoundary object

        bdy (required):
            type(GlobalSmoothBoundary) Smooth, closed boundary (see pybie2d doc)
        interior (required):
            bool, is this EmbeddedBoundary is for interior or exterior problem
        M (required):
            number of modes used to describe radial grid
        h (required):
            gridspacing for radial grid (in radial direction)

        accepted kwargs:
            pad_zone:
                extra space, in units of h, to protect boundary from
                the rolloff functions
                (note that the gridspacing is deferred until a grid is registered)
                DEFAULT VALUE: 0
            heaviside:
                regularized Heaviside function (defined on [-Inf, Inf])
                must return 0 for x =< -1, 1 for x >= 1
                DEFAULT VALUE: SlepianMollifier(2*M)
            coordinate_tolerance:
                tolerance used when finding local coordinates
                DEFAULT VALUE: 1e-14
            qfs_tolerance:
                tolerance used when computing effective surfaces for qfs
                DEFAULT VALUE: 1e-12
            qfs_fsuf:
                forced oversampling in qfs (useful for kernels with rapid decay)
                DEFAULT VALUE: None
            coordinate_scheme:
                interpolation scheme used for coordinate solving

        """
        self.bdy = bdy
        self.interior = interior
        self.M = M
        self.h = h
        self.pad_zone             = setit('pad_zone',             kwargs, 0    )
        self.coordinate_tolerance = setit('coordinate_tolerance', kwargs, 1e-10)
        self.qfs_tolerance        = setit('qfs_tolerance',        kwargs, 1e-10)
        self.qfs_fsuf             = setit('qfs_fsuf',             kwargs, None )
        self.coordinate_scheme    = setit('coordinate_scheme',    kwargs, 'nufft')
        # do this one differently to avoid constructing SlepianMollifier if not needed
        self.heaviside = kwargs['heaviside'] if 'heaviside' in kwargs else SlepianMollifier(2*self.M).step
        # parameters that depend on other parameters
        self.radial_width = self.M*self.h
        self.heaviside_width = self.radial_width - self.pad_zone*self.h

        # construct radial grid
        self._generate_radial_grid()

        # construct QFS boundaries
        self._generate_qfs_boundaries()

        # helper for near-radial interpolation
        self.near_radial_guess_inds = \
            np.tile(np.arange(self.bdy.N), self.M).reshape(self.M, self.bdy.N)

        self.kwargs = kwargs

    def regenerate(self, bx, by):
        bdy = GSB(x=bx.copy(), y=by.copy())
        return EmbeddedBoundary(bdy, self.interior, self.M, self.h, **self.kwargs)

    def register_grid(self, grid, close, int_helper1, int_helper2, float_helper, bool_helper, index, danger_zone_distance=None, verbose=False):
        """
        Register a grid object

        grid (required):
            grid of type(Grid) from pybie2d (see pybie2d doc)
        verbose (optional):
            bool, whether to pass verbose output onto gridpoints_near_curve
        """
        self.grid = grid
        
        ########################################################################
        # Locate grid points in annulus/curve and compute coordinates
        
        ddd = danger_zone_distance if danger_zone_distance is not None else 0.0

        # find near points and corresponding coordinates
        out = gridpoints_near_curve_update(self.bdy.x, self.bdy.y, 
            self.grid.xv, self.grid.yv, self.radial_width+ddd, index, close,
            int_helper1, int_helper2, float_helper, bool_helper,
            interpolation_scheme=self.coordinate_scheme,
            tol=self.coordinate_tolerance, verbose=verbose)
        self.near_curve_result = out
        # wrangle output from gridpoints_near_curve
        nclose = out[0]
        iax = out[1]
        iay = out[2]
        r = out[3]
        t = out[4]
        # actually in annulus
        ia_good, _, _ = self.check_if_r_in_annulus(r)
        # reduce to the good ones
        iax_small = iax[ia_good]
        iay_small = iay[ia_good]
        r_small   = r  [ia_good]
        t_small   = t  [ia_good]

        # save coordinate values for those points that are in annulus
        self.grid_ia_xind = iax_small
        self.grid_ia_yind = iay_small
        self.grid_ia_x = self.grid.xv[iax_small]
        self.grid_ia_y = self.grid.yv[iay_small]
        self.grid_ia_r = r_small
        self.grid_ia_t = t_small
        # for use in NUFFT interpolation
        self.grid_ia_r_transf = self.nufft_transform_r(self.grid_ia_r)
        self.interface_x_transf = affine_transformation(self.interface.x, self.grid.x_bounds[0], self.grid.x_bounds[1], 0.0, 2*np.pi, use_numexpr=True)
        self.interface_y_transf = affine_transformation(self.interface.y, self.grid.y_bounds[0], self.grid.y_bounds[1], 0.0, 2*np.pi, use_numexpr=True)

        ########################################################################
        # Compute regularized Heaviside functions for coupling grids

        lb = -self.heaviside_width if self.interior else self.heaviside_width
        grts = affine_transformation(self.grid_ia_r, lb, 0, -1, 1, use_numexpr=True)
        self.grid_to_radial_step = 1.0 - self.heaviside(grts)
        arts = affine_transformation(self.radial_rv, lb, 0, -1, 1, use_numexpr=True)
        self.radial_cutoff = self.heaviside(arts)

        # get indeces for everything in danger zone
        if danger_zone_distance is not None:
            # actually in danger zone
            # should these only those on the phys side or any?  I think any...
            # idz = np.logical_and(r <= ddd, r >= -self.radial_width-ddd) if self.interior \
                        # else np.logical_and(r >= -ddd, r <= self.radial_width+ddd)
            idz = np.logical_and(r <= ddd, r >= -self.radial_width-ddd) if self.interior \
                        else np.logical_and(r >= -ddd, r <= self.radial_width+ddd)
            # reduce to the good ones
            iax_small = iax[idz]
            iay_small = iay[idz]
            r_small   = r  [idz]
            t_small   = t  [idz]
            self.grid_in_danger_zone_x = iax_small
            self.grid_in_danger_zone_y = iay_small
            # get guess indeces for this
            gi = (t_small//self.bdy.dt).astype(int)
            self.grid_in_danger_zone_gi = gi

            # get correct vector needed
            # rr = np.ones(np.prod(self.radial_shape), dtype=bool)
            # self.in_danger_zone = np.concatenate([in_danger_zone[phys_na], rr])
            # self.guess_inds = np.concatenate([gi[phys_na], self.near_radial_guess_inds.ravel()])

    ############################################################################
    # Functions relating to the radial grid

    def register_ia_inds(self, phys_inds):
        """
        Get in_annulus indeces into physical only array
        """
        self.ia_inds = phys_inds[self.grid_ia_xind, self.grid_ia_yind]

    def _generate_radial_grid(self):
        bdy = self.bdy
        X, Y = bdy.x, bdy.y
        N = bdy.N
        NX, NY = bdy.normal_x, bdy.normal_y
        T, DT = bdy.t, bdy.dt
        # generate the inner/outer boundary
        sign = -1 if self.interior else 1
        OX = X + sign*self.radial_width*NX
        OY = Y + sign*self.radial_width*NY
        self.interface = GSB(x=OX, y=OY)
        # generate the boundary fitted grid
        lb = -self.radial_width if self.interior else 0.0
        ub = 0.0 if self.interior else self.radial_width
        rc, rv, rat = get_chebyshev_nodes(lb, ub, self.M)
        self.radial_rv = rv
        self.radial_tv = T
        self.radial_r, self.radial_t = np.meshgrid(
                    self.radial_rv, self.radial_tv, indexing='ij')
        self.radial_x = X + self.radial_r*NX
        self.radial_y = Y + self.radial_r*NY
        # get plotting radial x, y
        # get chebyshev nodes starting at 0 with 1 extra for plotting
        RP = 0.5*(lb+ub) + 0.5*(ub-lb)*np.cos(np.arange(self.M+1)[::-1]/(self.M)*np.pi)
        TP = np.concatenate([T, (2*np.pi,)])
        XP = np.concatenate([X, (X[0],)])
        YP = np.concatenate([Y, (Y[0],)])
        NXP = np.concatenate([NX, (NX[0],)])
        NYP = np.concatenate([NY, (NY[0],)])
        self.plot_radial_x = XP + RP[:,None]*NXP
        self.plot_radial_y = YP + RP[:,None]*NYP
        # max and minimum spatial values for everything drop related
        self._minx = X.min() if self.interior else OX.min()
        self._maxx = X.max() if self.interior else OX.max()
        self._miny = Y.min() if self.interior else OY.min()
        self._maxy = Y.max() if self.interior else OY.max()
        # things for taking derivatives on radial grid
        self.radial_k = np.fft.fftfreq(N, DT/(2*np.pi))
        self.radial_k_filter = SimpleFourierFilter(self.radial_k, 'fraction', fraction=2/3.)
        self.radial_speed = bdy.speed*(1.0 + bdy.curvature*self.radial_r)
        self.inverse_radial_speed = 1.0/self.radial_speed
        # this is used for radial-->grid interpolation
        self.interpolation_hold = np.zeros([2*self.M, N], dtype=float)
        self.interpolation_k = np.fft.fftfreq(2*self.M, 1/(2*self.M))
        self.interpolation_modifier = np.exp(-1j*self.interpolation_k[:,None]*np.pi/self.M/2)
        # shape
        self.radial_shape = (self.M, N)
        # chebshev differentiation matrix
        V0 = np.polynomial.chebyshev.chebvander(rc, self.M-1)
        VI0 = np.linalg.inv(V0)
        DC01 = np.polynomial.chebyshev.chebder(np.eye(self.M)) / rat
        DC00 = np.row_stack([DC01, np.zeros(self.M)])
        self.D00 = V0.dot(DC00.dot(VI0))
        # chebysev interpolation->bdy matrix
        if self.interior:
            self.chebyshev_interp_f_to_bdy = np.polynomial.chebyshev.chebvander(1, self.M-1).dot(VI0)[0]
            self.chebyshev_interp_f_to_interface = np.polynomial.chebyshev.chebvander(-1, self.M-1).dot(VI0)[0]
        else:
            # w = np.polynomial.chebyshev.chebvander(-1, self.M-1).dot(VI0)[0]
            self.chebyshev_interp_f_to_bdy = np.polynomial.chebyshev.chebvander(-1, self.M-1).dot(VI0)[0]
            self.chebyshev_interp_f_to_interface = np.polynomial.chebyshev.chebvander(1, self.M-1).dot(VI0)[0]
        # self.chebyshev_interp_f_to_bdy = w
        # chebyshev interpolation of normal  derivative->bdy matrix
        self.chebyshev_interp_df_dn_to_bdy = self.chebyshev_interp_f_to_bdy.dot(self.D00)
        self.chebyshev_interp_df_dn2_to_bdy = self.chebyshev_interp_f_to_bdy.dot(self.D00.dot(self.D00))
        # compute the approximate radius
        self.bdy_centroid_x = np.mean(self.bdy.x)
        self.bdy_centroid_y = np.mean(self.bdy.y)
        dx = self.bdy.x - self.bdy_centroid_x
        dy = self.bdy.y - self.bdy_centroid_y
        rd = np.sqrt(dx*dx + dy*dy)
        self.approximate_radius = np.mean(rd)

        # get PointSet for integration target
        self.radial_targ = PointSet(self.radial_x.ravel(), self.radial_y.ravel())

        # get quadrature weights for radial grid
        _, w = fejer_1(self.M)
        self.radial_quadrature_weights = self.bdy.dt*w[:,None]*self.radial_width/2.0*self.radial_speed

    def check_if_r_in_annulus(self, r):
        """
        checks if r is in the annulus or not
        Inputs:
            r, float(*): radii to check
        Returns:
            in_annulus, bool(*): whether corresponding r is in the annulus
            pna,        bool(*): physical, not in annulus
            ext,        bool(*): exterior, not in annulus
        """
        if self.interior:
            w1 = r <= 0
            w2 = r >= -self.radial_width
            ia = np.logical_and(w1, w2)
            pna = np.logical_not(w2)
            ext = np.logical_not(w1)
            return ia, pna, ext
        else:
            w1 = r >= 0
            w2 = r <= self.radial_width
            ia = np.logical_and(w1, w2)
            pna = np.logical_not(w2)
            ext = np.logical_not(w1)
            return ia, pna, ext

    def nufft_transform_r(self, r):
        lb = -self.radial_width if self.interior else 0.0
        ub = 0.0 if self.interior else self.radial_width
        return np.arccos(affine_transformation(r, lb, ub, 1.0, -1.0, use_numexpr=True))

    def fix_r(self, r):
        if self.interior:
            r[r > 0.0] = 0.0
        else:
            r[r < 0.0] = 0.0

    ############################################################################
    # Functions for communicating between radial grid and boundary
    def interpolate_radial_to_boundary(self, f):
        return self.chebyshev_interp_f_to_bdy.dot(f)
    def interpolate_radial_to_interface(self, f):
        return self.chebyshev_interp_f_to_interface.dot(f)
    def interpolate_radial_to_boundary_normal_derivative(self, f):
        return self.chebyshev_interp_df_dn_to_bdy.dot(f)
    def interpolate_radial_to_boundary_normal_derivative2(self, f):
        return self.chebyshev_interp_df_dn2_to_bdy.dot(f)
    def interpolate_radial_to_boundary_any_r_derivative(self, f, d):
        if d == 0:
            return self.interpolate_radial_to_boundary(f)
        elif d == 1:
            return self.interpolate_radial_to_boundary_normal_derivative(f)
        elif d == 2:
            return self.interpolate_radial_to_boundary_normal_derivative2(f)
        else:
            DER = np.linalg.matrix_power(D00, d)
            return self.interpolate_radial_to_boundary(DER.dot(f))

    ############################################################################
    # Functions for communicating between radial grid and background grid
    def interpolate_radial_to_points(self, fr, transf_r, t):
        """
        Interpolate the radial function fr defined on this annulus to points
        with coordinates (r, t) given by transf_r, t
        """
        self.interpolation_hold[:self.M,:] = fr
        self.interpolation_hold[self.M:,:] = fr[::-1]
        funch = np.fft.fft2(self.interpolation_hold)*self.interpolation_modifier
        funch[self.M] = 0.0
        out = np.empty(t.size, dtype=complex)
        diagnostic = finufftpy.nufft2d2(transf_r, t, out, 1, 1e-14, funch, modeord=1)
        vals = out.real/np.prod(funch.shape)
        return vals
    def interpolate_radial_to_grid1(self, fr, f):
        """
        Interpolate the function fr onto f in the part of the grid underlying
        This particular embedded boundary

        f here is only the physical portion of the grid values of f
        """
        vals = self.interpolate_radial_to_points(fr, self.grid_ia_r_transf, self.grid_ia_t)
        f[self.grid_ia_xind, self.grid_ia_yind] = vals
    def interpolate_radial_to_grid2(self, fr, f):
        """
        Interpolate the function fr onto f in the part of the grid underlying
        This particular embedded boundary

        f here is only the physical portion of the grid values of f
        """
        vals = self.interpolate_radial_to_points(fr, self.grid_ia_r_transf, self.grid_ia_t)
        f[self.ia_inds] = vals

    ############################################################################
    # FUNCTIONS FOR MERGING (EXPERIMENTAL AND NOT USED)
    # SAVE THIS HERE FOR NOW, BUT WILL REQUIRE EBDY_COLLECTION INTERFACES
    # IF IT IS ACTUALLY GOING TO BE USED
    def full_merge(self, f, fr, order=5):
        f1 = f*self.grid_step
        f2 = f*0.0
        self.interpolate_radial_to_grid(fr, f2)
        f2 = f2*(1-self.grid_step)
        gia = self.grid_in_annulus
        f[gia] = f1[gia] + f2[gia]
        fr1 = fr*self.radial_cutoff[:,None]
        fr2 = self.interpolate_grid_to_radial(f*self.grid_step, order)
        fr[:] = fr1 + fr2
    def merge_grids(self, f, fr):
        f1 = f*self.grid_step
        f2 = f*0.0
        self.interpolate_radial_to_grid(fr, f2)
        f2 = f2*(1-self.grid_step)
        gia = self.grid_in_annulus
        f[gia] = f1[gia] + f2[gia]

    ############################################################################
    # Functions for taking derivatives on the radial grid
    def _radial_grid_t_derivative(self, f):
        return np.fft.ifft(np.fft.fft(f)*1j*self.radial_k).real
    def _radial_grid_tau_derivative(self, f):
        return self._radial_grid_t_derivative(f)*self.inverse_radial_speed
    def _radial_grid_r_derivative(self, f):
        return self.D00.dot(f)
    def radial_grid_derivatives(self, f):
        bdy = self.bdy
        nx, ny = bdy.normal_x,  bdy.normal_y
        tx, ty = bdy.tangent_x, bdy.tangent_y
        ft = self._radial_grid_tau_derivative(f)
        fr = self._radial_grid_r_derivative(f)
        fx = fr*nx + ft*tx
        fy = fr*ny + ft*ty
        return fx, fy
    def radial_grid_laplacian(self, f):
        # the lazy way, should up to the way that uses coordinate formula
        fx, fy = self.radial_grid_derivatives(f)
        fxx, _ = self.radial_grid_derivatives(fx)
        _, fyy = self.radial_grid_derivatives(fy)
        return fxx + fyy
    def gradient(self, fx, fy, fr):
        """
        Compute the gradient on the radial grid
        And then interpolate it onto the portion of the grid underlying
        This particular embedded boundary
        """
        fxr, fyr = self.radial_grid_derivatives(fr)
        self.interpolate_radial_to_grid1(fxr, fx)
        self.interpolate_radial_to_grid1(fyr, fy)
        return fxr, fyr
    def laplacian(self, lapf, fr):
        """
        Compute the laplacian on the radial grid
        And then interpolate it onto the portion of the grid underlying
        This particular embedded boundary
        """
        lapfr = self.radial_grid_laplacian(fr)
        self.interpolate_radial_to_grid1(lapfr, lapf)
        return lapfr

    ############################################################################
    # Functions for converting (u,v) vector fields <--> (r,t) vector fields
    def convert_uv_to_rt(self, fu, fv):
        bdy = self.bdy
        fr = fu*bdy.normal_x  + fv*bdy.normal_y
        ft = fu*bdy.tangent_x + fv*bdy.tangent_y
        return fr, ft
    def convert_rt_to_uv(self, fr, ft):
        bdy = self.bdy
        fu = fr*bdy.normal_x + ft*bdy.tangent_x
        fv = fr*bdy.normal_y + ft*bdy.tangent_y
        return fu, fv

    ############################################################################
    # Construct the QFS boundaries for both the boundary and the interface
    def _generate_qfs_boundaries(self):
        eps  = self.qfs_tolerance
        fsuf = self.qfs_fsuf
        self.bdy_qfs = QFS_Boundary(self.bdy, eps=eps, forced_source_upsampling_factor=2, FF=0.3)
        self.interface_qfs = QFS_Boundary(self.interface, eps=eps, forced_source_upsampling_factor=2, FF=0.3)
        # collect interface relevant sources
        q = self.interface_qfs
        # these are sources for evaluating from interface out of the annulus
        self.interface_grid_source = q.interior_source_bdy if self.interior else q.exterior_source_bdy
        # these are sources for evaluating from interface into the annulus
        self.interface_radial_source = q.exterior_source_bdy if self.interior else q.interior_source_bdy
        # collect boundary relevant sources
        q = self.bdy_qfs
        # these are sources for evaluating from boundary into the exterior region
        self.bdy_outward_source = q.exterior_source_bdy if self.interior else q.interior_source_bdy
        # these are sources for evaluating from boundary into the physical region
        self.bdy_inward_source = q.interior_source_bdy if self.interior else q.exterior_source_bdy

    ############################################################################
    # Functions for integrating
    def radial_integral(self, fr):
        return np.sum(fr*self.radial_cutoff[:,None]*self.radial_quadrature_weights)
