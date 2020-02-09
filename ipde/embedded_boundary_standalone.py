import numpy as np
import pybie2d
import fast_interp
import finufftpy
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
from near_finder.near_routines import gridpoints_near_points, gridpoints_near_curve, points_near_curve, gridpoints_near_curve_sparse
from near_finder.phys_routines import points_inside_curve, points_inside_curve_sparse, points_inside_curve_update
from near_finder.coordinate_routines import compute_local_coordinates
from ipde.heavisides import SlepianMollifier
from ipde.utilities import affine_transformation, get_chebyshev_nodes, SimpleFourierFilter
from qfs.two_d_qfs import QFS_Boundary

class EmbeddedPointPartition(object):
    """
    Partition the points (x, y), which need to have no structure
    Into those points that are inside of the annulus,
    Those points that are not inside of the annulus
    And, if nearly_radial, those that are on the aphysical side of the annulus
    """
    def __init__(self, ebdy, rx, ry, gx, gy, mask):
        """
        For structured x, y
        """
        pass
    def __init__(self, ebdy, x, y, fix_r=False, danger_zone=None, gi=None):
        """
        For Unstructured x, y
        """
        self.ebdy = ebdy
        self.x_check = x
        self.y_check = y
        self.fix_r = fix_r
        self.sh = x.shape
        self.x = x.ravel()
        self.y = y.ravel()
        self.size = self.x.size
        self.bx = self.ebdy.bdy.x
        self.by = self.ebdy.bdy.y
        self.width = self.ebdy.radial_width
        self.interior = self.ebdy.interior
        # danger zone gives those points who need to have r checked
        self.has_danger_zone = danger_zone is not None
        if self.has_danger_zone:
            self.danger_zone = danger_zone
            self.gi = gi
        # find points near boundary
        if self.has_danger_zone:
            res = points_near_curve(self.bx, self.by, self.x, self.y, self.width, self.danger_zone, self.gi)
            # res = points_near_curve_danger(self.bx, self.by, self.x, self.y, self.width, self.danger_zone, self.gi)
        else:
            res = points_near_curve(self.bx, self.by, self.x, self.y, self.width)
        # extract results from near-curve finder
        in_full_annulus, er, et, _ = res
        # find points inside/outside boundary
        w = points_inside_curve(self.x, self.y, res)
        if self.fix_r:
            fixit = in_full_annulus.copy()
            fixit[in_full_annulus] = (er[in_full_annulus] > 0)
            # fixit = np.logical_and(er > 0, in_full_annulus)
            er[fixit] = 0.0
            w[fixit] = True
        if self.interior:
            phys = w
            ext = np.logical_not(w)
        else:
            ext = w
            phys = np.logical_not(w)
        # get the results (1: in annulus; 2: not in annulus, physical; 3: not in annulus, not physical)
        self.category1 = np.logical_and(in_full_annulus, phys)
        self.in_annulus = self.category1
        self.not_in_annulus = np.logical_not(self.in_annulus)
        self.category2 = np.logical_and(self.not_in_annulus, phys)
        self.category3 = np.logical_and(self.not_in_annulus, ext)
        self.category1_N = np.sum(self.category1)
        self.category2_N = np.sum(self.category2)
        self.category3_N = np.sum(self.category3)
        # spot check this
        total = np.sum(self.category1) + np.sum(self.category2) + np.sum(self.category3)
        if total != self.in_annulus.size:
            raise Exception('Categorization of points does not add up')
        # get shortened r and t vectors for those worth interpolating to
        self.good_r = er[self.in_annulus]
        self.good_size = self.good_r.size
        # transform the r values
        lb = -self.width if self.interior else 0.0
        ub = 0.0 if self.interior else self.width
        self.good_r_transf = np.arccos(affine_transformation(self.good_r, lb, ub, 1.0, -1.0, use_numexpr=True))
        # store the theta values
        self.good_t = et[self.in_annulus]

        # shortened r vectors for category 3
        self.test_r = er[self.category3]
        self.test_t = et[self.category3]
        # need the full r, actually....
        self.full_r = er
        self.full_t = et
    def is_it_i(self, x, y, fix_r):
        x_is_i = x is self.x_check
        y_is_i = y is self.y_check
        r_is_i = fix_r == self.fix_r
        return x_is_i and y_is_i and r_is_i
    def get_ia(self):
        return self.in_annulus, self.not_in_annulus
    def get_categories(self):
        return self.category1, self.category2, self.category3
    def get_category_Ns(self):
        return self.category1_N, self.category2_N, self.category3_N
    def get_points_in_category(self, category):
        if category == 1:
            sel = self.category1
        elif category == 2:
            sel = self.category2
        elif category == 3:
            sel = self.category3
        else:
            raise Exception('Category must be 1, 2, or 3.')
        return self.x[sel], self.y[sel]
    def get_ssrt(self):
        return self.size, self.good_size, self.good_r_transf, self.good_t
    def reshape(self, out):
        return out.reshape(self.sh)

def setit(name, dictionary, default):
    return dictionary[name] if name in dictionary else default

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
        self.heaviside = kwargs['heaviside'] if 'heaviside' in kwargs else SlepianMollifier(2*self.M)
        # parameters that depend on other parameters
        self.radial_width = self.M*self.h
        self.heaviside_width = self.radial_width - self.pad_zone*self.h

        # construct radial grid
        self._generate_radial_grid()

        # construct QFS boundaries
        self._generate_qfs_boundaries()

        # storage for registered x / y for interpolation to
        # general sets of points
        self.registered_xs = []
        self.registered_ys = []
        self.registered_partitions = []

        # helper for near-radial interpolation
        self.near_radial_guess_inds = \
            np.tile(np.arange(self.bdy.N), self.M).reshape(self.M, self.bdy.N)

    def register_grid(self, grid, danger_zone_distance=None, verbose=False):
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
        
        # find coords for everything within radial_width
        out = gridpoints_near_curve(self.bdy.x, self.bdy.y, self.grid.xv, 
                self.grid.yv, self.radial_width, interpolation_scheme=self.coordinate_scheme,
                tol=self.coordinate_tolerance, verbose=verbose)

        # find points inside of the curve
        inside = points_inside_curve(self.grid.xg, self.grid.yg, out)
        self.phys = inside if self.interior else np.logical_not(inside)
        self.ext = np.logical_not(self.phys)

        # wrangle output from gridpoints_near_curve
        ia = out[0]
        r = out[1][ia]
        t = out[2][ia]
        ia_good = r <= 0 if self.interior else r >= 0

        # construct some internal variables
        self.grid_in_annulus = np.zeros(self.grid.shape, dtype=bool)
        self.grid_in_annulus[ia] = ia_good
        self.grid_not_in_annulus = np.logical_not(self.grid_in_annulus)

        # save coordinate values for those points that are in annulus
        self.grid_ia_x = self.grid.xg[self.grid_in_annulus]
        self.grid_ia_y = self.grid.yg[self.grid_in_annulus]
        self.grid_ia_r = r[ia_good]
        self.grid_ia_t = t[ia_good]
        # for use in NUFFT interpolation
        lb = -self.radial_width if self.interior else 0.0
        ub = 0.0 if self.interior else self.radial_width
        self.grid_ia_r_transf = np.arccos(affine_transformation(self.grid_ia_r, lb, ub, 1.0, -1.0, use_numexpr=True))
        self.interface_x_transf = affine_transformation(self.interface.x, self.grid.x_bounds[0], self.grid.x_bounds[1], 0.0, 2*np.pi, use_numexpr=True)
        self.interface_y_transf = affine_transformation(self.interface.y, self.grid.y_bounds[0], self.grid.y_bounds[1], 0.0, 2*np.pi, use_numexpr=True)

        # get physical points that aren't in annulus
        phys_na = np.logical_and(self.phys, self.grid_not_in_annulus)

        ########################################################################
        # Compute regularized Heaviside functions for coupling grids

        lb = -self.heaviside_width if self.interior else self.heaviside_width
        grts = affine_transformation(self.grid_ia_r, lb, 0, -1, 1, use_numexpr=True)
        self.grid_to_radial_step = 1.0 - self.heaviside(grts)
        self.grid_step = np.zeros_like(self.grid.xg)
        self.grid_step[self.phys] = 1.0
        self.grid_step[self.grid_in_annulus] = self.grid_to_radial_step     
        arts = affine_transformation(self.radial_rv, lb, 0, -1, 1, use_numexpr=True)
        self.radial_cutoff = self.heaviside(arts)

        if danger_zone_distance != None:
            out = gridpoints_near_curve(self.bdy.x, self.bdy.y, self.grid.xv, 
                    self.grid.yv, self.radial_width+danger_zone_distance, interpolation_scheme=self.coordinate_scheme,
                    tol=self.coordinate_tolerance, verbose=verbose)
            ia = out[0]
            r = out[1][ia]
            ia_good = r <= 0 if self.interior else r >= 0
            in_danger_zone = ia.copy()
            in_danger_zone[ia] = ia_good
            self.grid_in_danger_zone = in_danger_zone
            # get correct vector needed
            rr = np.ones(np.prod(self.radial_shape), dtype=bool)
            self.in_danger_zone = np.concatenate([in_danger_zone[phys_na], rr])
            # now get guess indeces
            gi = (out[2]//self.bdy.dt).astype(int)
            self.guess_inds = np.concatenate([gi[phys_na], self.near_radial_guess_inds.ravel()])

    ############################################################################
    # Functions relating to the radial grid

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
            w = np.polynomial.chebyshev.chebvander(1, self.M-1).dot(VI0)[0]
        else:
            w = np.polynomial.chebyshev.chebvander(-1, self.M-1).dot(VI0)[0]
        self.chebyshev_interp_f_to_bdy = w
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

    ############################################################################
    # Functions for communicating between grids
    def interpolate_radial_to_boundary(self, f):
        return self.chebyshev_interp_f_to_bdy.dot(f)
    def interpolate_radial_to_boundary_normal_derivative(self, f):
        return self.chebyshev_interp_df_dn_to_bdy.dot(f)
    def interpolate_radial_to_boundary_normal_derivative2(self, f):
        return self.chebyshev_interp_df_dn2_to_bdy.dot(f)
    def interpolate_grid_to_interface(self, f, order=3):
        if order == np.Inf:
            return self.nufft_interpolate_grid_to_interface(f)
        else:
            return self.poly_interpolate_grid_to_interface(f, order)
    def poly_interpolate_grid_to_interface(self, f, order=None):
        """
        Interpolation from grid to inteface.  Note that f is not multiplied by
        grid_step in this function!  If f is not smooth on the whole grid, 
        first multiply it by grid_step to get an accurate result
        """
        grid = self.grid
        lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
        ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
        hs =   [self.grid.xh, self.grid.yh]
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        return interp(self.interface.x, self.interface.y)
    def nufft_interpolate_grid_to_interface(self, f):
        """
        Interpolation from grid to inteface.  Note that f is not multiplied by
        grid_step in this function!  If f is not smooth on the whole grid, 
        first multiply it by grid_step to get an accurate result
        """
        funch = np.fft.fft2(f)
        out = np.zeros(self.bdy.N, dtype=complex)
        diagnostic = finufftpy.nufft2d2(self.interface_x_transf, self.interface_y_transf, out, 1, 1e-14, funch, modeord=1)
        return out.real/np.prod(funch.shape)
    def interpolate_grid_to_radial(self, f, order=3):
        """
        This function will typically not produce correct results, unless f is
        smooth across the whole domain!  However, it is included as it is useful
        for initializing certain kinds of problems.
        """
        grid = self.grid
        lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
        ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
        hs =   [self.grid.xh, self.grid.yh]
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        return interp(self.radial_x, self.radial_y)
    def interpolate_radial_to_grid(self, fr, f=None):
        self.interpolation_hold[:self.M,:] = fr
        self.interpolation_hold[self.M:,:] = fr[::-1]
        funch = np.fft.fft2(self.interpolation_hold)*self.interpolation_modifier
        funch[self.M] = 0.0
        out = np.zeros(self.grid_ia_r.shape[0], dtype=complex)
        diagnostic = finufftpy.nufft2d2(self.grid_ia_r_transf, self.grid_ia_t, out, 1, 1e-14, funch, modeord=1)
        vals = out.real/np.prod(funch.shape)
        if f is not None: f[self.grid_in_annulus] = vals
        return vals
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
    def get_interpolation_key(self, x, y, fix_r=False, danger_zone=None, gi=None):
        here = [p.is_it_i(x, y, fix_r) for p in self.registered_partitions]
        where = np.where(here)[0]
        if len(where) > 0:
            key = where[0]
        else:
            key = self._register_points(x, y, fix_r, danger_zone, gi)
        return key
    def register_points(self, x, y, fix_r=False, danger_zone=None, gi=None):
        return self.get_interpolation_key(x, y, fix_r, danger_zone, gi)
    def _register_points(self, x, y, fix_r=False, danger_zone=None, gi=None):
        k = len(self.registered_partitions)
        p = EmbeddedPointPartition(self, x, y, fix_r, danger_zone, gi)
        self.registered_partitions.append(p)
        return k
    def interpolate_radial_to_points(self, fr, x, y):
        key = self.get_interpolation_key(x, y)
        p = self.registered_partitions[key]
        sz, gsz, gr, gt = p.get_ssrt()
        ia, oa = p.get_ia()
        self.interpolation_hold[:self.M,:] = fr
        self.interpolation_hold[self.M:,:] = fr[::-1]
        funch = np.fft.fft2(self.interpolation_hold)*self.interpolation_modifier
        funch[self.M] = 0.0
        out = np.zeros(gsz, dtype=complex)
        diagnostic = finufftpy.nufft2d2(gr, gt, out, 1, 1e-14, funch, modeord=1)
        vals = out.real/np.prod(funch.shape)
        out = np.empty(sz)
        out[ia] = vals
        out[oa] = np.nan
        return p.reshape(out)
    def interpolate_to_points(self, ff, x, y, fix_r=False, danger_zone=None, gi=None):
        key = self.get_interpolation_key(x, y, fix_r, danger_zone, gi)
        p = self.registered_partitions[key]
        # get the categories
        c1, c2, c3 = p.get_categories()
        c1n, c2n, c3n = p.get_category_Ns()
        # initialize output vector
        output = np.empty(x.size)
        # interpolate appropriate portion with radial
        if c1n > 0:
            sz, gsz, gr, gt = p.get_ssrt()
            fr = ff.get_radial_value_list()[0]
            self.interpolation_hold[:self.M,:] = fr
            self.interpolation_hold[self.M:,:] = fr[::-1]
            funch = np.fft.fft2(self.interpolation_hold)*self.interpolation_modifier
            funch[self.M] = 0.0
            out = np.zeros(gsz, dtype=complex)
            diagnostic = finufftpy.nufft2d2(gr, gt, out, 1, 1e-14, funch, modeord=1)
            vals = out.real/np.prod(funch.shape)
            output[c1] = vals
        # interpolate appropriate portion with grid (polynomial, for now...)
        if c2n > 0:
            f = ff.get_grid_value()#*self.grid_step
            grid = self.grid
            lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
            ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
            hs =   [self.grid.xh, self.grid.yh]
            interp = fast_interp.interp2d(lbds, ubds, hs, f, k=7, p=[True, True])
            output[c2] = interp(x[c2], y[c2])
        # fill in those that are not in the physical domain with nan
        if c3n > 0:
            output[c3] = np.nan
        return output

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
    def gradient2(self, fx, fy, fr):
        fxr, fyr = self.radial_grid_derivatives(fr)
        self.interpolate_radial_to_grid(fxr, fx)
        self.interpolate_radial_to_grid(fyr, fy)
        return fxr, fyr
    def gradient(self, f, fr, xd_func, yd_func):
        # compute gradient on the background grid
        fx = xd_func(f)
        fy = yd_func(f)
        # compute on the radial grid
        fxr, fyr = self.radial_grid_derivatives(fr)
        # interpolate from the radial grid to the background grid
        self.interpolate_radial_to_grid(fxr, fx)
        self.interpolate_radial_to_grid(fyr, fy)
        # set to 0 on regular grid in the exterior region
        fx *= self.phys
        fy *= self.phys
        return fx, fy, fxr, fyr

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
        self.bdy_qfs = QFS_Boundary(self.bdy, eps=eps, forced_source_upsampling_factor=fsuf)
        self.interface_qfs = QFS_Boundary(self.interface, eps=eps, forced_source_upsampling_factor=fsuf)
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

