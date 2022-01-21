import numpy as np
import pybie2d
import fast_interp
import finufft
PointSet = pybie2d.point_set.PointSet
from ipde.utilities import affine_transformation
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from near_finder.phys_routines import points_inside_curve_update
from ipde.embedded_boundary_tr import LoadEmbeddedBoundary

Grid = pybie2d.grid.Grid

def merge_sources(src_list):
    """
    Merge sources.  This is only sufficient for single layers.
    Need to construct something better, eventually
    """
    p = PointSet(c=np.concatenate([src.c for src in src_list]))
    p.weights = np.concatenate([src.weights for src in src_list])
    p.normal_x = np.concatenate([src.normal_x for src in src_list])
    p.normal_y = np.concatenate([src.normal_y for src in src_list])
    return p

class test(object):
    def __init__(self):
        pass

class EmbeddedPointPartition(object):
    """
    Partition the points (x, y), which need to have no structure

    In particular, we separate the points into:
        1) physical / exterior points
        2) physical+not in any annulus; physical+in an annulus; exterior
        3) physical+not in any annulus; physical+in annulus 1;
            physical+in annulus 2; ...; physical+in annulus n; exterior
    """
    # fix_r TO BE REMOVED!!!
    def __init__(self, ebdyc, x, y, fix_r=True):
        self.grid = ebdyc.grid
        self.x_check = x
        self.y_check = y
        self.sh = x.shape
        self.x = x.ravel()
        self.y = y.ravel()
        self.size = self.x.size
        # zone 1 is physical and not inside an annulus
        zone1 = np.ones(self.size, dtype=bool)
        zone1_or_2 = np.ones(self.size, dtype=bool)
        # zone 2 is physical and inside an annulus
        # given as a list of indeces
        # the r, transformed r, and t values are also saved
        zone2l = []
        zone2r = []
        zone2_transfr = []
        zone2t = []
        # zone 3 is non-physical
        # given as list of indeces
        # with r and t values also saved
        zone3l = []
        zone3r = []
        zone3t = []
        # loop over embedded boundaries
        for ind, ebdy in enumerate(ebdyc):
            code, interior, computed, full_r, full_t = ebdy.coordinate_mapper.classify(self.x, self.y)
            is_annulus = code == 1
            is_phys_not_annulus = code == (3 if ebdy.interior else 2)
            is_phys = interior if ebdy.interior else ~interior
            zone1 = np.logical_and(zone1, is_phys_not_annulus)
            zone_1_or_2 = np.logical_and(zone1_or_2, is_phys)
            # get in annulus points
            zone2l.append(np.where(is_annulus)[0])
            rhere = full_r[is_annulus]
            zone2r.append(rhere)
            zone2_transfr.append(ebdy.nufft_transform_r(rhere))
            zone2t.append(full_t[is_annulus])
            # i don't think there's a reason for zone3 to be a list?
        zone3 = ~zone_1_or_2

        # save these away
        self.zone1 = zone1
        self.zone1_or_2 = zone1_or_2
        self.zone2l = zone2l
        self.zone2r = zone2r
        self.zone2_transfr = zone2_transfr
        self.zone2t = zone2t

        # get Ns
        self.zone1_N = int(np.sum(self.zone1))
        self.zone2_Ns = [len(z2) for z2 in self.zone2l]
        self.zone2_N = int(np.sum(self.zone2_Ns))
        self.zone3_N = int(np.sum(zone3))

        # get transformed x's, y's for NUFFT interpolation (in zone1)
        self.x_transf = affine_transformation(self.x[self.zone1], self.grid.x_bounds[0], self.grid.x_bounds[1], 0.0, 2*np.pi, use_numexpr=True)
        self.y_transf = affine_transformation(self.y[self.zone1], self.grid.y_bounds[0], self.grid.y_bounds[1], 0.0, 2*np.pi, use_numexpr=True)

    def is_it_i(self, x, y, fix_r):
        x_is_i = x is self.x_check
        y_is_i = y is self.y_check
        return x_is_i and y_is_i

    def reshape(self, out):
        return out.reshape(self.sh)

    def get_Ns(self):
        return self.zone1_N, self.zone2_N, self.zone3_N

class _old_EmbeddedPointPartition(object):
    """
    Partition the points (x, y), which need to have no structure

    In particular, we separate the points into:
        1) physical / exterior points
        2) physical+not in any annulus; physical+in an annulus; exterior
        3) physical+not in any annulus; physical+in annulus 1;
            physical+in annulus 2; ...; physical+in annulus n; exterior
    """
    def __init__(self, ebdyc, x, y, fix_r=False):
        """
        ebdyc: embedded boundary collection
        x:     x-coordinates of points to partition
        y:     y-coordinates of points to partition
        fix_r: after finding coordinates, do we set any with r that would place
                them in the aphysical region to 0 (to lie on the boundary?)
                note: this should only be used if you're condifident all points
                are actually physical.  in this case this is really being used
                to deal with very small errors from the Newton solver
        """
        self.grid = ebdyc.grid
        self.x_check = x
        self.y_check = y
        self.fix_r = fix_r
        self.sh = x.shape
        self.x = x.ravel()
        self.y = y.ravel()
        self.size = self.x.size
        # zone 1 is physical and not inside an annulus
        zone1 = np.ones(self.size, dtype=bool)
        zone1_or_2 = np.ones(self.size, dtype=bool)
        # zone 2 is physical and inside an annulus
        # given as a list of indeces
        # the r, transformed r, and t values are also saved
        zone2l = []
        zone2r = []
        zone2_transfr = []
        zone2t = []
        # zone 3 is non-physical
        # given as list of indeces
        # with r and t values also saved
        zone3l = []
        zone3r = []
        zone3t = []
        # holders for the one that has to interact with everything
        long_in_this_annulus = np.zeros(self.size, dtype=bool)
        # loop over embedded boundaries
        for ind, ebdy in enumerate(ebdyc):
            bx = ebdy.bdy.x
            by = ebdy.bdy.y
            width = ebdy.radial_width
            interior = ebdy.interior
            # find coordintes
            # full_r, full_t, have_coords, interior = ebdy.coordinate_mapper.classify(x, y)
            code, interior, computed, full_r, full_t = ebdy.coordinate_mapper.classify(x, y)
            have_coords = computed
            r = full_r[have_coords]
            t = full_t[have_coords]
            # fix r values, if required
            if self.fix_r:
                ebdy.fix_r(r)
            # check on in annulus
            in_this_annulus, phys_not_in_this_annulus,  \
                                exterior = ebdy.check_if_r_in_annulus(r)
            # set indicators in zone1 for those owned by this boundary
            is_phys = interior if ebdy.interior else ~interior
            zone1 = np.logical_and(zone1, is_phys)
            zone1_or_2 = np.logical_and(zone1_or_2, is_phys)
            zone1[have_coords] = np.logical_and(zone1[have_coords], phys_not_in_this_annulus)
            zone1_or_2[have_coords] = np.logical_and(zone1_or_2[have_coords], np.logical_not(exterior))
            # dz is just an old name from past routines
            dz = np.where(have_coords)[0]
            # zone 2 inds and coords
            zone2l.append(dz[in_this_annulus])
            rhere = r[in_this_annulus]
            zone2r.append(rhere)
            zone2_transfr.append(ebdy.nufft_transform_r(rhere))
            zone2t.append(t[in_this_annulus])
            # zone 3 inds and coords
            zone3l.append(dz[exterior])
            zone3r.append(r[exterior])
            zone3t.append(t[exterior])

        # save these away
        self.zone1 = zone1
        self.zone1_or_2 = zone1_or_2
        self.zone2l = zone2l
        self.zone2r = zone2r
        self.zone2_transfr = zone2_transfr
        self.zone2t = zone2t
        self.zone3l = zone3l
        self.zone3r = zone3r
        self.zone3t = zone3t
        self.full_t = full_t
        self.full_r = full_r

        # get Ns
        self.zone1_N = int(np.sum(self.zone1))
        self.zone2_Ns = [len(z2) for z2 in self.zone2l]
        self.zone2_N = int(np.sum(self.zone2_Ns))
        self.zone3_Ns = [len(z3) for z3 in self.zone3l]
        self.zone3_N = int(np.sum(self.zone3_Ns))

        # get transformed x's, y's for NUFFT interpolation (in zone1)
        self.x_transf = affine_transformation(self.x[self.zone1], self.grid.x_bounds[0], self.grid.x_bounds[1], 0.0, 2*np.pi, use_numexpr=True)
        self.y_transf = affine_transformation(self.y[self.zone1], self.grid.y_bounds[0], self.grid.y_bounds[1], 0.0, 2*np.pi, use_numexpr=True)

    def is_it_i(self, x, y, fix_r):
        x_is_i = x is self.x_check
        y_is_i = y is self.y_check
        r_is_i = fix_r == self.fix_r
        return x_is_i and y_is_i and r_is_i

    def reshape(self, out):
        return out.reshape(self.sh)

    def get_Ns(self):
        return self.zone1_N, self.zone2_N, self.zone3_N

def LoadEmbeddedBoundaryCollection(d):
    ebdy_list = [LoadEmbeddedBoundary(ebdy_dict) for ebdy_dict in d['ebdy_list']]
    ebdyc = EmbeddedBoundaryCollection(ebdy_list)
    if d['grid'] is not None:
        ebdyc.register_grid(Grid(**d['grid']))
    if d['bumpy'] is not None:
        ebdyc.bumpy = d['bumpy']
        ebdyc.bumpy_readied = True
    return ebdyc

class EmbeddedBoundaryCollection(object):
    def __init__(self, ebdy_list):
        """
        Collection of embedded boundaries

        ebdy_list (required):
            list[type(EmbeddedBoundary)]
        """
        self.ebdys = ebdy_list
        self.N = len(self.ebdys)

        # storage for registered x / y for interpolation to
        # general sets of points
        self.registered_partitions = []

        # information for solvers that require bumpy functions
        self.bumpy_readied = False

    def __iter__(self):
        return (ebdy for ebdy in self.ebdys)
    def __len__(self):
        return len(self.ebdys)
    def __getitem__(self, ind):
        return self.ebdys[ind]

    def save(self):
        ebdy_list = [ebdy.save() for ebdy in self.ebdys]
        if hasattr(self, 'grid'):
            grid = {
                'x_bounds'    : self.grid.x_bounds,
                'y_bounds'    : self.grid.y_bounds,
                'Nx'          : self.grid.Nx,
                'Ny'          : self.grid.Ny,
                'mask'        : self.grid.mask,
                'x_endpoints' : self.grid.x_endpoints,
                'y_endpoints' : self.grid.y_endpoints,
            }
        else:
            grid = None
        bumpy = self.bumpy if hasattr(self, 'bumpy') else None
        d = {
            'ebdy_list' : ebdy_list,
            'grid'      : grid,
            'bumpy'     : bumpy,
        }
        return d
    def generate_grid(self, h=None, Ns=None, force_square=False, extra_coordinate_distance=0.0):
        """
        Auto generate an underlying grid
        Requires the first boundary to be interior
        Adds a "cheat space" to the outside of the interior boundary
        That is the same width as the radial boundary for the interior boundary

        h gives a gridspacing; will be taken the same as the radial gridspacing
        if not provided

        Ns = [Nx, Ny].  If None, will be computed. An error will be raised
        if these are not big enough given the specified value of h
        """
        iebdy = self[0]
        if not iebdy.interior:
            raise Exception('Generate grid may only be used if the first boundary is interior.')
        ibdy = iebdy.bdy
        # cheater space around the outside (buffer zone to make things easier)
        cheat_space = iebdy.radial_width
        # approximate bounds (see if we can make these smaller!)
        xmin = ibdy.x.min()# - cheat_space
        ymin = ibdy.y.min()# - cheat_space
        xmax = ibdy.x.max() + 2*cheat_space # this is so we have room for bumpy...
        ymax = ibdy.y.max() + 2*cheat_space # this is so we have room for bumpy...
        self.bump_location = [ibdy.x.max()+cheat_space, ibdy.y.max()+cheat_space]
        # approximate range
        xran = xmax - xmin
        yran = ymax - ymin
        # get h and Ns (conservatively for computing N)
        if h is None: h = iebdy.radial_width / iebdy.M
        if Ns is None:
            Nx = 2*int(0.5*np.ceil(xran/h))
            Ny = 2*int(0.5*np.ceil(yran/h))
        else:
            Nx = Ns[0]
            Ny = Ns[1]
            _xmax = xmin + Nx*h
            _ymax = ymin + Ny*h
            if _xmax < xmin + xran:
                raise Exception('Provided value of Nx is too small')
            if _ymax < ymin + yran:
                raise Exception('Provided value of Ny is too small')
        if force_square:
            N = max(Nx, Ny)
            Nx = N
            Ny = N
        # compute actual maximum bounds (minimum bounds are through definition)
        xmax = xmin + Nx*h
        ymax = ymin + Ny*h
        # construct the required grid
        grid = Grid([xmin, xmax], Nx, [ymin, ymax], Ny, x_endpoints=[True, False], y_endpoints=[True, False])
        # check to make sure the gridspacing is correct
        assert np.abs(grid.xh - h) < 1e-15, 'Gridspacing not what was requested'
        assert np.abs(grid.yh - h) < 1e-15, 'Gridspacing not what was requested'
        # register this grid
        self.register_grid(grid)#, extra_coordinate_distance=extra_coordinate_distance)
        # flag that the bumpy hasn't been constructed
        if self.bumpy_readied:
            self.bumpy_readied = False
        # return the grid
        return grid

    def search_for_physical_points(self, grid):
        """
        Finds physical points for some *other* grid, not the registered one
        """
        physical = np.ones(grid.shape, dtype=bool)
        for ei, ebdy in enumerate(self):
            ph = ebdy.search_for_physical_points(grid)
            physical = np.logical_and(physical, ph)
        return physical

    def register_grid(self, grid, verbose=False):
        """
        Register a grid object, construct a quadtree-based coordinate mapper

        grid: grid of type(Grid) from pybie2d (see pybie2d doc)
        """
        self.grid = grid

        # perform individual ebdy registration
        for ebdy in self:
            ebdy.register_grid(grid)

        # now get the physical region
        ext = np.ones(grid.shape, dtype=bool) if self.ebdys[0].interior \
                else np.zeros(grid.shape, dtype=bool)
        for ebdy in self:
            if ebdy.interior:
                ext = np.logical_and(ext, ebdy.grid_exterior)
            else:
                ext = np.logical_or(ext, ebdy.grid_interior)
        self.ext = ext
        self.phys = np.logical_not(self.ext)

        # phys = np.zeros(grid.shape, dtype=bool) if self.ebdys[0].interior \
        #             else np.ones(grid.shape, dtype=bool)
        # for ebdy in self:
        #     if ebdy.interior:
        #         phys = np.logical_or(phys, ebdy.grid_interior)
        #     else:
        #         phys = np.logical_or(phys, ebdy.grid_exterior)
        # self.phys = phys
        # self.ext = np.logical_not(self.phys)

        self.phys_inds = np.zeros(grid.shape, dtype=int)
        self.phys_n = np.sum(self.phys)
        self.phys_inds[self.phys] = np.arange(self.phys_n)

        # Fourier operators
        kxv = np.fft.fftfreq(grid.Nx, grid.xh/(2*np.pi))
        kyv = np.fft.fftfreq(grid.Ny, grid.yh/(2*np.pi))
        self.kx = kxv[:,None]
        self.ky = kyv
        self.ikx = 1j*self.kx
        self.iky = 1j*self.ky
        self.lap = -self.kx*self.kx - self.ky*self.ky

        # get in_annulus indeces
        iax = np.concatenate([ebdy.grid_ia_xind for ebdy in self])
        iay = np.concatenate([ebdy.grid_ia_yind for ebdy in self])
        ia = np.zeros(grid.shape, dtype=bool)
        ia[iax, iay] = True
        self.in_annulus = ia
        self.phys_not_in_annulus = np.logical_and(self.phys, np.logical_not(self.in_annulus))

        # construct the full grid_step
        self.grid_step = self.phys.astype(float)
        for ebdy in self:
            self.grid_step[ebdy.grid_ia_xind, ebdy.grid_ia_yind] *= \
                ebdy.grid_to_radial_step

        # physical gridpoints
        grid_phys_x = self.grid.xg[self.phys]
        grid_phys_y = self.grid.yg[self.phys]
        self.grid_phys = PointSet(grid_phys_x, grid_phys_y)

        # radial gridpoints
        radial_x_list = [ebdy.radial_x.ravel() for ebdy in self.ebdys]
        radial_y_list = [ebdy.radial_y.ravel() for ebdy in self.ebdys]
        self.radial_x = np.concatenate(radial_x_list)
        self.radial_y = np.concatenate(radial_y_list)
        self.radial_pts = PointSet(self.radial_x, self.radial_y)

        # physical gridpoints and radial gridpoints
        xx = np.concatenate([grid_phys_x, self.radial_x])
        yy = np.concatenate([grid_phys_y, self.radial_y])
        self.grid_and_radial_pts = PointSet(xx, yy)

        # physical gridpoints that aren't in an annulus
        grid_pna_x = self.grid.xg[self.phys_not_in_annulus]
        grid_pna_y = self.grid.yg[self.phys_not_in_annulus]
        self.grid_pna = PointSet(grid_pna_x, grid_pna_y)
        self.grid_pna_num = self.grid_pna.N

        # physical gridpoints that aren't in annuluas and radial gridpoints
        xx = np.concatenate([grid_pna_x, self.radial_x])
        yy = np.concatenate([grid_pna_y, self.radial_y])
        self.pnar = PointSet(xx, yy)

        # boundary points
        bvx = [ebdy.bdy.x for ebdy in self.ebdys]
        bvy = [ebdy.bdy.y for ebdy in self.ebdys]
        self.all_bvx = np.concatenate(bvx)
        self.all_bvy = np.concatenate(bvy)
        self.all_bv = PointSet(self.all_bvx, self.all_bvy)

        # interface points
        ivx = [ebdy.interface.x for ebdy in self.ebdys]
        ivy = [ebdy.interface.y for ebdy in self.ebdys]
        self.all_ivx = np.concatenate(ivx)
        self.all_ivy = np.concatenate(ivy)
        self.all_iv = PointSet(self.all_ivx, self.all_ivy)

        # exterior points
        self.exterior_x = self.grid.xg[self.ext]
        self.exterior_y = self.grid.yg[self.ext]
        self.exterior_targ = PointSet(self.exterior_x, self.exterior_y)

        # physical gridpoints, not in annulus, and interface points
        grid_pnai_x = np.concatenate([grid_pna_x, self.all_ivx])
        grid_pnai_y = np.concatenate([grid_pna_y, self.all_ivy])
        self.grid_pnai = PointSet(grid_pnai_x, grid_pnai_y)

        # get the Ns for each bounday
        self.bdy_Ns = [ebdy.bdy.N for ebdy in self.ebdys]
        self.cumsum_bdy_Ns = np.cumsum(self.bdy_Ns)
        # splitter for boundary-like things
        self.splitter = self.cumsum_bdy_Ns[:-1]
        # get Ns for radial grids
        self.radial_Ns = [ebdy.bdy.N*ebdy.M for ebdy in self.ebdys]
        self.cumsum_radial_Ns = np.cumsum(self.radial_Ns)
        # splitter for radial-grid-like things
        self.rsplitter = self.cumsum_radial_Ns[:-1]

        # get transformed interfaces for interpolation
        w = [ebdy.interface_x_transf for ebdy in self.ebdys]
        self.interfaces_x_transf = np.concatenate(w)
        w = [ebdy.interface_y_transf for ebdy in self.ebdys]
        self.interfaces_y_transf = np.concatenate(w)

        # number of effective degrees of freedom (used for EmbeddedFunction Class)
        self.grid_dof = self.grid_phys.N
        self.radial_dof_list = [np.product(ebdy.radial_shape) for ebdy in self]
        self.radial_dof = np.sum(self.radial_dof_list)
        self.dof = self.grid_dof + self.radial_dof

    def divide_grid_and_radial(self, v):
        g, w = np.split(v, [self.grid_phys.N])
        return g, self.v2r(w)

    def divide_grid_and_radial2(self, v):
        gv, w = np.split(v, [self.grid_phys.N])
        g = np.zeros(self.grid.shape, v.dtype)
        g[self.phys] = gv
        w = self.v2r(w)
        out = EmbeddedFunction(self)
        out.load_data(g, w)
        return out

    def divide_pnai(self, v):
        """
        Convert output of grid_pnai evaluations to grid / list of bdy format
        """
        g, w = np.split(v, [self.grid_pna.N])
        return g, self.v2l(w)

    def v2l(self, v):
        """
        Convert long vector (of length sum bdy Ns) to list of vectors
        """
        return np.split(v, self.splitter)
    def v2l2(self, v):
        return np.split(v, self.splitter*2)
    def v2l_vector(self, v):
        """
        Convert long [2,vector] (of length sum bdy Ns) to list of [2,vectors]
        """
        return np.split(v, self.splitter, axis=1)

    def v2r(self, v):
        L1 =  np.split(v, self.rsplitter)
        return [L.reshape(ebdy.radial_x.shape) for L, ebdy in zip(L1, self.ebdys)]

    ############################################################################
    # Functions for communicating between grids / interfaces / boundaries
    def interpolate_radial_to_boundary(self, f):
        fb_list = [ebdy.interpolate_radial_to_boundary(fr) for ebdy, fr in zip(self.ebdys, f)]
        out = BoundaryFunction(self)
        out.load_data(fb_list)
        return out
    def interpolate_radial_to_boundary_normal_derivative(self, f):
        fr_list = self.get_radial_value_list()
        fb_list = [ebdy.interpolate_radial_to_boundary_normal_derivative for ebdy in self.ebdys]
        out = BoundaryFunction(self)
        out.load_data(fb_list)
        return out
    def interpolate_grid_to_interface(self, f, order=3, cutoff=None):
        """
        Interpolate function on grid to all interfaces at once
        Inputs:
            f, float(nx, ny) or EmbeddedFunction, function to interpolate
            order: order of interpolation.  Either 1-7 or np.Inf.
                1-7 calls fast_interp, np.Inf calls a NUFFT
        """
        if type(f) == EmbeddedFunction: f = f.get_grid_value()
        if cutoff == None: cutoff = order == np.Inf
        # cutoff, if cutting off
        fc = f*self.grid_step if cutoff else f
        if order == np.Inf:
            return self.nufft_interpolate_grid_to_interface(fc)
        else:
            return self.poly_interpolate_grid_to_interface(fc, order)
    def poly_interpolate_grid_to_interface(self, f, order=None):
        """
        Call through interpolate_grid_to_interface
        """
        grid = self.grid
        lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
        ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
        hs =   [self.grid.xh, self.grid.yh]
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        return interp(self.all_ivx, self.all_ivy)
    def nufft_interpolate_grid_to_interface(self, f):
        """
        Call through interpolate_grid_to_interface
        """
        funch = np.fft.fft2(f)
        out = np.zeros(self.interfaces_x_transf.size, dtype=complex)
        diagnostic = finufft.nufft2d2(self.interfaces_x_transf, self.interfaces_y_transf, funch, out, isign=1, eps=1e-14, modeord=1)
        return out.real/np.prod(funch.shape)
    def update_radial_to_grid1(self, f):
        # _, fg, fr = f.get_components()
        _ = self.interpolate_radial_to_grid1(f.get_radial_value_list(), f['grid'])
        # _ = self.interpolate_radial_to_grid(f, fg)
    def interpolate_radial_to_grid1(self, fr_list, f=None):
        return [ebdy.interpolate_radial_to_grid1(fr, f) for ebdy, fr in zip(self.ebdys, fr_list)]
    # def update_radial_to_grid2(self, f):
    #     # _, fg, fr = f.get_components()
    #     _ = self.interpolate_radial_to_grid2(f.get_radial_value_list(), f['grid'])
    #     # _ = self.interpolate_radial_to_grid(f, fg)
    # def interpolate_radial_to_grid2(self, fr_list, f=None):
    #     return [ebdy.interpolate_radial_to_grid2(fr, f) for ebdy, fr in zip(self.ebdys, fr_list)]
    def merge_grids(self, f):
        for ebdy, fr in zip(self.ebdys, f.radial_value_list):
            ebdy.merge_grids(f.grid_value, fr)
    def interpolate_grid_to_radial(self, f, order=3):
        """
        This function will typically not produce correct results, unless f is
        smooth across the whole domain!  However, it is included as it is useful
        for initializing certain kinds of problems.

        Order = Inf is currently not supported, though it could be, but it would
        involve transforming all of the radial grid values (eating a lot of memory)
        """
        grid = self.grid
        lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
        ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
        hs =   [self.grid.xh, self.grid.yh]
        if type(f) == EmbeddedFunction:
            f = f.get_grid_value()
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        radial_list = []
        for ebdy in self:
            radial_list.append(interp(ebdy.radial_x, ebdy.radial_y))
        return radial_list

    ############################################################################
    # Functions for dealing with interpolation to generic sets of points
    def get_interpolation_key(self, x, y, fix_r=False):
        here = [p.is_it_i(x, y, fix_r) for p in self.registered_partitions]
        where = np.where(here)[0]
        if len(where) > 0:
            key = where[0]
        else:
            key = self._register_points(x, y, fix_r)
        return key
    def register_points(self, x, y, fix_r=False):
        return self.get_interpolation_key(x, y, fix_r)
    def _register_points(self, x, y, fix_r=False):
        k = len(self.registered_partitions)
        p = EmbeddedPointPartition(self, x, y, fix_r)
        self.registered_partitions.append(p)
        return k
    def interpolate_to_points(self, ff, x, y, fix_r=False):
        key = self.get_interpolation_key(x, y, fix_r)
        p = self.registered_partitions[key]
        # get the category numbers
        c1n, c2n, c3n = p.get_Ns()
        # initialize output vector
        output = np.empty(x.size)
        # interpolate appropriate portion with grid (polynomial, for now...)
        if c1n > 0:
            zone1 = p.zone1
            funch = np.fft.fft2(ff.get_smoothed_grid_value())
            out = np.zeros(p.zone1_N, dtype=complex)
            diagnostic = finufft.nufft2d2(p.x_transf, p.y_transf, funch, out, isign=1, eps=1e-14, modeord=1)
            out.real/np.prod(funch.shape)
            output[zone1] = out.real/np.prod(funch.shape)
        if c2n > 0:
            for ind in range(self.N):
                ebdy = self[ind]
                z2l = p.zone2l[ind]
                z2transfr = p.zone2_transfr[ind]
                z2t = p.zone2t[ind]
                fr = ff[ind]
                output[z2l] = ebdy.interpolate_radial_to_points(fr, z2transfr, z2t)
        # fill in those that are exterior with nan
        if c3n > 0:
            for z3 in p.zone3l:
                output[z3] = np.nan
        return output

    ############################################################################
    # Functions for taking derivatives on the radial grid
    def gradient(self, ff, derivative_type='spectral'):
        """
        Compute the gradient of a function defined on the embedded boundary collection
        Inputs:
            ff, EmbeddedFunction: function to take the gradient of
            derivative_type: 'spectral' or 'fourth'
                in both cases, spectral differentation is used in the radial
                regions. On the grid, if 'spectral', then the function is cutoff
                and fourier based estimation of the derivative is used;
                if 'fourth', then the function is NOT cutoff and fourth-order
                centered differences are used.  This may provide better accuracy
                when the cutoff windows are thin.  However, if the cutoff windows
                are not at least 2h, then this can give catastrophically bad values
        Outputs:
            fx, fy: EmbeddedFunctions giving each required derivative
        """
        f, _, fr_list = ff.get_components()
        if derivative_type == 'spectral':
            fc = f*self.grid_step
            fc[self.ext] = 0.0
            fch = np.fft.fft2(fc)
            fxh = fch*self.ikx
            fyh = fch*self.iky
            fx = np.fft.ifft2(fxh).real
            fy = np.fft.ifft2(fyh).real
        else:
            fx = fd_x_4(f, self.grid.xh)
            fy = fd_y_4(f, self.grid.yh)
        # compute on the radial grid
        fxrs, fyrs = [], []
        for i in range(self.N):
            fxr, fyr = self[i].gradient(fx, fy, fr_list[i])
            fxrs.append(fxr)
            fyrs.append(fyr)
        # set to 0 on regular grid in the exterior region
        fx *= self.phys
        fy *= self.phys
        # generate EmbeddedFunctions
        ffx = EmbeddedFunction(self)
        ffy = EmbeddedFunction(self)
        ffx.load_data(fx, fxrs)
        ffy.load_data(fy, fyrs)
        return ffx, ffy
    def laplacian(self, ff, derivative_type='spectral'):
        """
        Compute the laplacian of a function defined on the embedded boundary collection
        Inputs:
            ff, EmbeddedFunction: function to take the gradient of
            derivative_type: 'spectral' or 'fourth'
                in both cases, spectral differentation is used in the radial
                regions. On the grid, if 'spectral', then the function is cutoff
                and fourier based estimation of the derivative is used;
                if 'fourth', then the function is NOT cutoff and fourth-order
                centered differences are used.  This may provide better accuracy
                when the cutoff windows are thin.  However, if the cutoff windows
                are not at least 2h, then this can give catastrophically bad values
        Outputs:
            lapf: EmbeddedFunction giving the laplacian of f
        """
        f, _, fr_list = ff.get_components()
        if derivative_type == 'spectral':
            fc = f*self.grid_step
            fc[self.ext] = 0.0
            fch = np.fft.fft2(fc)
            lapfh = fch*self.lap
            lapf = np.fft.ifft2(lapfh).real
        else:
            fx = fd_x_4(f, self.grid.xh)
            fy = fd_y_4(f, self.grid.yh)
            fxx = fd_x_4(fx, self.grid.xh)
            fyy = fd_y_4(fy, self.grid.yh)
        # compute on the radial grid
        lapfrs = []
        for i in range(self.N):
            lapfr = self.ebdys[i].laplacian(lapf, fr_list[i])
            lapfrs.append(lapfr)
        # set to 0 on regular grid in the exterior region
        lapf *= self.phys
        # generate EmbeddedFunctions
        lapff = EmbeddedFunction(self)
        lapff.load_data(lapf, lapfrs)
        return lapff

    ############################################################################
    # Functions for handling the de-meaning of functions
    def ready_bump(self, bump_loc=None, bump_width=None):
        if bump_width == None:
            bump_width = self[0].radial_width
        if bump_loc == None:
            if self.bump_location is None:
                raise Exception('if ebdyc has no bump_location, need to give bump_loc')
            bump_loc = self.bump_location
        grr = np.hypot(self.grid.xg-bump_loc[0], self.grid.yg-bump_loc[1])
        bumpy = self[0].heaviside.bump(affine_transformation(grr, 0, bump_width, 0, 1))
        bumpy_int = self.grid_integral(bumpy)
        self.bumpy = bumpy / bumpy_int
        self.bumpy_readied = True
    def demean_function(self, f):
        f_int = self.grid_integral(f)
        return f - f_int*self.bumpy

    ############################################################################
    # Functions for taking integrals
    def grid_integral(self, f):
        weight = self.grid.xh*self.grid.yh
        return np.sum(f)*weight

    def volume_integral(self, f):
        """
        Computes an integral of the EmbeddedFunction over the EmbeddedBoundaryCollection
        """
        if f.ebdyc() is not self:
            raise Exception('EmbeddedFunction must be defined over this EmbeddedBoundaryCollection.')
        # first compute the grid based integral
        integral = self.grid_integral(f.get_smoothed_grid_value())
        # now compute the radial integrals
        for fr, ebdy in zip(f, self):
            integral += ebdy.radial_integral(fr)
        return integral
