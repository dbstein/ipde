import numpy as np
import pybie2d
import fast_interp
import finufftpy
PointSet = pybie2d.point_set.PointSet
from near_finder.phys_routines import points_inside_curve_update
from near_finder.coordinate_routines import compute_local_coordinates
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.utilities import affine_transformation
from ipde.annular.annular import ApproximateAnnularGeometry
from ipde.derivatives import fd_x_4, fd_y_4, fourier

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
    def __init__(self, ebdyc, x, y, fix_r=False, dzl=None, gil=None):
        """
        ebdyc: embedded boundary collection
        x:     x-coordinates of points to partition
        y:     y-coordinates of points to partition
        fix_r: after finding coordinates, do we set any with r that would place
                them in the aphysical region to 0 (to lie on the boundary?)
               note: this should only be used if you're condifident all points
                are actually physical.  in this case this is really being used
                to deal with very small errors from the Newton solver
        dzl:   danger zone list: if available, a list of "danger zones" which
                give points for which coordinates should be computed
                if this is accurate enough, this will considerably speed up the
                calculation, as a near-points finder will not need to be called
                if it is not accurate, it may cause the coordinate solver to fail
        gil:   guess index list: a list of "guess indeces", corresponding to
                the "danger zone list", giving initial guesses for the
                coordinate solving scheme
        """
        self.ebdyc = ebdyc
        self.grid = self.ebdyc.grid
        self.x_check = x
        self.y_check = y
        self.fix_r = fix_r
        self.dzl = dzl
        self.gil = gil
        self.sh = x.shape
        self.x = x.ravel()
        self.y = y.ravel()
        self.size = self.x.size
        if self.dzl is not None and self.gil is not None:
            self.has_danger_zone = [dz is not None and gi is not None for
                                        dz, gi in zip(self.dzl, self.gil)]
        else:
            self.has_danger_zone = [None,]*len(self.ebdyc)
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
        # holders for t, r for anything that we compute
        full_t = np.empty(self.size, dtype=float)
        full_r = np.empty(self.size, dtype=float)
        # holders for the one that has to interact with everything
        long_in_this_annulus = np.zeros(self.size, dtype=bool)
        # loop over embedded boundaries
        for ind, ebdy in enumerate(ebdyc):
            bx = ebdy.bdy.x
            by = ebdy.bdy.y
            width = ebdy.radial_width
            interior = ebdy.interior
            has_danger_zone = self.has_danger_zone[ind]
            if has_danger_zone:
                dz = self.dzl[ind]
                gi = self.gil[ind]
            else:
                dz = None
                gi = None
            # find coordinates for danger points
            check_x = self.x[dz]
            check_y = self.y[dz]
            res = compute_local_coordinates(bx, by, check_x, check_y,
                newton_tol=ebdy.coordinate_tolerance,
                interpolation_scheme=ebdy.coordinate_scheme,
                guess_ind=gi, verbose=False)
            t = res[0]
            r = res[1]
            # fix r values, if required
            if self.fix_r:
                ebdy.fix_r(r)
            full_t[dz] = t
            full_r[dz] = r
            # check on in annulus
            in_this_annulus, phys_not_in_this_annulus,  \
                                exterior = ebdy.check_if_r_in_annulus(r)
            # this is a plot to check on how this looks...
            # assuming the points come from new_ebdyc
            if False:
                fig, ax = plt.subplots()
                ax.pcolormesh(grid.xg, grid.yg, ebdyc.phys)
                ax.plot(ebdyc[0].bdy.x, ebdyc[0].bdy.y, color='gray')
                ax.plot(ebdyc[0].interface.x, ebdyc[0].interface.y, color='gray')
                ax.plot(new_ebdyc[0].bdy.x, new_ebdyc[0].bdy.y, color='black')
                ax.plot(new_ebdyc[0].interface.x, new_ebdyc[0].interface.y, color='black')
                ax.scatter(check_x[in_this_annulus], check_y[in_this_annulus], color='white')
                ax.scatter(check_x[exterior], check_y[exterior], color='orange')
                ax.scatter(check_x[phys_not_in_this_annulus], check_y[phys_not_in_this_annulus], color='blue')

            # set indicators in zone1 for those owned by this boundary
            zone1[dz] = np.logical_and(zone1[dz], phys_not_in_this_annulus)
            zone1_or_2[dz] = np.logical_and(zone1_or_2[dz], np.logical_not(exterior))

            # now save the required information in the zone2 and zone3 lists
            zone2l.append(dz[in_this_annulus])
            rhere = r[in_this_annulus]
            zone2r.append(rhere)
            zone2_transfr.append(ebdy.nufft_transform_r(rhere))
            zone2t.append(t[in_this_annulus])

            zone3l.append(dz[exterior])
            zone3r.append(r[exterior])
            zone3t.append(t[exterior])

        # plot to check on zone1 / zone1_or_2
        if False:
            fig, ax = plt.subplots()
            ax.pcolormesh(grid.xg, grid.yg, ebdyc.phys)
            ax.plot(ebdyc[0].bdy.x, ebdyc[0].bdy.y, color='gray')
            ax.plot(ebdyc[0].interface.x, ebdyc[0].interface.y, color='gray')
            ax.plot(new_ebdyc[0].bdy.x, new_ebdyc[0].bdy.y, color='black')
            ax.plot(new_ebdyc[0].interface.x, new_ebdyc[0].interface.y, color='black')
            ax.scatter(self.x[zone1], self.y[zone1], color='white')
            nz1 = np.logical_not(zone1)
            ax.scatter(self.x[nz1], self.y[nz1], color='orange')

            fig, ax = plt.subplots()
            ax.pcolormesh(grid.xg, grid.yg, ebdyc.phys)
            ax.plot(ebdyc[0].bdy.x, ebdyc[0].bdy.y, color='gray')
            ax.plot(ebdyc[0].interface.x, ebdyc[0].interface.y, color='gray')
            ax.plot(new_ebdyc[0].bdy.x, new_ebdyc[0].bdy.y, color='black')
            ax.plot(new_ebdyc[0].interface.x, new_ebdyc[0].interface.y, color='black')
            ax.scatter(self.x[zone1_or_2], self.y[zone1_or_2], color='white')
            nz1_or_2 = np.logical_not(zone1_or_2)
            ax.scatter(self.x[nz1_or_2], self.y[nz1_or_2], color='orange')

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
        self.zone1_N = np.sum(self.zone1)
        self.zone2_Ns = [len(z2) for z2 in self.zone2l]
        self.zone3_Ns = [len(z3) for z3 in self.zone3l]
        self.zone2_N = np.sum(self.zone2_Ns)
        self.zone3_N = np.sum(self.zone3_Ns)

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
        self.registered_xs = []
        self.registered_ys = []
        self.registered_partitions = []

        # information for solvers that require bumpy functions
        self.bumpy_readied = False

    def __iter__(self):
        return (ebdy for ebdy in self.ebdys)
    def __len__(self):
        return len(self.ebdys)
    def __getitem__(self, ind):
        return self.ebdys[ind]

    def register_grid(self, grid, danger_zone_distance=None, verbose=False):
        """
        Register a grid object

        grid: grid of type(Grid) from pybie2d (see pybie2d doc)
        verbose (optional):
            bool, whether to pass verbose output onto gridpoints_near_curve
        """
        self.grid = grid

        # generate memory for phys/close arrays
        close        = np.zeros(grid.shape, dtype=bool)
        int_helper1  = np.zeros(grid.shape, dtype=int)
        int_helper2  = np.zeros(grid.shape, dtype=int)
        float_helper = np.full(grid.shape, np.Inf, dtype=float)
        bool_helper  = np.zeros(grid.shape, dtype=bool)
        for ei, ebdy in enumerate(self):
            index = ei + 1
            if verbose:
                print('Regisgering ebdy #', index, 'of', self.N)
            ebdy.register_grid(self.grid, close, int_helper1, int_helper2, 
                float_helper, bool_helper, index,
                danger_zone_distance=danger_zone_distance, verbose=verbose)

        # now get the physical region
        phys = np.zeros(grid.shape, dtype=bool) if self.ebdys[0].interior \
                    else np.ones(grid.shape, dtype=bool)
        for ebdy in self:
            points_inside_curve_update(self.grid.xv, self.grid.yv, 
                ebdy.near_curve_result, phys, inside=ebdy.interior)
        self.phys = phys
        self.ext = np.logical_not(self.phys)

        # Fourier operators
        kxv = np.fft.fftfreq(grid.Nx, grid.xh/(2*np.pi))
        kyv = np.fft.fftfreq(grid.Ny, grid.yh/(2*np.pi))
        self.kx, self.ky = np.meshgrid(kxv, kyv, indexing='ij')
        self.ikx, self.iky = 1j*self.kx, 1j*self.ky
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
        # danger zone informatioin for pna
        if danger_zone_distance is not None:
            pna_ordering = np.full(self.grid.shape, -1, dtype=int)
            pna_ordering[self.phys_not_in_annulus] = np.arange(self.grid_pna_num)
            self.danger_zone_list = []
            self.guess_ind_list = []
            start_ind = self.grid_pna_num
            for ind, ebdy in enumerate(self):
                # points in phys/not annulus
                idx = ebdy.grid_in_danger_zone_x
                idy = ebdy.grid_in_danger_zone_y
                gi  = ebdy.grid_in_danger_zone_gi
                short_pna_ordering = pna_ordering[idx, idy]
                sel = short_pna_ordering != -1
                dz1 = short_pna_ordering[sel]
                gz1 = gi[sel]
                # points in associated radial region
                tot = np.prod(ebdy.radial_shape)
                dz2 = start_ind + np.arange(tot)
                gz2 = ebdy.near_radial_guess_inds.ravel()
                start_ind += tot
                # put these together
                dz = np.concatenate([dz1, dz2])
                gz = np.concatenate([gz1, gz2])
                # append lists
                self.danger_zone_list.append(dz)
                self.guess_ind_list.append(gz)

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

        # interface --> gridpoint/radial effective source points
        grid_source_list = [ebdy.interface_grid_source for ebdy in self.ebdys]
        radial_source_list = [ebdy.interface_radial_source for ebdy in self.ebdys]
        self.grid_source = merge_sources(grid_source_list)
        self.radial_source = merge_sources(radial_source_list)

        # bdy --> gridpoint/radial effective source points
        bdy_inward_source_list = [ebdy.bdy_inward_source for ebdy in self.ebdys]
        bdy_outward_source_list = [ebdy.bdy_outward_source for ebdy in self.ebdys]
        self.bdy_inward_sources = merge_sources(bdy_inward_source_list)
        self.bdy_outward_sources = merge_sources(bdy_outward_source_list)

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
        fr_list = f.get_radial_value_list()
        fb_list = [ebdy.interpolate_radial_to_boundary(fr) for ebdy, fr in zip(self.ebdys, fr_list)]
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
        diagnostic = finufftpy.nufft2d2(self.interfaces_x_transf, self.interfaces_y_transf, out, 1, 1e-14, funch, modeord=1)
        return out.real/np.prod(funch.shape)
    def update_radial_to_grid(self, f):
        _ = self.interpolate_radial_to_grid(f.radial_value_list, f.grid_value)
    def interpolate_radial_to_grid(self, fr_list, f=None):
        return [ebdy.interpolate_radial_to_grid(fr, f) for ebdy, fr in zip(self.ebdys, fr_list)]
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
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        radial_list = []
        for ebdy in self:
            radial_list.append(interp(ebdy.radial_x, ebdy.radial_y))
        return radial_list

    ############################################################################
    # Functions for dealing with interpolation to generic sets of points
    def get_interpolation_key(self, x, y, fix_r=False, dzl=None, gil=None):
        here = [p.is_it_i(x, y, fix_r) for p in self.registered_partitions]
        where = np.where(here)[0]
        if len(where) > 0:
            key = where[0]
        else:
            key = self._register_points(x, y, fix_r, dzl, gil)
        return key
    def register_points(self, x, y, fix_r=False, dzl=None, gil=None):
        return self.get_interpolation_key(x, y, fix_r, dzl, gil)
    def _register_points(self, x, y, fix_r=False, dzl=None, gil=None):
        k = len(self.registered_partitions)
        p = EmbeddedPointPartition(self, x, y, fix_r, dzl, gil)
        self.registered_partitions.append(p)
        return k
    def interpolate_to_points(self, ff, x, y, fix_r=False, dzl=None, gil=None):
        key = self.get_interpolation_key(x, y, fix_r, dzl, gil)
        p = self.registered_partitions[key]
        # get the category numbers
        c1n, c2n, c3n = p.get_Ns()
        # initialize output vector
        output = np.empty(x.size)
        # interpolate appropriate portion with grid (polynomial, for now...)
        if c1n > 0:
            if False:
                zone1 = p.zone1
                f = ff.get_grid_value()
                grid = self.grid
                lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
                ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
                hs =   [self.grid.xh, self.grid.yh]
                interp = fast_interp.interp2d(lbds, ubds, hs, f, k=7, p=[True, True])
                output[zone1] = interp(x[zone1], y[zone1])
            else:
                # HERE
                zone1 = p.zone1
                funch = np.fft.fft2(ff.get_smoothed_grid_value())
                out = np.zeros(p.zone1_N, dtype=complex)
                diagnostic = finufftpy.nufft2d2(p.x_transf, p.y_transf, out, 1, 1e-14, funch, modeord=1)
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
            fxr, fyr = self.ebdys[i].gradient(fx, fy, fr_list[i])
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
    def ready_bump(self, bump, bump_loc, bump_width):
        grr = np.hypot(self.grid.xg-bump_loc[0], self.grid.yg-bump_loc[1])
        bumpy = bump(affine_transformation(grr, 0, bump_width, 0, 1))
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

# class EmbeddedFunction(np.ndarray):
#     """
#     Updated version of the EmbeddedFunction Class
#     Subclasses ndarray, instead of MaskedArray; defines only the actual physical
#     data for the background grid, and will return a MaskedArray upon request

#     Goal: Handle vector/tensor valued functions seamlessly!
#     """
#     def __new__(cls, ebdyc, dtype=float):
#         gn = ebdyc.grid_phys.N
#         rn = np.sum([np.product(ebdy.radial_shape) for ebdy in ebdyc])
#         n = gn + rn
#         obj = super(EmbeddedFunction, cls).__new__(cls, n, dtype)
#         # extra things required for this class
#         obj.ebdyc = ebdyc
#         obj._generate()
#         return obj
#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.ebdyc = getattr(obj, 'ebdyc', None)
#         self._generate()
#     def _generate(self):
#         self.n_grid = self.ebdyc.grid_phys.N
#         self.n_data = self.size
#         # accessor for grid/radial data into masked data
#         self.gdata = self[:self.n_grid]
#         self.rdata = self[self.n_grid:]
#         # easy accessors to properly shaped/masked grid/radial arrays
#         self.radial_value_list = []
#         start = self.n_grid
#         for ebdy in self.ebdyc:
#             rsh = ebdy.radial_shape
#             end = start + np.product(rsh)
#             self.radial_value_list.append(self[start:end].reshape(rsh))
#             start = end
#     def __getitem__(self, arg):
#         if type(arg) == int:
#             return self.radial_value_list[arg]
#         else:
#             return self.view(np.ndarray).__getitem__(arg)
#     def __setitem__(self, arg, value):
#         if type(arg) == int:
#             self.radial_value_list[arg] = value
#         else:
#             self.view(np.ndarray).__setitem__(arg, value)
#     def load_data(self, grid_value, radial_value_list):
#         self[:self.n_grid] = grid_value
#         for ind, rv in enumerate(radial_value_list):
#             self[ind][:] = rv
#     def define_via_function(self, f):
#         gpx = self.ebdyc.grid_phys.x
#         gpy = self.ebdyc.grid_phys.y
#         self.gdata[:] = f(gpx, gpy)
#         for rv, ebdy in zip(self.radial_value_list, self.ebdyc.ebdys):
#             rv[:] = f(ebdy.radial_x, ebdy.radial_y)
#     def plot(self, ax, **kwargs):
#         grid = self.ebdyc.grid
#         xv = grid.xg
#         yv = grid.yg
#         xh = grid.xh
#         yh = grid.yh
#         gv = grid_value
#         vmin = self.min()
#         vmax = self.max()
#         if 'vmin' not in kwargs:
#             kwargs['vmin'] = vmin
#             kwargs['vmax'] = vmax
#         clf = ax.pcolor(xv-0.5*xh, yv-0.5*yh, gv, **kwargs)
#         for ebdy, fr in zip(self.ebdyc.ebdys, self.radial_value_list):
#             x = ebdy.plot_radial_x
#             y = ebdy.plot_radial_y
#             ax.pcolor(x, y, fr, **kwargs)
#         return clf    
#     def get_smoothed_grid_value(self):
#         arr = np.zeros(self.ebdyc.grid.shape, dtype=self.dtype)
#         arr[self.ebdyc.phys] = self.gdata
#         arr *= self.ebdyc.grid_step
#         return arr
#     def __str__(self):
#         string = 'Physical Grid Values:\n' + self.gdata.__str__() + \
#             '\nRadial Values (first 5 embedded boundaries):\n'
#         for i in range(min(len(self.radial_value_list), 5)):
#             string += 'Embedded Boundary ' + str(i) + '\n'
#             string += self.radial_value_list[i].__str__()
#             if i < 4: string += '\n'
#         return string
#     def __repr__(self):
#         return self.__str__()
#     def copy(self):
#         copy = EmbeddedFunction(self.ebdyc, self.dtype)
#         copy[:] = self
#         return copy

class EmbeddedFunction(np.ma.MaskedArray):
    def __new__(cls, ebdyc, dtype=float):
        gn = np.product(ebdyc.grid.shape)
        rn = np.sum([np.product(ebdy.radial_shape) for ebdy in ebdyc.ebdys])
        n = gn + rn
        ext = np.concatenate([ ebdyc.ext.ravel(), np.zeros(rn) ])
        self = super(EmbeddedFunction, cls).__new__(cls, np.zeros(n, dtype=dtype), mask=ext)
        # extra things required for this class
        self.ebdyc = ebdyc
        self.RN = len(self.ebdyc.ebdys)
        self.grid = ebdyc.grid
        self.sh = self.grid.shape
        self.phys = ebdyc.phys
        self.ext = ebdyc.ext
        self.gpx = ebdyc.grid_phys.x
        self.gpy = ebdyc.grid_phys.y
        self.n_grid = np.product(self.sh)
        self.radial_shape_list = [ebdy.radial_shape for ebdy in self.ebdyc.ebdys]
        self.n_radial_list = [np.product(sh) for sh in self.radial_shape_list]
        self.n_radial = np.sum(self.n_radial_list)
        self.n_data = self.n_grid + self.n_radial
        self.dof = ebdyc.grid_phys.N + self.n_radial
        self._generate_accessors()
        return self
    def __getitem__(self, ind):
        return self.radial_value_list[ind]
    def __setitem__(self, ind, value):
        self.radial_value_list[ind] = value
    def _generate_accessors(self):
        # accessor for grid/radial data into masked data
        self.gdata = np.ma.array(self.data[:self.n_grid], mask=self.mask[:self.n_grid])
        self.rdata = self.data[self.n_grid:]
        # easy accessors to properly shaped/masked grid/radial arrays
        self.grid_value = np.ma.array(self.data[:self.n_grid].reshape(self.sh), mask=self.ext)
        self.radial_value_list = []
        start = self.n_grid
        for i in range(self.RN):
            sh = self.radial_shape_list[i]
            n = self.n_radial_list[i]
            end = start + n
            self.radial_value_list.append(self.data[start:end].reshape(sh))
            start = end
    def _update_from(self, obj):
        super(EmbeddedFunction, self)._update_from(obj)
        if hasattr(obj, '_mask') and not hasattr(self, '_mask'): self._mask = obj._mask
        if hasattr(obj, '_mask') and not hasattr(self, '_mask'): self._mask = obj._mask
        if hasattr(obj, 'ebdyc'):
            # copy required objects from obj to self
            self.ebdyc             = obj.ebdyc
            self.RN                = obj.RN
            self.grid              = obj.grid
            self.sh                = obj.grid.shape
            self.gpx               = obj.gpx
            self.gpy               = obj.gpy
            self.n_grid            = obj.n_grid
            self.radial_shape_list = obj.radial_shape_list
            self.n_radial_list     = obj.n_radial_list
            self.n_radial          = obj.n_radial
            self.n_data            = obj.n_data
            self.dof               = obj.dof
            self.ext = self._mask[:self.n_grid].reshape(self.sh)
            self.phys = np.logical_not(self.ext).reshape(self.sh)
            # regenerate accessors
            self._generate_accessors()
    def load_data(self, grid_value, radial_value_list):
        dl = [grid_value.ravel(),]
        for rv in radial_value_list:
            dl.append(rv.ravel())
        self.data[:] = np.concatenate(dl)
    def define_via_function(self, f):
        self.grid_value[self.phys] = f(self.gpx, self.gpy)
        for rv, ebdy in zip(self.radial_value_list, self.ebdyc.ebdys):
            rv[:] = f(ebdy.radial_x, ebdy.radial_y)
    def zero(self):
        self.data[:] = 0.0
    def plot(self, ax, **kwargs):
        xv = self.grid.xg
        yv = self.grid.yg
        xh = self.grid.xh
        yh = self.grid.yh
        gv = self.grid_value
        vmin = self.min()
        vmax = self.max()
        if 'vmin' not in kwargs:
            kwargs['vmin'] = vmin
            kwargs['vmax'] = vmax
        clf = ax.pcolor(xv-0.5*xh, yv-0.5*yh, gv, **kwargs)
        for ebdy, fr in zip(self.ebdyc.ebdys, self.radial_value_list):
            x = ebdy.plot_radial_x
            y = ebdy.plot_radial_y
            ax.pcolor(x, y, fr, **kwargs)
        return clf    
    def get_radial_value_list(self):
        return self.radial_value_list
    def get_grid_value(self):
        return self.grid_value
    def get_smoothed_grid_value(self):
        arr = np.array(self.grid_value)
        arr[self.ext] = 0.0
        return arr*self.ebdyc.grid_step
    def get_grid_phys_value(self):
        return self.grid_value[self.phys]
    def get_components(self):
        return self.grid_value, self.get_grid_phys_value(), self.radial_value_list
    def radial_min(self):
        return np.min(self.rdata)
    def radial_max(self):
        return np.max(self.rdata)
    def grid_min(self):
        return np.min(self.gdata)
    def grid_max(self):
        return np.max(self.gdata)
    def __str__(self):
        string = 'Physical Grid Values:\n' + self.get_grid_phys_value().__str__() + \
            '\nRadial Values (first 5 embedded boundaries):\n'
        for i in range(min(len(self.radial_value_list), 5)):
            string += 'Embedded Boundary ' + str(i) + '\n'
            string += self.radial_value_list[i].__str__()
            if i < 4: string += '\n'
        return string
    def __repr__(self):
        return self.__str__()
    def copy(self):
        copy = EmbeddedFunction(self.ebdyc, self.dtype)
        copy.data[:] = self.data
        return copy
    def get_linear_data(self):
        return self.data[~self.mask]
    def load_linear_data(self, data):
        self.data[:] = 0.0
        self.data[~self.mask] = data

class BoundaryFunction(object):
    # I should make this subclass np.ndarray at some point?
    def __init__(self, ebdyc):
        self.ebdyc = ebdyc
        self.defined = False
    def __getitem__(self, ind):
        return self.bdy_value_list[ind]
    def __setitem__(self, ind, value):
        self.bdy_value_list[ind] = value
    def load_data(self, bdy_value_list):
        self.bdy_value_list = bdy_value_list
        self.defined = True
    def define_via_function(self, f):
        self.bdy_value_list = [f(ebdy.bdy.x, ebdy.bdy.y) for ebdy in self.ebdyc.ebdys]
        self.defined = True
    def define_via_functions(self, f_list):
        self.bdy_value_list = [f(ebdy.bdy.x, ebdy.bdy.y) for f, ebdy in zip(f_list, self.ebdyc.ebdys)]
        self.defined = True
    def aggregate(self):
        return np.concatenate(self.bdy_value_list)
    def __add__(self, other):
        out = BoundaryFunction(self.ebdyc)
        outv = [s + o for s, o in zip(self.bdy_value_list, other.bdy_value_list)]
        out.load_data(outv)
        return out
    def __radd__(self, other):
        return self + other
    def __neg__(self):
        out = BoundaryFunction(self.ebdyc)
        outv = [-s for s in self.bdy_value_list]
        out.load_data(outv)
        return out
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return -self + other
    def __abs__(self):
        out = BoundaryFunction(self.ebdyc)
        outv = [np.abs(s) for s in self.bdy_value_list]
        out.load_data(outv)
        return out
    def __str__(self):
        if self.defined:
            string = '\nBoundary Values (first 5 boundaries):\n'
            for i in range(min(len(self.bdy_value_list), 5)):
                string += 'Boundary ' + str(i) + '\n'
                string += self.bdy_value_list[i].__str__()
                if i < 4: string += '\n'
            return string
        else:
            return 'BoundaryFunction not yet defined.'
    def __repr__(self):
        return self.__str__()
    def copy(self):
        copy = BoundaryFunction(self.ebdyc)
        if self.defined:
            copy.load_data(self.bdy_value_list)
        return copy

            


