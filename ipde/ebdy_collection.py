import numpy as np
import pybie2d
import fast_interp
import finufftpy
PointSet = pybie2d.point_set.PointSet
from .embedded_boundary import EmbeddedBoundary
from .utilities import affine_transformation
from .annular.annular import ApproximateAnnularGeometry
from .derivatives import fd_x_4, fd_y_4, fourier

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

class EmbeddedBoundaryCollection(object):
    def __init__(self, ebdy_list):
        """
        Collection of embedded boundaries

        ebdy_list (required):
            list[type(EmbeddedBoundary)]
        """
        self.ebdys = ebdy_list
        self.N = len(self.ebdys)

    def register_grid(self, grid=None, danger_zone_distance=None, verbose=False):
        """
        Register a grid object

        grid (optional):
            grid of type(Grid) from pybie2d (see pybie2d doc)
            if None, will fetch the grid from the first ebdy
            and assumes all ebdys have already registered the grid
        verbose (optional):
            bool, whether to pass verbose output onto gridpoints_near_curve
        """
        if grid is None:
            self.grid = self.ebdys[0].grid
        else:
            self.grid = grid
            for ei, ebdy in enumerate(self.ebdys):
                if verbose:
                    print('Regisgering ebdy #', ei+1, 'of', self.N)
                ebdy.register_grid(self.grid, danger_zone_distance, verbose)

        # compute some basic things so they don't have to be redone
        grid = self.ebdys[0].grid
        kxv = np.fft.fftfreq(grid.Nx, grid.xh/(2*np.pi))
        kyv = np.fft.fftfreq(grid.Ny, grid.yh/(2*np.pi))
        self.kx, self.ky = np.meshgrid(kxv, kyv, indexing='ij')
        self.ikx, self.iky = 1j*self.kx, 1j*self.ky

        # construct the full phys/ext variables
        # this is memory wasteful, can be improved!
        phys_list = [ebdy.phys for ebdy in self.ebdys]
        self.phys = np.logical_and.reduce(phys_list)
        self.ext = np.logical_not(self.phys)

        # construct the full in_annulus
        in_annulus_list = [ebdy.grid_in_annulus for ebdy in self.ebdys]
        self.in_annulus = np.logical_and.reduce(in_annulus_list)
        self.phys_not_in_annulus = np.logical_and(self.phys, np.logical_not(self.in_annulus))

        # construct the full grid_step
        self.grid_step = np.ones(grid.shape, dtype=float)
        for ebdy in self.ebdys:
            self.grid_step *= ebdy.grid_step

        # compute the point evaluation sets

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
        if cutoff == None: cutoff = order == np.Inf
        # cutoff, if cutting off
        fc = f*self.grid_step if cutoff else f
        if order == np.Inf:
            return self.nufft_interpolate_grid_to_interface(fc)
        else:
            return self.poly_interpolate_grid_to_interface(fc, order)
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
        return interp(self.all_ivx, self.all_ivy)
    def nufft_interpolate_grid_to_interface(self, f):
        """
        Interpolation from grid to inteface.  Note that f is not multiplied by
        grid_step in this function!  If f is not smooth on the whole grid, 
        first multiply it by grid_step to get an accurate result
        """
        funch = np.fft.fft2(f)
        out = np.zeros(self.interfaces_x_transf.size, dtype=complex)
        diagnostic = finufftpy.nufft2d2(self.interfaces_x_transf, self.interfaces_y_transf, out, 1, 1e-14, funch, modeord=1)
        return out.real/np.prod(funch.shape)
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
        """
        grid = self.grid
        lbds = [self.grid.x_bounds[0], self.grid.y_bounds[0]]
        ubds = [self.grid.x_bounds[1], self.grid.y_bounds[1]]
        hs =   [self.grid.xh, self.grid.yh]
        interp = fast_interp.interp2d(lbds, ubds, hs, f, k=order, p=[True, True])
        return interp(self.radial_x, self.radial_y)

    ############################################################################
    # Functions for taking derivatives on the radial grid
    def gradient(self, f, fr_list, xd_func, yd_func, cutoff=True):
        # cutoff, if cutting off
        fc = f*self.grid_step if cutoff else f
        # compute gradient on the background grid
        fx = xd_func(fc)
        fy = yd_func(fc)
        # compute on the radial grid
        fxrs, fyrs = [], []
        for i in range(self.N):
            fxr, fyr = self.ebdys[i].radial_grid_derivatives(fr_list[i])
            fxrs.append(fxr)
            fyrs.append(fyr)
        # interpolate from the radial grid to the background grid
        self.interpolate_radial_to_grid(fxrs, fx)
        self.interpolate_radial_to_grid(fyrs, fy)
        # set to 0 on regular grid in the exterior region
        fx *= self.phys
        fy *= self.phys
        return fx, fy, fxrs, fyrs
    def gradient2(self, ff, derivative_type='spectral'):
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
            fxr, fyr = self.ebdys[i].gradient2(fx, fy, fr_list[i])
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

class EmbeddedFunction(np.ma.MaskedArray):
    def __new__(cls, ebdyc, dtype=float):
        gn = np.product(ebdyc.grid.shape)
        rn = np.sum([np.product(ebdy.radial_shape) for ebdy in ebdyc.ebdys])
        n = gn + rn
        ext = np.concatenate([ ebdyc.ext.ravel(), np.zeros(rn) ])
        self = super(EmbeddedFunction, cls).__new__(cls, np.empty(n, dtype=dtype), mask=ext)
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
        self._generate_accessors()
        return self
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
        copy[:] = self
        return copy

class EmbeddedFunctionOld(object):
    """
    Representation of a function for embedded boundary collection class
    """
    def __init__(self, ebdyc, dtype=float):
        self.ebdyc = ebdyc
        self.RN = len(self.ebdyc.ebdys)
        self.grid = ebdyc.grid
        self.sh = self.grid.shape
        self.phys = ebdyc.phys
        self.ext = ebdyc.ext
        self.gpx = ebdyc.grid_phys.x
        self.gpy = ebdyc.grid_phys.y
        self.dtype = dtype
        # numbers for varius elements
        self.n_grid = np.product(self.sh)
        self.radial_shape_list = [ebdy.radial_shape for ebdy in self.ebdyc.ebdys]
        self.n_radial_list = [np.product(sh) for sh in self.radial_shape_list]
        self.n_radial = np.sum(self.n_radial_list)
        self.n_data = self.n_grid + self.n_data
        # sequential memory for all the values to live in
        self.data = np.empty(self.n_data, self.dtype)
        # linear phys variable
        self.linear_phys = np.concatenate([ self.phys.ravel(), np.ones(self.n_radial, dtype=bool) ])
        self.linear_ext = np.logical_not(self.linear_phys)
        # masked version of data
        self.masked_data = np.ma.array(self.data, mask=self.linear_ext)
        # accessor for grid/radial data into masked data
        self.gdata = self.masked_data[:self.n_grid]
        self.rdata = self.masked_data[self.n_grid:]
        # easy accessors to properly shaped/masked grid/radial arrays
        self.grid_value = np.ma.array(self.data[:self.n_grid].reshape(self.sh))
        self.radial_value_list = []
        start = self.n_grid
        for i in range(self.RN):
            sh = self.radial_shape_list[i]
            n = self.n_radial_list[i]
            end = start + n
            self.radial_value_list.append(self.data[start:end].reshape(sh))
            start = end
    def load_data(self, grid_value, radial_value_list):
        dl = [grid_value.ravel(),]
        for rv in radial_value_list:
            dl += rv.ravel()
        self._load_data(np.concatenate(dl))
    def _load_data(self, data):
        self.data[:] = data
    def define_via_function(self, f):
        self.grid_value[self.phys] = f(self.gpx, self.gpy)
        for rv, ebdy in zip(self.radial_value_list, self.ebdyc.ebdys):
            rv[:] = f(ebdy.radial_x, ebdy.radial_y)
    def zero(self, dtype=float):
        self.data *= 0.0
    def plot(self, ax, **kwargs):
        self.check()
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
    def min(self):
        return np.min(self.mdata)
    def max(self):
        return np.max(self.mdata)
    def _access_other(self, other):
        if type(other) == EmbeddedFunction:
            return 
    def __add__(self, other):
        o = other.data if type(other) == EmbeddedFunction else other
        if type(other) == EmbeddedFunction:
            data = self.data + other.data
        else:
            data = self.data + other
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __radd__(self, other):
        return self.__add__(other)
    def __neg__(self):
        self.check()
        gv = -self.grid_value
        rvs = [-rs for rs in self.radial_value_list]
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return -self + other
    def __mul__(self, other):
        self.check()
        if type(other) == EmbeddedFunction:
            other.check()
            gv = self.grid_value * other.grid_value
            rvs = [rs * ro for rs, ro in zip(self.radial_value_list, other.radial_value_list)]
        else:
            gv = self.grid_value * other
            rvs = [rs * other for rs in self.radial_value_list]
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        self.check()
        if type(other) == EmbeddedFunction:
            other.check()
            gv = self.grid_value / other.grid_value
            rvs = [rs / ro for rs, ro in zip(self.radial_value_list, other.radial_value_list)]
        else:
            gv = self.grid_value / other
            rvs = [rs / other for rs in self.radial_value_list]
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __rtruediv__(self, other):
        self.check()
        if type(other) == EmbeddedFunction:
            other.check()
            gpv = other.grid_value / self.grid_value
            rvs = [ro / rs for rs, ro in zip(self.radial_value_list, other.radial_value_list)]
        else:
            gv = other / self.grid_value
            rvs = [other / rs for rs in self.radial_value_list]
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __abs__(self):
        self.check()
        gv = np.abs(self.grid_value)
        rvs = [np.abs(rs) for rs in self.radial_value_list]
        out = EmbeddedFunction(self.ebdyc)
        out.load_data(gv, rvs)
        return out
    def __str__(self):
        if self.defined:
            string = 'Physical Grid Values:\n' + self.get_grid_phys_value().__str__() + \
                '\nRadial Values (first 5 embedded boundaries):\n'
            for i in range(min(len(self.radial_value_list), 5)):
                string += 'Embedded Boundary ' + str(i) + '\n'
                string += self.radial_value_list[i].__str__()
                if i < 4: string += '\n'
            return string
        else:
            return 'EmbeddedFunction not yet defined.'
    def __repr__(self):
        return self.__str__()
    def copy(self):
        copy = EmbeddedFunction(self.ebdyc)
        if self.defined:
            g, _, r = self.get_components()
            copy.load_data(g, r)
        return copy

class BoundaryFunction(object):
    # I should make this subclass np.ndarray at some point?
    def __init__(self, ebdyc):
        self.ebdyc = ebdyc
        self.defined = False
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

            


