import numpy as np
import pybie2d
import weakref
from ipde.embedded_boundary import EmbeddedBoundary

def LoadEmbeddedFunction(d, ebdyc=None):
    from ipde.ebdy_collection import LoadEmbeddedBoundaryCollection
    if ebdyc is None:
        if 'ebdyc_dict' not in d:
            raise Exception('Need ebdyc provided unless save was generated with full_save method.')
        ebdyc = LoadEmbeddedBoundaryCollection(d['ebdyc_dict'])
    ef = EmbeddedFunction(ebdyc)
    ef.load_linear_data(d['linear_data'])
    return ef, ebdyc

class EmbeddedFunction(np.ndarray):
    """
    Updated version of the EmbeddedFunction Class
    Subclasses ndarray, instead of MaskedArray; defines only the actual physical
    data for the background grid, and will return a MaskedArray upon request

    Goal: Handle vector/tensor valued functions seamlessly!
    """
    def __new__(cls, ebdyc, dtype=float, array=None, function=None, grid_value=None, radial_value_list=None, linear_data=None, zero=False):
        gn = ebdyc.grid_phys.N
        rn = np.sum([np.product(ebdy.radial_shape) for ebdy in ebdyc])
        n = gn + rn
        if array is None:
            array = super(EmbeddedFunction, cls).__new__(cls, n, dtype)
        array.ebdyc = weakref.ref(ebdyc)
        out = array.view(EmbeddedFunction)
        if function is not None:
            out.define_via_function(function)
        if grid_value is not None and radial_value_list is not None:
            out.load_data(grid_value, radial_value_list)
        if grid_value is not None and radial_value_list is None:
            out.load_full_grid(grid_value)
        if linear_data is not None:
            out.load_linear_data(linear_data)
        if zero:
            out.zero()
        return out
        # return array.view(EmbeddedFunction)
    def __array_finalize__(self, obj):
        if obj is None: return
        self.ebdyc = getattr(obj, 'ebdyc', None)
        self._generate()
    def _ebdyc_test(self):
        ebdyc = self.ebdyc()
        if ebdyc is not None:
            return ebdyc
        else:
            raise Exception('Underlying ebdyc has been deleted.')
    def save(self):
        return {'linear_data': self.view(np.ndarray)}
    def full_save(self):
        d = {
            'ebdyc_dict'  : self.ebdyc().save(),
            'linear_data' : self.view(np.ndarray),
        }
        return d
    def _generate(self):
        ebdyc = self._ebdyc_test()
        self.n_grid = ebdyc.grid_phys.N
        self.n_data = self.size
        # accessor for grid/radial data into masked data
        self.gslice = slice(0, self.n_grid)
        self.rslice = slice(self.n_grid, None)
        # easy accessors to properly shaped/masked grid/radial arrays
        self.radial_slices = []
        self.radial_shapes = []
        start = self.n_grid
        for ebdy in ebdyc:
            rsh = ebdy.radial_shape
            end = start + np.product(rsh)
            self.radial_slices.append(slice(start, end))
            self.radial_shapes.append(rsh)
            start = end
        self.radial_length = len(self.radial_shapes)
    def get_gdata(self):
        return super().__getitem__(slice(0, self.n_grid)).view(np.ndarray)
    def get_rdata(self):
        return super().__getitem__(slice(self.n_grid, None)).view(np.ndarray)
    def __getitem__(self, arg):
        if type(arg) == int:
            sl = self.radial_slices[arg]
            rsh = self.radial_shapes[arg]
            return super().__getitem__(sl).view(np.ndarray).reshape(rsh)
        elif arg == 'grid':
            return super().__getitem__(self.gslice).view(np.ndarray)
        else:
            return super().__getitem__(arg).view(np.ndarray)
    def __setitem__(self, arg, value):
        if type(arg) == int:
            arg = self.radial_slices[arg]
            super().__setitem__(arg, value.ravel())
        elif arg == 'grid':
            super().__setitem__(self.gslice, value)
        else:
            super().__setitem__(arg, value)
    def min(self):
        return np.min(self.data)
    def max(self):
        return np.max(self.data)
    def load_data(self, grid_value, radial_value_list):
        gd = self.get_gdata()
        try:
            gd[:] = grid_value
        except:
            ebdyc = self._ebdyc_test()
            gd[:] = grid_value[ebdyc.phys]
        for arg, rv in enumerate(radial_value_list):
            self.__setitem__(arg, rv)
    def load_full_grid(self, grid_value):
        """
        Load data from a set of values defined on the *whole* grid
        if it's not defined on the whole grid, this won't be accurate
        """
        ebdyc = self._ebdyc_test()
        rvl = ebdyc.interpolate_grid_to_radial(grid_value, order=5)
        self.load_data(grid_value, rvl)
    def load_linear_data(self, data):
        # should this make a copy or a replacement????
        # probably safer to make a copy...
        super().__setitem__(slice(None,None), data)
    def define_via_function(self, f):
        ebdyc = self._ebdyc_test()
        gpx = ebdyc.grid_phys.x
        gpy = ebdyc.grid_phys.y
        self.get_gdata()[:] = f(gpx, gpy)
        for arg, ebdy in enumerate(ebdyc):
            self.__setitem__(arg, f(ebdy.radial_x, ebdy.radial_y))
    def get_radial_value_list(self):
        return [self.__getitem__(arg) for arg in range(self.radial_length)]
    def get_components(self):
        ebdyc = self._ebdyc_test()
        gv = self.get_grid_value()
        return gv, gv*ebdyc.grid_step, self.get_radial_value_list()
    def plot(self, ax, **kwargs):
        ebdyc = self._ebdyc_test()
        grid = ebdyc.grid
        xv = grid.xg
        yv = grid.yg
        xh = grid.xh
        yh = grid.yh
        gv_raw = np.empty(grid.shape)
        gv_raw[ebdyc.phys] = self['grid']
        gv = np.ma.array(gv_raw, mask=ebdyc.ext)
        vmin = self.min()
        vmax = self.max()
        if 'vmin' not in kwargs:
            kwargs['vmin'] = vmin
            kwargs['vmax'] = vmax
        clf = ax.pcolormesh(xv-0.5*xh, yv-0.5*yh, gv, **kwargs)
        for ebdy, fr in zip(ebdyc, self):
            x = ebdy.plot_radial_x
            y = ebdy.plot_radial_y
            ax.pcolormesh(x, y, fr, **kwargs)
        return clf
    def get_grid_value(self, masked=False):
        ebdyc = self._ebdyc_test()
        arr = np.zeros(ebdyc.grid.shape, dtype=self.dtype)
        arr[ebdyc.phys] = self.get_gdata()
        if masked: arr = np.ma.array(arr, mask=ebdyc.ext)
        return arr
    def get_smoothed_grid_value(self):
        ebdyc = self._ebdyc_test()
        arr = self.get_grid_value()
        arr *= ebdyc.grid_step
        return arr
    def __str__(self):
        string = 'Physical Grid Values:\n' + self.get_gdata().__str__() + \
            '\nRadial Values (first 5 embedded boundaries):\n'
        for i in range(min(self.radial_length, 5)):
            string += 'Embedded Boundary ' + str(i) + '\n'
            string += self.__getitem__(i).__str__()
            if i < 4: string += '\n'
        return string
    def __repr__(self):
        return self.__str__()
    def copy(self):
        ebdyc = self._ebdyc_test()
        copy = EmbeddedFunction(ebdyc, self.dtype)
        super(EmbeddedFunction, copy).__setitem__(slice(None, None), self)
        return copy
    def asarray(self):
        return np.array(self)
    def zero(self):
        super().__setitem__(slice(None, None), 0.0)
    def integrate(self):
        """
        Compute the volume integral of this function
        """
        return self._ebdyc_test().volume_integral(self)
    def gradient(self):
        return self._ebdyc_test().gradient(self)
    def get_ebdyc(self):
        return self._ebdyc_test()
    def extract_pnar(self):
        # there is a better way to do this...
        gv, _, rvl = self.get_components()
        gvp = gv[self._ebdyc_test().phys_not_in_annulus]
        rvlf = [rv.ravel() for rv in rvl]
        rv = np.concatenate(rvlf)
        return np.concatenate([gvp, rv])

class BoundaryFunction(np.ndarray):
    """
    Updated version of the BoundaryFunction Class
    Now subclasses np.ndarray
    """
    def __new__(cls, ebdyc, dtype=float, array=None, function=None, functions=None, data=None, zero=False):
        n = np.sum([ebdy.bdy.N for ebdy in ebdyc])
        if array is None:
            array = super(BoundaryFunction, cls).__new__(cls, n, dtype)
        array.ebdyc = weakref.ref(ebdyc)
        out = array.view(BoundaryFunction)
        out.defined = False
        if function is not None:
            out.define_via_function(function)
        if functions is not None:
            out.define_via_functions(functions)
        if data is not None:
            out.load_data(data)
        if zero:
            out.zero()
        return out
    def __array_finalize__(self, obj):
        if obj is None: return
        self.ebdyc = getattr(obj, 'ebdyc', None)
        self.defined = getattr(obj, 'defined', None)
        self._generate()
    def _ebdyc_test(self):
        ebdyc = self.ebdyc()
        if ebdyc is not None:
            return ebdyc
        else:
            raise Exception('Underlying ebdyc has been deleted.')
    def _generate(self):
        ebdyc = self._ebdyc_test()
        self.ns = [ebdy.bdy.N for ebdy in ebdyc]
        sum_ns = np.concatenate([(0,), np.cumsum(self.ns)])
        # accessor for individual components
        self.slices = []
        for i in range(ebdyc.N):
            self.slices.append( slice(sum_ns[i], sum_ns[i+1]) )
    def _locate_ebdy(self, e):
        is_it_i = [e is ebdy for ebdy in self.ebdyc]
        ind = np.where(is_it_i)[0]
        if type(ind) != int:
            raise Exception('EmbeddedBoundary not found in EmbeddedBoundaryCollection definining this BoundaryFunction')
        return ind
    def __getitem__(self, arg):
        if type(arg) == int:
            return super().__getitem__(self.slices[arg]).view(np.ndarray)
        elif type(arg) == EmbeddedBoundary:
            return self[self._locate_ebdy(ebdy)]
        else:
            return super().__getitem__(arg).view(np.ndarray)
    def __setitem__(self, arg, value):
        if type(arg) == int:
            super().__setitem__(self.slices[arg], value.ravel())
        elif type(arg) == EmbeddedBoundary:
            self[self._locate_ebdy(ebdy)] = value
        else:
            super().__setitem__(arg, value)
    def load_data(self, bdy_value_list):
        ebdyc = self._ebdyc_test()
        for ei, ebdy in enumerate(ebdyc):
            self[ei] = bdy_value_list[ei]
        self.bdy_value_list = bdy_value_list
        self.defined = True
    def define_via_function(self, f):
        ebdyc = self._ebdyc_test()
        for ei, ebdy in enumerate(ebdyc):
            self[ei] = f(ebdy.bdy.x, ebdy.bdy.y)
        self.defined = True
    def define_via_functions(self, f_list):
        ebdyc = self._ebdyc_test()
        for ind in range(ebdyc.N):
            ebdy = ebdyc[ind]
            self[ind] = f_list[ind](ebdy.bdy.x, ebdy.bdy.y)
        self.defined = True
    
    def min(self):
        return np.min(self.data)
    def max(self):
        return np.max(self.data)

    def __str__(self):
        ebdyc = self._ebdyc_test()
        if self.defined:
            string = '\nBoundary Values (first 5 boundaries):\n'
            for i in range(min(ebdyc.N, 5)):
                string += 'Boundary ' + str(i) + '\n'
                string += self[i].__str__()
                if i < 4: string += '\n'
            return string
        else:
            return 'BoundaryFunction not yet defined.'
    def __repr__(self):
        return self.__str__()

    def copy(self):
        return BoundaryFunction(self.ebdyc, data=self.asarray().copy())

    def asarray(self):
        return np.array(self)
    def zero(self):
        super().__setitem__(slice(None, None), 0.0)
        self.defined=True






