import numpy as np
import pybie2d
import weakref

class EmbeddedFunction(np.ndarray):
    """
    Updated version of the EmbeddedFunction Class
    Subclasses ndarray, instead of MaskedArray; defines only the actual physical
    data for the background grid, and will return a MaskedArray upon request

    Goal: Handle vector/tensor valued functions seamlessly!
    """
    def __new__(cls, ebdyc, dtype=float, array=None):
        gn = ebdyc.grid_phys.N
        rn = np.sum([np.product(ebdy.radial_shape) for ebdy in ebdyc])
        n = gn + rn
        if array is None:
            array = super(EmbeddedFunction, cls).__new__(cls, n, dtype)
        array.ebdyc = weakref.ref(ebdyc)
        return array.view(EmbeddedFunction)
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
