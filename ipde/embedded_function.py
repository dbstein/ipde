import numpy as np
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 800
ng = int(nb/2)
M = 4*int(nb/100)
M = max(4, M)
M = min(30, M)
pad_zone = 0
slepian_r = 1.5*M
grid_upsample = 1

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
bh = bdy.dt*bdy.speed.min()
grid = Grid([-np.pi/2, np.pi/2], ng, [-np.pi/2, np.pi/2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
ebdy = EmbeddedBoundary(bdy, True, M, grid.xh*0.75, pad_zone=pad_zone, heaviside=MOL.step, qfs_tolerance=1e-14)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
ebdyc.register_grid(grid)

class TensorEmbeddedFunction(np.ndarray):
    def __new__(cls, ebdyc, sh, dtype=float):
        fsh = tuple(list(sh) + [ebdyc.dof])
        self = super(TensorEmbeddedFunction, cls).__new__(cls, fsh, dtype)
        self.sh = sh
        self.fsh = fsh
        self.ebdyc = ebdyc
        return self
    def __array_finalize__(self, obj):
        if obj is None: return
        self.ebdyc = getattr(obj, 'ebdyc', None)
        self.sh    = getattr(obj, 'sh',    None)
        self.fsh   = getattr(obj, 'fsh',   None)
    def __getitem__(self, arg):
        arr = self.view(np.ndarray)[arg]
        return EmbeddedFunction(ebdyc=self.ebdyc, dtype=self.dtype, array=arr)
    def __setitem__(self, arg, value):
        self.view(np.ndarray)[arg] = value
        # arr = self.view(np.ndarray)[arg]
        # return EmbeddedFunction(self.ebdyc, dtype=self.dtype, buffer=arr.data)
    def __str__(self):
        string = 'Tensor of Embedded Functions'
        return string
    def __repr__(self):
        return self.__str__()

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
            obj = super(EmbeddedFunction, cls).__new__(cls, n, dtype)
        else:
            obj = array.view(EmbeddedFunction)
        print(type(obj))
        # extra things required for this class
        print('new')
        obj.ebdyc = ebdyc
        print('ebdyc', ebdyc, obj.ebdyc)
        obj._generate()
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.ebdyc = getattr(obj, 'ebdyc', None)
        self._generate()
        print('finalize')
    def _generate(self):
        self.n_grid = self.ebdyc.grid_phys.N
        self.n_data = self.size
        # accessor for grid/radial data into masked data
        self.gdata = self[:self.n_grid]
        self.rdata = self[self.n_grid:]
        # easy accessors to properly shaped/masked grid/radial arrays
        self.radial_value_list = []
        start = self.n_grid
        for ebdy in self.ebdyc:
            rsh = ebdy.radial_shape
            end = start + np.product(rsh)
            self.radial_value_list.append(self[start:end].reshape(rsh))
            start = end
    def __getitem__(self, arg):
        if type(arg) == int:
            return self.radial_value_list[arg]
        else:
            return self.view(np.ndarray).__getitem__(arg)
    def __setitem__(self, arg, value):
        if type(arg) == int:
            self.radial_value_list[arg] = value
        else:
            self.view(np.ndarray).__setitem__(arg, value)
    def load_data(self, grid_value, radial_value_list):
        self[:self.n_grid] = grid_value
        for ind, rv in enumerate(radial_value_list):
            self[ind][:] = rv
    def define_via_function(self, f):
        gpx = self.ebdyc.grid_phys.x
        gpy = self.ebdyc.grid_phys.y
        self.gdata[:] = f(gpx, gpy)
        for rv, ebdy in zip(self.radial_value_list, self.ebdyc.ebdys):
            rv[:] = f(ebdy.radial_x, ebdy.radial_y)
    def plot(self, ax, **kwargs):
        grid = self.ebdyc.grid
        xv = grid.xg
        yv = grid.yg
        xh = grid.xh
        yh = grid.yh
        gv = grid_value
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
    def get_smoothed_grid_value(self):
        arr = np.zeros(self.ebdyc.grid.shape, dtype=self.dtype)
        arr[self.ebdyc.phys] = self.gdata
        arr *= self.ebdyc.grid_step
        return arr
    def __str__(self):
        string = 'Physical Grid Values:\n' + self.gdata.__str__() + \
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

f = EmbeddedFunction(ebdyc)
f.define_via_function(np.sin)
f.load_data(f.gdata, f.radial_value_list)
g = np.sin(f)

D = np.empty([2,2], dtype=object)
D[0,0] = f
D[0,1] = f
D[1,0] = f
D[1,1] = f

D = TensorEmbeddedFunction(ebdyc, [2,2])
D[0,0] = 1
D[0,1] = 2
D[1,0] = 3
D[1,1] = 4

E = np.einsum('ij...,jk...->ik...', D, D)



