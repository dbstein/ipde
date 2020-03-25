import numpy as np
from .scalar import ScalarSolver
from ..internals.poisson import PoissonHelper
from ...utilities import affine_transformation

class PoissonSolver(ScalarSolver):
    def __init__(self, ebdyc, solver_type='spectral', AS_list=None):
        # check that ebdyc has bumps defined
        # if not ebdyc.bumpy_readied:
            # raise Exception('Laplace solver requires embedded boundary collection with a bump function.')
        super().__init__(ebdyc, solver_type, AS_list)
    def _get_helper(self, ebdy, AS):
        return PoissonHelper(ebdy, AS)
    def _grid_solve(self, fc):
        fc = self.ebdyc.demean_function(fc)
        return np.fft.ifft2(np.fft.fft2(fc)*self.ilap).real
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.lap[0,0] = np.Inf
        self.ilap = 1.0/self.lap
        self.lap[0,0] = 0.0
