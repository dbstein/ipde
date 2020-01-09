import numpy as np
from .scalar import ScalarSolver
from ..internals.modified_helmholtz import ModifiedHelmholtzHelper

class ModifiedHelmholtzSolver(ScalarSolver):
    def __init__(self, ebdyc, solver_type='spectral', AS_list=None, k=1.0):
        super().__init__(ebdyc, solver_type, AS_list, k=k)
    def _extract_extra_kwargs(self, k):
        self.k = k
    def _get_helper(self, ebdy, AS):
        return ModifiedHelmholtzHelper(ebdy, AS, k=self.k)
    def _grid_solve(self, fc):
        return np.fft.ifft2(np.fft.fft2(fc)*self.ihelm).real
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.helm = self.k**2 - self.lap
        self.ihelm = 1.0/self.helm
