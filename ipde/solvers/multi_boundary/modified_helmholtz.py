import numpy as np
from .scalar import ScalarSolver
from ..internals.modified_helmholtz import ModifiedHelmholtzHelper

class ModifiedHelmholtzSolver(ScalarSolver):
    def __init__(self, ebdyc, solver_type='spectral', helpers=None, k=1.0):
        super().__init__(ebdyc, solver_type, helpers, k=k)
    def _extract_extra_kwargs(self, k):
        self.k = k
    def _get_helper_combatibility(self, ebdy, helper):
        """
        Returns 0, 1, or 2
        0:  no compatiblity --- helper is worthless, start over
        1:  some compatibility --- annular solver is okay, use that
        2:  full compatibility --- start from scratch
        """
        if helper == None:
            return 0
        if self.k != helper.k:
            return 0
        if ebdy.bdy.N != helper.ebdy.bdy.N:
            return 0
        if helper.ebdy is not ebdy:
            return 1
        return 2
    def _get_helper(self, ebdy, helper):
        c = self._get_helper_combatibility(ebdy, helper)
        if c == 0:
            return ModifiedHelmholtzHelper(ebdy, k=self.k)
        elif c == 1:
            return ModifiedHelmholtzHelper(ebdy, helper.annular_solver, k=self.k)
        elif c == 2:
            return helper
        raise Exception('Helper compatibility returned unimplemented value.')
    def _grid_solve(self, fc):
        return np.fft.ifft2(np.fft.fft2(fc)*self.ihelm).real
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.helm = self.k**2 - self.lap
        self.ihelm = 1.0/self.helm
