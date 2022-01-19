import numpy as np
from .scalar import ScalarSolver
from ..internals.modified_helmholtz import ModifiedHelmholtzHelper
from ipde.grid_evaluators.modified_helmholtz_grid_evaluator import ModifiedHelmholtzFreespaceGridEvaluator, ModifiedHelmholtzGridBackend

from ipde.utilities import fft2, ifft2

class ModifiedHelmholtzSolver(ScalarSolver):
    def __init__(self, ebdyc, k, solver_type='spectral', helpers=None, grid_backend='pybie2d'):
        self.k = k
        super().__init__(ebdyc, solver_type, helpers, grid_backend)
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
        uch = fft2(fc)*self.ihelm
        return uch, ifft2(uch).real
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.helm = self.k**2 - self.lap
        self.ihelm = 1.0/self.helm
    def _define_grid_evaluator(self):
        if type(self.grid_backend) in [ModifiedHelmholtzGridBackend, ModifiedHelmholtzFreespaceGridEvaluator]:
            if type(self.grid_backend) == ModifiedHelmholtzGridBackend:
                self.ewald_evaluator = ModifiedHelmholtzFreespaceGridEvaluator(
                                self.grid_backend, self.grid.xv, self.grid.yv)
            else:
                self.ewald_evaluator = self.grid_backend
            def evaluator(ch):
                return self.ewald_evaluator(
                            self.grid_sources.get_stacked_boundary(),
                            ch*self.grid_sources.weights)
            self.Grid_Evaluator = evaluator
            self.split_grid_evaluation = True
        elif self.grid_backend == 'pybie2d':
            self.split_grid_evaluation = False
            def evaluator(ch):
                return self.Layer_Apply(self.grid_sources, self.ebdyc.grid_pnai, ch)
            self.Grid_Evaluator = evaluator
            self.split_grid_evaluation = False
        else:
            raise Exception('grid_backend not recognized')
