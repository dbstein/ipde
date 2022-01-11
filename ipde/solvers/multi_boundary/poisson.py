import numpy as np
from ipde.solvers.multi_boundary.scalar import ScalarSolver
from ipde.solvers.internals.poisson import PoissonHelper
from ipde.utilities import affine_transformation
from ipde.grid_evaluators.laplace_grid_evaluator import LaplaceFreespaceGridEvaluator, LaplaceGridBackend
from ipde.utilities import fft2, ifft2

try:
    from flexmm.flexmm2d.kifmm import KI_FMM
    import numba
    @numba.njit(fastmath=True)
    def Laplace_Eval(sx, sy, tx, ty):
        dx = tx-sx
        dy = ty-sy
        d2 = dx*dx + dy*dy
        scale = -1.0/(4*np.pi)
        return scale*np.log(d2)
    flexmm_okay = True
except:
    flexmm_okay = False

class PoissonSolver(ScalarSolver):
    def __init__(self, ebdyc, solver_type='spectral', AS_list=None, grid_backend=None):
        # check that ebdyc has bumps defined
        # if not ebdyc.bumpy_readied:
            # raise Exception('Laplace solver requires embedded boundary collection with a bump function.')
        super().__init__(ebdyc, solver_type, AS_list, grid_backend)
    def _get_helper(self, ebdy, helper):
        return PoissonHelper(ebdy, helper)
    def _grid_solve(self, fc):
        fc = self.ebdyc.demean_function(fc)
        self._fc_save = fc.copy() # for plotting only!
        uch = fft2(fc)*self.ilap
        return uch, ifft2(uch).real
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.lap[0,0] = np.Inf
        self.ilap = 1.0/self.lap
        self.lap[0,0] = 0.0
    def _define_grid_evaluator(self):
        if type(self.grid_backend) == LaplaceGridBackend:
            self.ewald_evaluator = LaplaceFreespaceGridEvaluator(
                            self.grid_backend, self.grid.xv, self.grid.yv)
            def evaluator(ch):
                return self.ewald_evaluator(
                            self.grid_sources.get_stacked_boundary(),
                            ch*self.grid_sources.weights)
            self.Grid_Evaluator = evaluator
            self.split_grid_evaluation = True
        elif self.grid_backend == 'flexmm':
            self.FMM = KI_FMM(self.grid_sources.x, self.grid_sources.y, Laplace_Eval, 20, 64)
            def evaluator(ch):
                self.FMM.build_expansions(ch*self.grid_sources.weights)
                targ = self.ebdyc.grid_pnai
                return self.FMM.target_evaluation(targ.x, targ.y)
            self.Grid_Evaluator = evaluator
            self.split_grid_evaluation = False
        elif self.grid_backend == 'pybie2d':
            self.split_grid_evaluation = False
            def evaluator(ch):
                return self.Layer_Apply(self.grid_sources, self.ebdyc.grid_pnai, ch)
            self.Grid_Evaluator = evaluator
            self.split_grid_evaluation = False
        else:
            raise Exception('grid_backend not recognized')

