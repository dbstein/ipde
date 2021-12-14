import numpy as np
import numba
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarGridBackend
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarFreespaceGridEvaluator
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarPeriodicGridEvaluator
from scipy.special import j0, j1, k0, k1

def gf(r, helmholtz_k=None):
    return k0(helmholtz_k*r)/(2*np.pi)
def fs(kx, ky, helmholtz_k=None):
    return -helmholtz_k**2 - kx**2 - ky**2
def ifs(kx, ky, helmholtz_k=None):
    return 1.0/fs(kx, ky)
def trunc_sgf(k, L, helmholtz_k=None):
    kk0 = k0(L*helmholtz_k)
    kk1 = k1(L*helmholtz_k)
    return (1.0 + L*k*j1(L*k)*kk0 - L*helmholtz_k*j0(L*k)*kk1 ) / (k**2 + helmholtz_k**2)

class ModifiedHelmholtzGridBackend(ScalarGridBackend):
    def __init__(self, h, spread_width, helmholtz_k, funcgen_tol=1e-10, inline_core=True):
        kernel_kwargs = {'helmholtz_k' : helmholtz_k}
        super().__init__(gf, fs, ifs, h, spread_width, kernel_kwargs, funcgen_tol, inline_core)

class ModifiedHelmholtzFreespaceGridEvaluator(ScalarFreespaceGridEvaluator):
    def __init__(self, backend, xv, yv):
        super().__init__(backend, xv, yv, trunc_sgf)

class ModifiedHelmholtzPeriodicGridEvaluator(ScalarPeriodicGridEvaluator):
    def __init__(self, backend, xv, yv):
        super().__init__(backend, xv, yv)
