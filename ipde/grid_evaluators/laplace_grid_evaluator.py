import numpy as np
import numba
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarGridBackend
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarFreespaceGridEvaluator
from ipde.grid_evaluators.scalar_grid_evaluator import ScalarPeriodicGridEvaluator
from scipy.special import j0, j1

scale = -1.0/(2*np.pi)

@numba.njit
def gf(r):
    return scale*np.log(r)
def fs(kx, ky):
    return -(kx*kx + ky*ky)
def ifs(kx, ky):
    lap = fs(kx, ky)
    lap[0,0] = 1.0
    ilap = 1.0/lap
    ilap[0,0] = 0.0
    return ilap
def trunc_sgf(k, L):
    if type(k) == np.ndarray:
        out = np.zeros(k.shape, dtype=float)
        sel = k == 0
        nsel = ~sel
        knsel = k[nsel]
        out[sel] = -L**2*np.log(L) + L**2*(1+2*np.log(L))/4
        out[nsel] = (1.0 - j0(L*knsel))/knsel**2 - L*np.log(L)*j1(L*knsel)/knsel
        return out
    elif k == 0:
        return -L**2*np.log(L) + L**2*(1+2*np.log(L))/4
    else:
        return (1.0 - j0(L*k))/k**2 - L*np.log(L)*j1(L*k)/k

class LaplaceGridBackend(ScalarGridBackend):
    def __init__(self, h, spread_width, funcgen_tol=1e-10, inline_core=True):
        super().__init__(gf, fs, ifs, h, spread_width, {}, funcgen_tol, inline_core)

class LaplaceFreespaceGridEvaluator(ScalarFreespaceGridEvaluator):
    def __init__(self, backend, xv, yv):
        super().__init__(backend, xv, yv, trunc_sgf)

class LaplacePeriodicGridEvaluator(ScalarPeriodicGridEvaluator):
    def __init__(self, backend, xv, yv):
        super().__init__(backend, xv, yv)
