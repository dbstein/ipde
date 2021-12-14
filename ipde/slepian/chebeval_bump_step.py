import numpy as np
from ipde.slepian.chebeval import numba_chebeval_even, numba_step_eval
from ipde.slepian.chebeval import numba_chebeval_even1, numba_step_eval1
from ipde.slepian.heaviside_coefficients import bump_even_list, step_even_list

def _wrap(x, func1, func, coef, check_bounds):
    if type(x) == float:
        if check_bounds and (x < -1 or x > 1):
            return 0.0
        else:
            return func1(x, coef)
    elif type(x) == np.ndarray and x.dtype == float:
        out = np.zeros_like(x)
        if check_bounds:
            good = np.logical_and(x >= -1.0, x <= 1.0)
            out[good] = func(x[good], coef)
        else:
            out = func(x, coef)
        return out        
    else:
        raise Exception('Type of x must be float or np.ndarray(float)')

class SlepianMollifier:
    """
    Constructs mollification 'step' and 'bump' functions
    from a prolate spheroidal wavefunction

    This version uses precomputed Chebyshev expansions to avoid
    construction costs.  these are stored in ipde.slepian.heaviside_coefficients
    """
    def __init__(self, r):
        self.r = r
        self.bump_c = bump_even_list[self.r]
        self.step_c = step_even_list[self.r]
    def bump(self, x, check_bounds=True):
        return _wrap(x, numba_chebeval_even1, numba_chebeval_even, self.bump_c, check_bounds)
    def step(self, x, check_bounds=True):
        out = _wrap(x, numba_step_eval1, numba_step_eval, self.step_c, check_bounds)
        if check_bounds:
            if type(x) == float:
                if x > 1.0: out = 1.0
            else:
                out[x > 1.0] = 1.0
        return out
