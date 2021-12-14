import numpy as np
import numba

def odd_even_converter(c):
    """
    Converts odd Chebyhsev series for f(x) to even Chebyshev series for f(x)/x
    """
    cout = np.zeros(c.size+1)
    cout[-1] = 0.0
    for i in range(c.size-1,-2,-2):
        cout[i-1] = 2*c[i]-cout[i+1]
    cout[0] *= 0.5
    return cout[:-1:2].copy() # copy to get dense array

@numba.njit(fastmath=True)
def numba_chebeval1(x, c):
    x2 = 2*x
    c0 = c[-2]
    c1 = c[-1]
    for i in range(3, len(c) + 1):
        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1*x2
    return c0 + c1*x
@numba.njit(fastmath=True)
def numba_chebeval_even1(x, c):
    return numba_chebeval1(2.0*x*x-1.0, c)
@numba.njit(fastmath=True)
def numba_chebeval_odd1(x, c):
    return x*numba_chebeval_even1(x, c)
@numba.njit(fastmath=True)
def numba_step_eval1(x, c):
    return 0.5 + numba_chebeval_odd1(x, c)

@numba.njit(fastmath=True, parallel=True)
def _wrapper(x, c, function):
    xo = x.ravel()
    out = np.zeros_like(xo)
    for j in numba.prange(xo.size):
        out[j] = function(xo[j], c)
    return out.reshape(x.shape)

@numba.njit
def numba_chebeval(x, c):
    """
    Evaluate a general real Chebyshev series c at the points in the array x(*)
    """
    return _wrapper(x, c, numba_chebeval1)

@numba.njit
def numba_chebeval_even(x, c):
    """
    Evaluate an even Chebyshev series c at the points in the array x(*)
    i.e. c should only have [c0, c2, c4, c6, ...]
    """
    return _wrapper(x, c, numba_chebeval_even1)

@numba.njit
def numba_chebeval_odd(x, c):
    """
    Evaluate an 'odd' Chebyshev series c at the points in the array x(*)
    here c should only have [c0, c2, c4, c6, ...]
    for the even Chebyshev series for f(x)/x, which may be generated
    using the function odd_even_converter
    """
    return _wrapper(x, c, numba_chebeval_odd1)

@numba.njit
def numba_step_eval(x, c):
    """
    Evaluate a step function that goes from 0 --> 1 over -1 --> 1
    By evaluating the odd series for step(x) - 0.5
    c here should contain only the even coefficients for the Chebysehv series
    for f(x) = (step(x)-0.5)/x
    see documentation for numba_chebeval_odd
    """
    return _wrapper(x, c, numba_step_eval1)

