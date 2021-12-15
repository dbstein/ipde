import numpy as np
import numexpr as ne
from scipy.linalg.lapack import zgetrs

try:
    from mkl_fft import fft, ifft
    from mkl_fft import fft2, ifft2
except:
    fft = np.fft.fft
    ifft = np.fft.ifft
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2

def concat(*args):
    return np.concatenate([ np.array(arg).ravel() for arg in args ])

def affine_transformation(xin, min_in, max_in, min_out, max_out, 
                                    return_ratio=False, use_numexpr=False):
    ran_in = max_in - min_in
    ran_out = max_out - min_out
    rat = ran_out/ran_in
    xout = _affine_transformation(xin, rat, min_in, min_out, use_numexpr)
    if return_ratio:
        out = xout, rat
    else:
        out = xout
    return out

def _affine_transformation(xin, rat, min_in, min_out, use_numexpr):
    if use_numexpr:
        xout = ne.evaluate('(xin - min_in)*rat + min_out')
    else:
        xout = (xin - min_in)*rat + min_out
    return xout

def get_chebyshev_nodes(lb, ub, order):
    """
    Provides chebyshev quadratures nodes
    scaled to live on the interval [lb, ub], of specified order
    The nodes are reversed from traditional chebyshev nodes
        (so that the lowest valued node comes first)
    Returns:
        unscaled nodes
        scaled nodes
        scaling ratio
    """
    xc, _ = np.polynomial.chebyshev.chebgauss(order)
    x, rat = affine_transformation(xc[::-1], -1, 1, lb, ub, return_ratio=True)
    return xc[::-1], x, rat

def fast_dot(M1, M2):
    """
    Specialized interface to the numpy.dot function
    This assumes that A and B are both 2D arrays (in practice)
    When A or B are represented by 1D arrays, they are assumed to reprsent
        diagonal arrays
    This function then exploits that to provide faster multiplication
    """
    if len(M1.shape) in [1, 2] and len(M2.shape) == 1:
        return M1*M2
    elif len(M1.shape) == 1 and len(M2.shape) == 2:
        return M1[:,None]*M2
    elif len(M1.shape) == 2 and len(M2.shape) == 2:
        return M1.dot(M2)
    else:
        raise Exception('fast_dot requires shapes to be 1 or 2')

def fast_LU_solve(LU, b):
    """
    When running many small LU solves, the scipy call sp.linalg.lu_solve incurs
    significant overhead.  This calls the same LAPACK function, with no checks.

    Solves the system Ax=b for x, where LU = sp.linalg.lu_factor(A)
    (only for complex matrices and vectors...)
    """
    return zgetrs(LU[0], LU[1], b)[0]

def mfft(f):
    M = f.shape[0]
    N = f.shape[1]
    NS = N - 1
    N2 = int(N/2)
    fh = fft(f)
    temp = np.empty((M, NS), dtype=complex)
    temp[:,:N2] = fh[:,:N2]
    temp[:,N2:] = fh[:,N2+1:]
    return temp
def mifft(fh):
    M = fh.shape[0]
    NS = fh.shape[1]
    N = NS + 1
    N2 = int(N/2)
    temp = np.empty((M, N), dtype=complex)
    temp[:,:N2]   = fh[:,:N2]
    temp[:,N2]    = 0.0
    temp[:,N2+1:] = fh[:,N2:]
    return ifft(temp)
def fourier_multiply(fh, m):
    return mfft(m*mifft(fh))

def ffourier_multiply(fh, m):
    return fft(m*ifft(fh))

class SimpleFourierFilter(object):
    """
    Class to apply simple Fourier Filtration to a vector

    Filter types:
        'fraction' (requires kwarg: 'fraction' to be set)
        'rule 36'  (can set kwarg: 'power' but not necessary)
    """
    def __init__(self, modes, filter_type, **kwargs):
        self.n = modes.shape[0]
        self.modes = modes
        self.filter_type = filter_type
        self._get_filter(**kwargs)
    def __call__(self, fin, input_type='space', output_type='space'):
        input_is_real = fin.dtype == float and input_type == 'space'
        if input_type=='space':
            fin = np.fft.fft(fin)
        fout = fin*self.filter
        if output_type == 'space':
            fout = np.fft.ifft(fout)
            if input_is_real:
                fout = fout.real
        return fout
    def _get_filter(self, **kwargs):
        if self.filter_type == 'fraction':
            max_k = np.abs(self.modes).max()
            self.filter = np.ones(self.n, dtype=float)
            self.filter[np.abs(self.modes) > max_k*kwargs['fraction']] = 0.0
        elif self.filter_type == 'rule 36':
            max_k = np.abs(self.modes).max()
            if 'power' in kwargs:
                power36 = kwargs['power']
            else:
                power36 = 36
            self.filter = np.exp(-power36*(np.abs(self.modes)/max_k)**power36)
        else:
            raise Exception('Filter type not defined.')
