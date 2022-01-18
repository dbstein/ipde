try:
    import finufft
    have_new_finufft = True
except:
    import finufftpy
    have_new_finufft = False
import numpy as np
# from ipde.heavisides import SlepianMollifier
from ipde.slepian.chebeval_bump_step import SlepianMollifier
import scipy as sp
import scipy.interpolate
from function_generator import FunctionGenerator
import numba

try:
    from mkl_fft import fft2, ifft2
except:
    from numpy.fft import fft2, ifft2

def excisor(r, r1, r2, MOL):
    """
    provides a smooth function s.t.
    excisor(r) = 1 for r < r1,
    excisor(r) = 0 for r > r2.
    """
    try:
        out = np.empty(r.shape, dtype=float)
        low = r<r1
        high = r>r2
        other = ~np.logical_or(low, high)
        out[r<r1] = 1.0
        out[r>r2] = 0.0
        rh = r[other]
        out[other] = MOL.step(1.0-(rh - r1)/(r2-r1)*2)
        return out
    except:
        if r < r1:
            return 1.0
        if r > r2:
            return 0.0
        else:
            return MOL.step(np.array(1.0-(r - r1)/(r2-r1)*2))
def gf_apply(gf, sx, sy, tx, ty, ch):
    dx = tx[:,None] - sx
    dy = ty[:,None] - sy
    d = np.hypot(dx, dy)
    return gf(d).dot(ch)
class ScalarGridBackend(object):
    def __init__(self, gf, fs, ifs, h, spread_width, kernel_kwargs=None, funcgen_tol=1e-10, inline_core=True):
        """
        Backend class for re-usable 'ewald' sum grid evaluation
            for reusability, the grid must have the same h
            and the ewald sum must use the same spread width
        gf: numba callable greens function, gf(r)
        fs: Fourier symbol for operator, fs(kx, ky)
        ifs: Inverse Fourier symbol for operator, ifs(kx, ky)
        h: grid spacing
        spread_width: width to do spreading on
            for Laplace, 15 gives ~7 digits
                         20 gives ~10 digits
                can't seem to do much better than that, right now
        kernel_kwargs: dict of arguments to be passed to gf, fs, ifs, tsgf functions
        funcgen_tol: tolerance for function generator representation of
            functions used in interior spread funciton.  can't seem to beat
            ~10 digits overall now, so no real reason to do more than that
        inline_core: whether to inline the function generator functions into
            the compiled ewald functions
            (inlining may speed things up but slows compilation time,
            sometimes dramatically)
        """
        self.kernel_kwargs = {} if kernel_kwargs is None else kernel_kwargs
        self.gf = lambda r: gf(r, **self.kernel_kwargs)
        self.fourier_symbol = lambda kx, ky: fs(kx, ky, **self.kernel_kwargs)
        self.inverse_fourier_symbol = lambda kx, ky: ifs(kx, ky, **self.kernel_kwargs)
        self.h = h
        self.spread_width = spread_width
        self.funcgen_tol = funcgen_tol
        self.inline_core = inline_core
        # construct mollifier
        self.mollifier = SlepianMollifier(2*self.spread_width)
        self.ssw = self.spread_width*self.h
        # screened greens function
        _excisor_gf = lambda d: excisor(d, 0.0, self.ssw, self.mollifier)*self.gf(d)
        try:
            self.ex_funcgen = FunctionGenerator(_excisor_gf, 0.0, self.ssw, 
                                tol=self.funcgen_tol, inline_core=self.inline_core)
            self.excisor_gf = self.ex_funcgen.get_base_function(check=False)
        except:
            raise Exception('Failed constructing FunctionGenerator function for mollifier')
        # construct differential operator applied to residual of screened greens function
        _sn = 4*self.spread_width
        _sgv = np.linspace(0, 4*self.ssw, _sn, endpoint=False)
        _sgx, _sgy = np.meshgrid(_sgv, _sgv, indexing='ij')
        _skv = np.fft.fftfreq(_sn, self.h/(2*np.pi))
        _skx, _sky = np.meshgrid(_skv, _skv, indexing='ij')
        _slap = self.fourier_symbol(_skx, _sky)
        pt = np.array([[2*self.ssw], [2*self.ssw]])
        targ = np.row_stack([_sgx.ravel(), _sgy.ravel()])
        u = gf_apply(self.gf, pt[0], pt[1], targ[0], targ[1], np.array([1.0,])).reshape(_sn, _sn)
        u[_sn//2, _sn//2] = 0.0
        dist = np.hypot(_sgx-2*self.ssw, _sgy-2*self.ssw)
        dec1 = excisor(dist, 0.0, self.ssw, self.mollifier)
        dec2 = excisor(dist, self.ssw, 2*self.ssw, self.mollifier)
        uf = u*(1-dec1)*dec2
        self.do_ufd = ifft2(fft2(uf)*_slap).real
        # get an interpolater for this
        _ax = np.linspace(np.pi, 1.5*np.pi, 1000)
        _ay = np.repeat(np.pi, _ax.size)
        _ar = np.linspace(0, self.ssw, _ax.size)        
        _fh = fft2(self.do_ufd)/(_sn*_sn)
        out = finufft.nufft2d2(_ax, _ay, _fh, isign=1, eps=1e-15, modeord=1)
        self._do_ufd_interpolater = sp.interpolate.InterpolatedUnivariateSpline(
                                _ar, out.real, k=5, bbox=[0, self.ssw], ext=1)
        try:
            self.do_ufd_funcgen = FunctionGenerator(self._do_ufd_interpolater, 
                0.0, self.ssw, tol=self.funcgen_tol, inline_core=self.inline_core)
            self.do_ufd_interpolater = self.do_ufd_funcgen.get_base_function(check=False)
        except:
            raise Exception('Failed constructing FunctionGenerator function for laplacian of greens function times mollifier')
    def initialize_periodic(self):
        """
        Define periodic local evaluator function
        """
        _ex_gf = self.excisor_gf
        _do_ufd = self.do_ufd_interpolater
        h = self.h
        sw = self.spread_width
        @numba.njit(parallel=True, fastmath=True)
        def ewald_local_periodic(source, charge, xv, yv):
            xmin = xv[0]
            ymin = yv[0]
            shape = (charge.size, 2*sw+2, 2*sw+2)
            fwork1 = np.empty(shape, dtype=numba.float64)
            fwork2 = np.empty(shape, dtype=numba.float64)
            iwork1 = np.empty(shape, dtype=numba.int64)
            iwork2 = np.empty(shape, dtype=numba.int64)
            bwork1 = np.zeros(shape, dtype=numba.boolean)
            sh = (xv.size, yv.size)
            op = np.zeros(sh, dtype=numba.float64)
            u = np.zeros_like(op)
            N = source.shape[1]
            nx = xv.size
            ny = yv.size
            md = sw*h
            for i in numba.prange(N):
                sx = source[0,i]
                sy = source[1,i]
                ch = charge[i]
                indx = int((sx - xmin) // h)
                indy = int((sy - ymin) // h)
                lxi = indx-sw-1
                lyi = indy-sw-1
                hxi = indx+sw+1
                hyi = indy+sw+1
                for ixind, ix in enumerate(range(lxi, hxi)):
                    ixm = ix % nx
                    xvh = xmin + ix*h
                    for iyind, iy in enumerate(range(lyi, hyi)):
                        iym = iy % ny
                        yvh = ymin + iy*h
                        d = np.hypot(xvh-sx, yvh-sy)
                        if d <= md:
                            fwork1[i,ixind,iyind] = _ex_gf(d)*ch
                            fwork2[i,ixind,iyind] = _do_ufd(d)*ch
                            iwork1[i,ixind,iyind] = ixm
                            iwork2[i,ixind,iyind] = iym
                            bwork1[i,ixind,iyind] = True
            for i in range(N):
                for ixind in range(2*sw+2):
                    for iyind in range(2*sw+2):
                        if bwork1[i,ixind,iyind]:
                            ixm = iwork1[i,ixind,iyind]
                            iym = iwork2[i,ixind,iyind]
                            u[ixm,  iym] += fwork1[i, ixind, iyind]
                            op[ixm, iym] += fwork2[i, ixind, iyind]
            return op, u
        self.ewald_local_periodic = ewald_local_periodic
    def initialize_freespace(self):
        """
        Define periodic local evaluator function
        """
        _ex_gf = self.excisor_gf
        _do_ufd = self.do_ufd_interpolater
        h = self.h
        sw = self.spread_width
        @numba.njit(parallel=True)
        def ewald_local_freespace(source, charge, xv, yv, op, u, op_na):
            xmin = xv[0]
            ymin = yv[0]
            shape = (charge.size, 2*sw+2, 2*sw+2)
            fwork1 = np.empty(shape, dtype=numba.float64)
            fwork2 = np.empty(shape, dtype=numba.float64)
            iwork1 = np.empty(shape, dtype=numba.int64)
            iwork2 = np.empty(shape, dtype=numba.int64)
            bwork1 = np.zeros(shape, dtype=numba.boolean)
            N = source.shape[1]
            nx = xv.size
            ny = yv.size
            md = sw*h
            for i in numba.prange(N):
                sx = source[0,i]
                sy = source[1,i]
                ch = charge[i]
                indx = int((sx - xmin) // h)
                indy = int((sy - ymin) // h)
                lxi = indx-sw-1
                lyi = indy-sw-1
                hxi = indx+sw+1
                hyi = indy+sw+1
                for ixind, ix in enumerate(range(lxi, hxi)):
                    xvh = xmin + ix*h
                    for iyind, iy in enumerate(range(lyi, hyi)):
                        yvh = ymin + iy*h
                        d = np.hypot(xvh-sx, yvh-sy)
                        if d <= md:
                            fwork1[i,ixind,iyind] = _ex_gf(d)*ch
                            fwork2[i,ixind,iyind] = _do_ufd(d)*ch
                            iwork1[i,ixind,iyind] = ix + op_na
                            iwork2[i,ixind,iyind] = iy + op_na
                            bwork1[i,ixind,iyind] = True
            for i in range(N):
                for ixind in range(2*sw+2):
                    for iyind in range(2*sw+2):
                        if bwork1[i,ixind,iyind]:
                            ix = iwork1[i,ixind,iyind]
                            iy = iwork2[i,ixind,iyind]
                            u [ix, iy] += fwork1[i, ixind, iyind]
                            op[ix, iy] += fwork2[i, ixind, iyind]
        self.ewald_local_freespace = ewald_local_freespace
    def check_periodic(self, xv, yv):
        self.check_either(xv, yv)
    def check_freespace(self, xv, yv):
        self.check_either(xv, yv)
        if xv.size != yv.size:
            raise Exception('Square grid required for freespace evaluator')
    def check_either(self, xv, yv):
        xh = xv[1] - xv[0]
        if np.abs(xh-self.h) > 1e-15:
            raise Exception('h of input xv vector not same as backend')
        yh = yv[1] - yv[0]
        if np.abs(yh-self.h) > 1e-15:
            raise Exception('h of input yv vector not same as backend')

class ScalarPeriodicGridEvaluator(object):
    def __init__(self, backend, xv, yv):
        self.backend = backend
        self.xv = xv
        self.yv = yv
        self.inverse_fourier_symbol = self.backend.inverse_fourier_symbol
        self.backend.check_periodic(self.xv, self.yv)
        self.h = self.backend.h
        self.nx = self.xv.size
        self.ny = self.yv.size
        self.kxv = np.fft.fftfreq(self.nx, self.h/(2*np.pi))
        self.kyv = np.fft.fftfreq(self.ny, self.h/(2*np.pi))
        self.kx, self.ky = np.meshgrid(self.kxv, self.kyv, indexing='ij')
        self.inverse_symbol = self.inverse_fourier_symbol(self.kx, self.ky)
        self.local_func = self.backend.ewald_local_periodic
    def __call__(self, src, ch):
        op, u = self.local_func(src, ch, self.xv, self.yv)
        u += ifft2(fft2(op)*self.inverse_symbol).real
        return u

class ScalarFreespaceGridEvaluator(object):
    def __init__(self, backend, xv, yv, tsgf_or_TH):
        self.backend = backend
        self.xv = xv
        self.yv = yv
        self.backend.check_freespace(self.xv, self.yv)
        self.backend.initialize_freespace()
        self.h = self.backend.h
        self.n = self.xv.size
        self.spread_width = self.backend.spread_width
        # get the T operator
        self.expand_n = self.n + 2*self.spread_width
        self.n_extract = self.expand_n//2 + self.spread_width
        self.big_n = 2*self.expand_n
        self.Big_n = 4*self.expand_n
        if type(tsgf_or_TH) == np.ndarray:
            self.TH = tsgf_or_TH
        else:
            self.truncated_spectral_gf = lambda k, L: tsgf_or_TH(k, L, **self.backend.kernel_kwargs)
            self.drange = self.xv[-1] - self.xv[0] + self.h
            self.kv = np.fft.fftfreq(self.Big_n, self.h/(2*np.pi))
            self.kx, self.ky = np.meshgrid(self.kv, self.kv, indexing='ij')
            self.tsgf = self.truncated_spectral_gf(np.hypot(self.kx, self.ky), 2.5*self.drange)
            w1 = np.zeros([self.Big_n, self.Big_n], dtype=float)
            w1[self.big_n, self.big_n] = 1.0
            w2 = ifft2(fft2(w1)*self.tsgf)
            self.T = w2[self.expand_n:-self.expand_n,self.expand_n:-self.expand_n]
            self.TH = fft2(self.T)
        # storage for padded array
        self.big_op = np.zeros([self.big_n, self.big_n], dtype=float)
        self.big_u  = np.zeros([self.big_n, self.big_n], dtype=float)
        # extract local ewald function from backend
        self.local_func = self.backend.ewald_local_freespace
    def __call__(self, src, ch):
        self.big_op[:] = 0.0
        self.big_u[:] = 0.0
        self.local_func(src, ch, self.xv, self.yv, self.big_op, self.big_u, self.n_extract)
        big_adj = np.fft.fftshift(ifft2(fft2(self.big_op)*self.TH).real)
        small_u = self.big_u[self.n_extract:-self.n_extract,self.n_extract:-self.n_extract].copy()
        small_adj = big_adj[self.n_extract:-self.n_extract,self.n_extract:-self.n_extract].copy()
        np.subtract(small_u, small_adj, small_u)
        return small_u


