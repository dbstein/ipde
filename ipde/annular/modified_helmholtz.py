import numpy as np
from ipde.utilities import fast_dot, concat, fast_LU_solve, mfft, mifft, fourier_multiply, fft, ifft, ffourier_multiply
import scipy as sp
import scipy.linalg
from personal_utilities.scipy_gmres import right_gmres, gmres
import numexpr as ne
from ipde.sparse_matvec import zSpMV_viaMKL
import numba

def scalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    iks = AAG.iks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    uh_t = R01.dot(uh*iks)
    uh_tt = R12.dot(fourier_multiply(uh_t, ipsi1)*iks)
    uh_rr = D12.dot(fourier_multiply(D01.dot(uh), psi1))
    luh = fourier_multiply(uh_rr+uh_tt, ipsi2)
    return luh

def fscalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    iks = AAG.iks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    uh_t = R01.dot(uh*iks)
    uh_tt = R12.dot(ffourier_multiply(uh_t, ipsi1)*iks)
    uh_rr = D12.dot(ffourier_multiply(D01.dot(uh), psi1))
    luh = ffourier_multiply(uh_rr+uh_tt, ipsi2)
    return luh


# custom numba function for preconditioner
# basically a batched matvec; but note the ordering for the input vector
# this conforms with how that vector is stored in the rest of the solve
@numba.njit(parallel=True, fastmath=True)
def batch_matvecT_par(A, x):
    sh = (A.shape[0], A.shape[1])
    out = np.zeros(sh, dtype=np.complex128)
    for i in numba.prange(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                out[i, j] += A[i, j, k] * x[k, i]
    return out
@numba.njit(parallel=False, fastmath=True)
def batch_matvecT_ser(A, x):
    sh = (A.shape[0], A.shape[1])
    out = np.zeros(sh, dtype=np.complex128)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                out[i, j] += A[i, j, k] * x[k, i]
    return out
def batch_matvecT(A, x):
    if A.shape[0]*A.shape[1] > 10000:
        return batch_matvecT_par(A, x)
    else:
        return batch_matvecT_ser(A, x)

@numba.njit(parallel=True, fastmath=True)
def optim_batch_matvecT_par(A, x, out):
    for i in numba.prange(A.shape[0]):
        for j in range(A.shape[1]):
            kaccum = 0.0
            for k in range(A.shape[2]):
                kaccum += A[i, j, k] * x[i+k*A.shape[0]]
            out[i+j*A.shape[0]] = kaccum
@numba.njit(parallel=False, fastmath=True)
def optim_batch_matvecT_ser(A, x, out):
    for i in numba.prange(A.shape[0]):
        for j in range(A.shape[1]):
            kaccum = 0.0
            for k in range(A.shape[2]):
                kaccum += A[i, j, k] * x[i+k*A.shape[0]]
            out[i+j*A.shape[0]] = kaccum
def optim_batch_matvecT(A, x, out):
    if A.shape[0]*A.shape[1] > 10000:
        optim_batch_matvecT_par(A, x, out)
    else:
        optim_batch_matvecT_ser(A, x, out)

class AnnularModifiedHelmholtzSolver(object):
    """
    Spectrally accurate Modified Helmholtz solver on annular domain

    Solves (k^2-L)u = f in the annulus described by the Annular Geometry AG
    Subject to the Robin boundary condition:
    ia*u(ri) + ib*u_r(ri) = ig (boundary condition at the inner radius)
    oa*u(ro) + ob*u_r(ro) = og (boundary condition at the outer radius)

    On instantionation, a preconditioner is formed with ia, ib, ua, ub
        defining the boundary conditions
    These can be changed at solvetime, but preconditioning may not work so well
    """
    def __init__(self, AAG, k, ia=1.0, ib=0.0, oa=1.0, ob=0.0):
        self.AAG = AAG
        self.ia = ia
        self.ib = ib
        self.oa = oa
        self.ob = ob
        self.k = k
        M =  AAG.M
        ns = AAG.ns
        n = AAG.n
        NB = M*ns
        self.M = M
        self.ns = ns
        self.n = n
        self.NB = NB
        self.small_shape = (self.M, self.ns)
        self.shape = (self.M, self.n)
        self._construct()
        self.APPLY = scipy.sparse.linalg.LinearOperator((self.NB, self.NB), dtype=complex, matvec=self._apply)
        self.PREC = scipy.sparse.linalg.LinearOperator((self.NB, self.NB), dtype=complex, matvec=self._optim_preconditioner)
    def _construct(self):
        AAG = self.AAG
        CO = AAG.CO
        apsi1 =  AAG.approx_psi1
        aipsi1 = AAG.approx_inv_psi1
        aipsi2 = AAG.approx_inv_psi2
        ks =     AAG.ks
        D01 =    CO.D01
        D12 =    CO.D12
        R01 =    CO.R01
        R12 =    CO.R12
        R02 =    CO.R02
        ibcd =   CO.ibc_dirichlet
        ibcn =   CO.ibc_neumann
        obcd =   CO.obc_dirichlet
        obcn =   CO.obc_neumann
        ns =     self.ns
        M =      self.M
        self._KLUS = []
        self._KINVS = []
        for i in range(ns):
            K = np.empty((M,M), dtype=complex)
            LL = fast_dot(aipsi2, fast_dot(D12, fast_dot(apsi1, D01))) - \
                fast_dot(np.ones(M-2)*ks[i]**2, fast_dot(R12, fast_dot(aipsi1, R01)))
            K[:M-2] = self.k**2*R02 - LL
            K[M-2:M-1] = self.ia*ibcd + self.ib*ibcn
            K[M-1:M-0] = self.oa*obcd + self.ob*obcn
            # self._KLUS.append(sp.linalg.lu_factor(K)) # for old preconditioner
            self._KINVS.append(sp.linalg.inv(K.real))
        self.KINV = sp.sparse.block_diag(self._KINVS, 'csr').astype('complex') # for SpMV preconditioner
        self.Stacked_KINVS = np.stack(self._KINVS).copy()
        self.prealloc = np.zeros(self.M*self.ns, dtype=complex)
    def _preconditioner(self, fh):
        return batch_matvecT(self.Stacked_KINVS, fh.reshape(self.small_shape)).ravel('F')
    def _optim_preconditioner(self, fh):
        optim_batch_matvecT(self.Stacked_KINVS, fh, self.prealloc)
        return self.prealloc
    def _SpMV_preconditioner(self, fh):
        # could avoid these reshapes if self.KINV constructed differently
        # however, they don't seem to take a huge amount of time
        w1 = fh.reshape(self.small_shape).ravel('F')
        w2 = zSpMV_viaMKL(self.KINV, w1)
        return w2.reshape(self.small_shape, order='F').ravel()
    def _old_preconditioner(self, fh):
        fh = fh.reshape(self.small_shape)
        fo = np.empty(self.small_shape, dtype=complex)
        for i in range(self.ns):
            fo[:,i] = fast_LU_solve(self._KLUS[i], fh[:,i])
        return fo.ravel()
    def _apply(self, uh):
        AAG = self.AAG
        RAG = self.RAG
        CO = self.AAG.CO
        ibcd = CO.ibc_dirichlet
        ibcn = CO.ibc_neumann
        obcd = CO.obc_dirichlet
        obcn = CO.obc_neumann
        R02  = CO.R02
        uh = uh.reshape(self.small_shape)
        luh = fscalar_laplacian(CO, AAG, RAG, uh)
        fuh = self.k**2*R02.dot(uh) - luh
        ibc = (self.ia*ibcd + self.ib*ibcn).dot(uh)
        obc = (self.oa*obcd + self.ob*obcn).dot(uh)
        return concat(fuh, ibc, obc)
    def solve(self, RAG, f, ig, og, ia=None, ib=None, oa=None, ob=None,
                                                    verbose=False, **kwargs):
        self.RAG = RAG
        self.ia = ia if ia is not None else self.ia
        self.ib = ib if ib is not None else self.ib
        self.oa = oa if oa is not None else self.oa
        self.ob = ob if ob is not None else self.ob
        R02 = self.AAG.CO.R02
        ff = concat(R02.dot(f), ig, og)
        # ffh = mfft(ff.reshape(self.shape)).ravel()
        ffh = fft(ff.reshape(self.shape)).ravel()
        out = right_gmres(self.APPLY, ffh, M=self.PREC, verbose=verbose, **kwargs)
        res = out[0]
        if verbose:
            print('GMRES took:', len(out[2]), 'iterations.')
        self.iterations_last_call = len(out[2])
        return ifft(res.reshape(self.small_shape)).real
