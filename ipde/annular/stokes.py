import numpy as np
from ..utilities import fast_dot, concat, fast_LU_solve, mfft, mifft, fourier_multiply
import scipy as sp
import scipy.linalg
from personal_utilities.scipy_gmres import right_gmres

def scalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    ks = AAG.ks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    uh_t = R01.dot(uh*ks*1j)
    uh_tt = R12.dot(fourier_multiply(uh_t, ipsi1)*ks*1j)
    uh_rr = D12.dot(fourier_multiply(D01.dot(uh), psi1))
    luh = fourier_multiply(uh_rr+uh_tt, ipsi2)
    return luh

class AnnularStokesSolver(object):
    """
    Spectrally accurate Stokes solver on annular domain

    Solves -mu Lu + grad p = f in the annulus described by the Annular Geometry AG
                     div u = 0
    Subject to the boundary condition:
    u(ri) = ig (boundary condition at the inner radius)
    u(ro) = og (boundary condition at the outer radius)

    Note that this solver solves the problem in (r, t) coordinates
    And not in (u, v) coordinates

    The forces, and boundary conditions, must be given in this manner
    """
    def __init__(self, AAG, mu):
        self.AAG = AAG
        self.mu = mu
        M =  AAG.M
        ns = AAG.ns
        n = AAG.n
        self.M = M
        self.ns = ns
        self.n = n
        self.NU = self.M*self.ns
        self.NP = (self.M-1)*self.ns
        self.NB = 2*self.NU + self.NP
        self.u_small_shape = (self.M, self.ns)
        self.u_shape = (self.M, self.n)
        self.p_small_shape = (self.M-1, self.ns)
        self.p_shape = (self.M-1, self.n)
        self._construct()
        self.APPLY = scipy.sparse.linalg.LinearOperator((self.NB, self.NB), dtype=complex, matvec=self._apply)
        self.PREC = scipy.sparse.linalg.LinearOperator((self.NB, self.NB), dtype=complex, matvec=self._preconditioner)
    def _construct(self):
        AAG = self.AAG
        CO = AAG.CO
        apsi0 =  AAG.approx_psi0
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
        obcd =   CO.obc_dirichlet
        VI1 =    CO.VI1
        ns =     self.ns
        M =      self.M
        self._KLUS = []
        for i in range(ns):
            K = np.zeros((3*M-1, 3*M-1), dtype=complex)
            LL = fast_dot(aipsi2, fast_dot(D12, fast_dot(apsi1, D01))) - \
                fast_dot(np.ones(M-2)*ks[i]**2, fast_dot(R12, fast_dot(aipsi1, R01)))
            # ur things
            K[ 0*M + 0   : 0*M + M-2  , 0*M : 1*M ] = -LL + fast_dot(aipsi2**2, R02)
            K[ 0*M + 0   : 0*M + M-2  , 1*M : 2*M ] = fast_dot(2*aipsi2**2, fast_dot(R02, 1j*ks[i]*np.ones(M)) )
            K[ 0*M + 0   : 0*M + M-2  , 0*M : 2*M ] *= self.mu
            K[ 0*M + 0   : 0*M + M-2  , 2*M :     ] = D12
            K[ 0*M + M-2 : 0*M + M-1  , 0*M : 1*M ] = ibcd
            K[ 0*M + M-1 : 0*M + M    , 0*M : 1*M ] = obcd
            # ut things
            K[ 1*M + 0   : 1*M + M-2  , 0*M : 1*M ] = -fast_dot(2*aipsi2**2, fast_dot(R02, 1j*ks[i]*np.ones(M)) )
            K[ 1*M + 0   : 1*M + M-2  , 1*M : 2*M ] = -LL + fast_dot(aipsi2**2, R02)
            K[ 1*M + 0   : 1*M + M-2  , 0*M : 2*M ] *= self.mu
            K[ 1*M + 0   : 1*M + M-2  , 2*M :     ] = fast_dot(aipsi2, fast_dot(R12, 1j*ks[i]*np.ones(M-1)))
            K[ 1*M + M-2 : 1*M + M-1  , 1*M : 2*M ] = ibcd
            K[ 1*M + M-1 : 1*M + M    , 1*M : 2*M ] = obcd
            # div u things
            K[ 2*M + 0   : 2*M + M-1  , 0*M : 1*M ] = fast_dot(aipsi1, fast_dot(D01, apsi0))
            K[ 2*M + 0   : 2*M + M-1  , 1*M : 2*M ] = fast_dot(aipsi1, fast_dot(R01, 1j*ks[i]*np.ones(M)))
            # fix the pressure nullspace
            if i == 0:
                K[2*M:, 2*M:] += VI1[0]
            self._KLUS.append(sp.linalg.lu_factor(K))
    def _preconditioner(self, ffh):
        M = self.M
        frh, fth, fph = self._extract_stokes(ffh, withcopy=True)
        for i in range(self.ns):
            vec = concat(frh[:,i], fth[:,i], fph[:,i])
            vec = fast_LU_solve(self._KLUS[i], vec)
            frh[:,i] = vec[0*M:1*M]
            fth[:,i] = vec[1*M:2*M]
            fph[:,i] = vec[2*M:]
        return concat(frh, fth, fph)
    def _extract_stokes(self, fh, withcopy=False):
        frh = fh[0*self.NU:1*self.NU         ].reshape(self.u_small_shape)
        fth = fh[1*self.NU:2*self.NU         ].reshape(self.u_small_shape)
        fph = fh[2*self.NU:2*self.NU+self.NP ].reshape(self.p_small_shape)
        if withcopy:
            frh = frh.copy()
            fth = fth.copy()
            fph = fph.copy()
        return frh, fth, fph
    def _apply(self, uuh):
        AAG = self.AAG
        RAG = self.RAG
        CO = self.AAG.CO
        ibcd = CO.ibc_dirichlet
        obcd = CO.obc_dirichlet
        D01  = CO.D01
        D12  = CO.D12
        R01  = CO.R01
        R12  = CO.R12
        R02  = CO.R02
        VI1  = CO.VI1
        ks = AAG.ks
        psi0 = RAG.psi0
        psi1 = RAG.psi1
        ipsi1 = RAG.inv_psi1
        ipsi2 = RAG.inv_psi2
        DR_psi2 = RAG.DR_psi2
        ipsi_DR_ipsi_DT_psi2 = RAG.ipsi_DR_ipsi_DT_psi2
        ipsi_DT_ipsi_DR_psi2 = RAG.ipsi_DT_ipsi_DR_psi2
        # a lot of room for optimization in this function!
        urh, uth, ph = self._extract_stokes(uuh)
        # compute scalar laplacian
        lap_urh = scalar_laplacian(CO, AAG, RAG, urh)
        lap_uth = scalar_laplacian(CO, AAG, RAG, uth)
        # ur equation
        t1 = fourier_multiply(R02.dot(uth*1j*ks), 2*DR_psi2*ipsi2**2)
        t2 = fourier_multiply(R02.dot(urh), DR_psi2**2*ipsi2**2)
        t3 = fourier_multiply(R02.dot(uth), ipsi_DR_ipsi_DT_psi2)
        t4 = D12.dot(ph)
        frh = self.mu*(-lap_urh + t1 + t2 + t3) + t4
        # ut equation
        t1 = fourier_multiply(R02.dot(urh*1j*ks), 2*DR_psi2*ipsi2**2)
        t2 = fourier_multiply(R02.dot(uth), DR_psi2**2*ipsi2**2)
        t3 = fourier_multiply(R02.dot(urh), ipsi_DT_ipsi_DR_psi2)
        t4 = fourier_multiply(R12.dot(ph*1j*ks), ipsi2)
        fth = self.mu*(-lap_uth - t1 + t2 - t3) + t4
        # div u equation
        fph = fourier_multiply(D01.dot(fourier_multiply(urh, psi0)) + R01.dot(uth*1j*ks), ipsi1)
        # add BCS
        frh_full = concat( frh, ibcd.dot(urh), obcd.dot(urh) )
        fth_full = concat( fth, ibcd.dot(uth), obcd.dot(uth) )
        # fph output
        fph[:,0] += VI1.dot(ph)[0,0] # this is the mean of the pressure!
        # get mean of the pressure
        return concat( frh_full, fth_full, fph )
    def solve(self, RAG, fr, ft, irg, itg, org, otg, verbose=False, **kwargs):
        self.RAG = RAG
        R02 = self.AAG.CO.R02
        P10 = self.AAG.CO.P10
        ffr = concat(R02.dot(fr), irg, org)
        fft = concat(R02.dot(ft), itg, otg)
        ffrh = mfft(ffr.reshape(self.u_shape))
        ffth = mfft(fft.reshape(self.u_shape))
        ffph = np.zeros(self.NP, dtype=complex)
        ffh = concat(ffrh, ffth, ffph)
        out = right_gmres(self.APPLY, ffh, M=self.PREC, verbose=verbose, **kwargs)
        res = out[0]
        if verbose:
            print('GMRES took:', len(out[2]), 'iterations.')
        urh, uth, ph = self._extract_stokes(res)
        ur = mifft(urh).real
        ut = mifft(uth).real
        p = P10.dot(mifft(ph).real)
        return ur, ut, p
