import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg
from ..utilities import concat, get_chebyshev_nodes

class ChebyshevOperators(object):
    def __init__(self, M, rat):
        """
        Construct Chebyshev operators to be used in annular solvers

        Inputs:
            M (int),     number of modes in chebyshev grid
            rat (float), ratio giving width of annulus to [-1,1]
        """
        self.M = M
        xc0, _ = np.polynomial.chebyshev.chebgauss(M-0)
        xc1, _ = np.polynomial.chebyshev.chebgauss(M-1)
        xc2, _ = np.polynomial.chebyshev.chebgauss(M-2)
        # vandermonde and inverse vandermonde matrices
        self.V0 = np.polynomial.chebyshev.chebvander(xc0, M-1)
        self.V1 = np.polynomial.chebyshev.chebvander(xc1, M-2)
        self.V2 = np.polynomial.chebyshev.chebvander(xc2, M-3)
        self.VI0 = np.linalg.inv(self.V0)
        self.VI1 = np.linalg.inv(self.V1)
        self.VI2 = np.linalg.inv(self.V2)
        # differentiation matrices
        DC01 = np.polynomial.chebyshev.chebder(np.eye(M-0)) / rat
        DC12 = np.polynomial.chebyshev.chebder(np.eye(M-1)) / rat
        DC00 = np.row_stack([DC01, np.zeros(M)])
        self.D00 = self.V0.dot(DC00.dot(self.VI0))
        self.D01 = self.V1.dot(DC01.dot(self.VI0))
        self.D12 = self.V2.dot(DC12.dot(self.VI1))
        # boundary condition operators
        self.ibc_dirichlet = np.polynomial.chebyshev.chebvander(1, M-1).dot(self.VI0)
        self.obc_dirichlet = np.polynomial.chebyshev.chebvander(-1, M-1).dot(self.VI0)
        self.ibc_neumann = self.ibc_dirichlet.dot(self.D00)
        self.obc_neumann = self.obc_dirichlet.dot(self.D00)
        # rank reduction operators
        temp = np.zeros([M-1, M-0], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.R01 = self.V1.dot(temp.dot(self.VI0))
        temp = np.zeros([M-2, M-1], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.R12 = self.V2.dot(temp.dot(self.VI1))
        self.R02 = self.R12.dot(self.R01)
        # get poof operator from M-1 --> M
        temp = np.zeros([M, M-1], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.P10 = self.V0.dot(temp.dot(self.VI1))

class ApproximateAnnularGeometry(object):
    """
    Approximate Annular Geometry for solving PDE in annular regions
    n: number of discrete points in tangential direction
    M: number of chebyshev modes in radial direction
    width: width of radial region
    approx_r: approximate radius of annulus
    """
    def __init__(self, n, M, width, approx_r):
        self.n = n
        self.M = M
        self.radius = approx_r
        self.width = width
        self.radial_h = self.width/self.M
        self.tangent_h = 2*np.pi/n
        self.ns = self.n - 1
        self.n2 = int(self.n/2)
        self.k = np.fft.fftfreq(self.n, 1.0/self.n)
        self.ks = concat(self.k[:self.n2], self.k[self.n2+1:])
        self.iks = 1j*self.ks
        # r grids
        _, self.rv0, rat0 = get_chebyshev_nodes(-self.width, 0.0, self.M-0)
        _, self.rv1, rat1 = get_chebyshev_nodes(-self.width, 0.0, self.M-1)
        _, self.rv2, rat2 = get_chebyshev_nodes(-self.width, 0.0, self.M-2)
        self.ratio = -rat0
        # coordinate transfromations
        self.approx_psi0 = self.radius+self.rv0
        self.approx_psi1 = self.radius+self.rv1
        self.approx_psi2 = self.radius+self.rv2
        self.approx_inv_psi0 = 1.0/self.approx_psi0
        self.approx_inv_psi1 = 1.0/self.approx_psi1
        self.approx_inv_psi2 = 1.0/self.approx_psi2
        # Chebyshev Operators
        self.CO = ChebyshevOperators(M, self.ratio)

class RealAnnularGeometry(object):
    def __init__(self, speed, curvature, AAG):
        k = np.fft.fftfreq(curvature.shape[0], 1.0/curvature.shape[0])
        dt_curvature = np.fft.ifft(np.fft.fft(curvature)*1j*k).real
        rv0 = AAG.rv0
        rv1 = AAG.rv1
        rv2 = AAG.rv2
        self.psi0 = speed*(1+rv0[:,None]*curvature)
        self.psi1 = speed*(1+rv1[:,None]*curvature)
        self.psi2 = speed*(1+rv2[:,None]*curvature)
        self.inv_psi0 = 1.0/self.psi0
        self.inv_psi1 = 1.0/self.psi1
        self.inv_psi2 = 1.0/self.psi2
        self.DR_psi2 = speed*curvature*np.ones(rv2[:,None].shape)
        denom2 = speed*(1+rv2[:,None]*curvature)**3
        idenom2 = 1.0/denom2
        # these are what i think it should be? need to check computation
        self.ipsi_DR_ipsi_DT_psi2 = (curvature-dt_curvature)*idenom2
        self.ipsi_DT_ipsi_DR_psi2 = -dt_curvature*idenom2
        # these are what work...
        self.ipsi_DR_ipsi_DT_psi2 = dt_curvature*idenom2
        self.ipsi_DT_ipsi_DR_psi2 = dt_curvature*idenom2



