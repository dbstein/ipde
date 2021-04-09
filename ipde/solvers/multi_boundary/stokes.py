import numpy as np
from .vector import VectorSolver
from ipde.solvers.internals.stokes import StokesHelper
from ipde.utilities import affine_transformation

class StokesSolver(VectorSolver):
    def __init__(self, ebdyc, solver_type='spectral', helpers=None):
        # check that ebdyc has bumps defined
        if not ebdyc.bumpy_readied:
            raise Exception('Stokes solver requires embedded boundary collection with a bump function.')
        super().__init__(ebdyc, solver_type, helpers)
    def _get_helper_compatability(self, ebdy, helper):
        """
        Returns 0, 1, or 2
        0:  no compatiblity --- helper is worthless, start over
        1:  some compatibility --- annular solver is okay, use that
        2:  full compatibility --- start from scratch
        """
        if helper == None:
            return 0
        if ebdy.bdy.N != helper.ebdy.bdy.N:
            return 0
        if helper.ebdy is not ebdy:
            return 1
        return 2
    def _get_helper(self, ebdy, helper):
        c = self._get_helper_compatability(ebdy, helper)
        if c == 0:
            return StokesHelper(ebdy)
        elif c == 1:
            return Stokes_Helper(ebdy, helper.annular_solver)
        else:
            return helper
    def _grid_solve(self, fuc, fvc):
        fuc = self.ebdyc.demean_function(fuc)
        fvc = self.ebdyc.demean_function(fvc)
        fuch = np.fft.fft2(fuc)
        fvch = np.fft.fft2(fvc)
        cph = self.ilap*(self.ikx*fuch + self.iky*fvch)
        cuh = self.ilap*(self.ikx*cph - fuch)
        cvh = self.ilap*(self.iky*cph - fvch)
        uc = np.fft.ifft2(cuh).real
        vc = np.fft.ifft2(cvh).real
        pc = np.fft.ifft2(cph).real
        return uc, vc, pc
    def _get_specific_operators(self):
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.lap[0,0] = np.Inf
        self.ilap = 1.0/self.lap
        self.lap[0,0] = 0.0
