import numpy as np
from .vector import VectorSolver
from ..internals.stokes import StokesHelper
from ...utilities import affine_transformation

class StokesSolver(VectorSolver):
    def __init__(self, ebdyc, solver_type='spectral', AS_list=None):
        # check that ebdyc has bumps defined
        if not ebdyc.bumpy_readied:
            raise Exception('Stokes solver requires embedded boundary collection with a bump function.')
        super().__init__(ebdyc, solver_type, AS_list)
    def _get_helper(self, ebdy, AS):
        return StokesHelper(ebdy, AS)
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
