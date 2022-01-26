import numpy as np
import pybie2d
from pybie2d.kernels.high_level.modified_helmholtz import Modified_Helmholtz_Layer_Apply
from qfs.modified_helmholtz_qfs import Modified_Helmholtz_QFS
from ipde.solvers.internals.scalar import ScalarHelper
from ipde.annular.modified_helmholtz import AnnularModifiedHelmholtzSolver

try:
    import fmm2dpy
except:
    pass

class ModifiedHelmholtzHelper(ScalarHelper):
    """
    Inhomogeneous Modified-Helmholtz Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None, k=1.0, source_upsample_factor=1.0, grid_backend='pybie2d'):
        self.k = k
        self.source_upsample_factor = source_upsample_factor
        super().__init__(ebdy, annular_solver, grid_backend)
    def _define_annular_solver(self):
        self.annular_solver = AnnularModifiedHelmholtzSolver(self.AAG, k=self.k)
    def _get_qfs(self):
        self.interface_qfs_g = Modified_Helmholtz_QFS(self.ebdy.interface, 
                                    self.interior, True, True, self.k, source_upsample_factor=self.source_upsample_factor, closer_source=True)
        self.interface_qfs_r = Modified_Helmholtz_QFS(self.ebdy.interface, 
                                    not self.interior, True, True, self.k, source_upsample_factor=self.source_upsample_factor, closer_source=True)
    def _define_layer_apply(self):
        if self.grid_backend == 'fmm2d':
            print('FMM2D!')
            def func(src, trg, ch):
                chh = (ch*src.weights).astype(complex)
                out = fmm2dpy.hfmm2d(eps=1e-14, zk=0.0+1j*self.k, sources=src.get_stacked_boundary(), charges=chh, targets=trg.get_stacked_boundary(), pgt=1)
                return out.pottarg.real
            self.Layer_Apply = func
        else:
            self.Layer_Apply = lambda src, trg, ch: Modified_Helmholtz_Layer_Apply(src, trg, charge=ch, k=self.k)
