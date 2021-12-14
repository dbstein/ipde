import numpy as np
import pybie2d
from pybie2d.kernels.high_level.modified_helmholtz import Modified_Helmholtz_Layer_Apply
from qfs.modified_helmholtz_qfs import Modified_Helmholtz_QFS
from ipde.solvers.internals.scalar import ScalarHelper
from ipde.annular.modified_helmholtz import AnnularModifiedHelmholtzSolver

class ModifiedHelmholtzHelper(ScalarHelper):
    """
    Inhomogeneous Modified-Helmholtz Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None, k=1.0):
        self.k = k
        super().__init__(ebdy, annular_solver)
    def _define_annular_solver(self):
        self.annular_solver = AnnularModifiedHelmholtzSolver(self.AAG, k=self.k)
    def _get_qfs(self):
        self.interface_qfs_g = Modified_Helmholtz_QFS(self.ebdy.interface, 
                                    self.interior, True, True, self.k)
        self.interface_qfs_r = Modified_Helmholtz_QFS(self.ebdy.interface, 
                                    not self.interior, True, True, self.k)
    def _define_layer_apply(self):
        self.Layer_Apply = lambda src, trg, ch: Modified_Helmholtz_Layer_Apply(src, trg, charge=ch, k=self.k)
