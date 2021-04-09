import numpy as np
import pybie2d
from qfs.laplace_qfs import Laplace_QFS
from ipde.solvers.internals.scalar import ScalarHelper
from ipde.annular.poisson import AnnularPoissonSolver

Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply

class PoissonHelper(ScalarHelper):
    """
    Inhomogeneous Poisson Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None):
        super().__init__(ebdy, annular_solver)
    def _define_annular_solver(self):
        self.annular_solver = AnnularPoissonSolver(self.AAG)
    def _get_qfs(self):
        self.interface_qfs_g = Laplace_QFS(self.ebdy.interface, 
                                        self.interior, True, True)
        self.interface_qfs_r = Laplace_QFS(self.ebdy.interface, 
                                        not self.interior, True, True)
    def _define_layer_apply(self):
        def func(src, trg, ch):        
            return Laplace_Layer_Apply(src, trg, charge=ch, backend='fly')
        self.Layer_Apply = func
