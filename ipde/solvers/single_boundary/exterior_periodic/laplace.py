import numpy as np
import scipy as sp
import scipy.linalg
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator

Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Singular_SLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifcharge=True)
Singular_DLP = lambda src, _, sign: Laplace_Layer_Singular_Form(src, ifdipole=True) - sign*0.5*np.eye(src.N)
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)
Naive_DLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifdipole=True)

class PoissonSolver(object):
    """
    Inhomogeneous Laplace Solver on general domain
    """
    def __init__(self, ebdy, AnnularPoissonSolver):
        self.ebdy = ebdy
        self.interior = self.ebdy.interior
        self.annular_solver = AnnularPoissonSolver
        self._get_qfs()
        self.sign = 1 if self.interior else -1
        self.Singular_DLP = lambda src, _: Singular_DLP(src, _, self.sign)
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        self.interface_qfs_1 = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [Singular_SLP,],
                                Naive_SLP, on_surface=True, form_b2c=False)
        self.interface_qfs_2 = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [Singular_SLP,],
                                Naive_SLP, on_surface=True, form_b2c=False)
        # construct qfs evaluator for the boundary
        self.boundary_qfs = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [Singular_SLP, self.Singular_DLP],
                                Naive_SLP, on_surface=True, form_b2c=False)
    def __call__(self, f, fr):
        pass