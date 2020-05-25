import numpy as np
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from .scalar import ScalarHelper
from ...annular.poisson import AnnularPoissonSolver

Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form

class PoissonHelper(ScalarHelper):
    """
    Inhomogeneous Poisson Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None):
        super().__init__(ebdy, annular_solver)
    def _define_annular_solver(self):
        self.annular_solver = AnnularPoissonSolver(self.AAG)
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        self.Singular_SLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifcharge=True)
        self.Singular_DLP_I = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(src.N)
        self.Singular_DLP_E = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) + 0.5*np.eye(src.N)
        self.Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)
        self.interface_qfs_g = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [self.Singular_SLP, self.Singular_DLP_I],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
        self.interface_qfs_r = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [self.Singular_SLP, self.Singular_DLP_E],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
    def _define_layer_apply(self):
        def func(src, trg, ch):        
            # xmin = min(src.x.min(), trg.x.min())
            # xmax = max(src.x.max(), trg.x.max())
            # ymin = min(src.y.min(), trg.y.min())
            # ymax = max(src.y.max(), trg.y.max())
            # return Laplace_Layer_Apply(src, trg, charge=ch, backend='KIFMM', bbox=[xmin, xmax, ymin, ymax], Nequiv=40, Ncutoff=20)
            return Laplace_Layer_Apply(src, trg, charge=ch, backend='fly')
        self.Layer_Apply = func
