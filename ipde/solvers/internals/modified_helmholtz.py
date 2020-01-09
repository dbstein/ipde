import numpy as np
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from .scalar import ScalarHelper
from ...annular.modified_helmholtz import AnnularModifiedHelmholtzSolver

MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

class ModifiedHelmholtzHelper(ScalarHelper):
    """
    Inhomogeneous Modified-Helmholtz Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None, k=1.0):
        super().__init__(ebdy, annular_solver, k=k)
    def _extract_extra_kwargs(self, k=None):
        self.k = k
    def _define_annular_solver(self):
        self.annular_solver = AnnularModifiedHelmholtzSolver(self.AAG, k=self.k)
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        sign = 1 if self.interior else -1
        self.Singular_SLP = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=self.k)
        self.Singular_DLP_G = lambda src, _: src.Modified_Helmholtz_DLP_Self_Form(k=self.k) - sign*0.5*np.eye(src.N)
        self.Singular_DLP_R = lambda src, _: src.Modified_Helmholtz_DLP_Self_Form(k=self.k) + sign*0.5*np.eye(src.N)
        self.Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, k=self.k, ifcharge=True)
        self.interface_qfs_g = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [self.Singular_SLP, self.Singular_DLP_G],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
        self.interface_qfs_r = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [self.Singular_SLP, self.Singular_DLP_R],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
    def _define_layer_apply(self):
        self.Layer_Apply = lambda src, trg, ch: MH_Layer_Apply(src, trg, charge=ch, k=self.k)
