import numpy as np
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from .vector import VectorHelper
from ...annular.stokes import AnnularStokesSolver

Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form

# SLP Function with fixed pressure nullspace
def Fixed_SLP(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x
    Nxy = trg.normal_x[:,None]*src.normal_y
    Nyx = trg.normal_y[:,None]*src.normal_x
    Nyy = trg.normal_y[:,None]*src.normal_y
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
    MAT = Stokes_Layer_Form(src, trg, ifforce=True) + NN
    return MAT

class StokesHelper(VectorHelper):
    """
    Inhomogeneous Stokes Solver on general domain
    """
    def __init__(self, ebdy, annular_solver=None):
        super().__init__(ebdy, annular_solver)
    def _define_annular_solver(self):
        self.annular_solver = AnnularStokesSolver(self.AAG, mu=1.0)
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        sign = 1 if self.interior else -1
        self.Singular_SLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifforce=True)
        self.Singular_DLP_G = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - sign*0.5*np.eye(2*self.ebdy.bdy.N)
        self.Singular_DLP_R = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) + sign*0.5*np.eye(2*self.ebdy.bdy.N)
        self.Naive_SLP = lambda src, trg: Fixed_SLP(src, trg)
        self.interface_qfs_g = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [self.Singular_SLP, self.Singular_DLP_G],
                                self.Naive_SLP, on_surface=True, form_b2c=False, vector=True)
        self.interface_qfs_r = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [self.Singular_SLP, self.Singular_DLP_R],
                                self.Naive_SLP, on_surface=True, form_b2c=False, vector=True)
    def _define_layer_apply(self):
        self.Layer_Apply = lambda src, trg, f: Stokes_Layer_Apply(src, trg, forces=f, out_type='stacked')
