import numpy as np
import pybie2d
from pyfmmlib2d import SFMM
from qfs.two_d_qfs import QFS_Evaluator
from qfs.two_d_qfs import QFS_Evaluator_Pressure
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

def Naive_SLP(src, trg):
    return Stokes_Layer_Form(src, trg, ifforce=True)
def Naive_DLP(src, trg):
    return Stokes_Layer_Form(src, trg, ifdipole=True)

# SLP with pressure evaluation at 0th target point
def PSLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    return out

# DLP with pressure evaluation at 0th target point
def PDLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_DLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    rdotn = dx*src.normal_x + dy*src.normal_y
    ir2 = 1.0/r2
    rdotnir4 = rdotn*ir2*ir2
    out[-1, 0*src.N:1*src.N] = (-src.normal_x*ir2 + 2*rdotnir4*dx)*src.weights
    out[-1, 1*src.N:2*src.N] = (-src.normal_y*ir2 + 2*rdotnir4*dy)*src.weights
    out[-1] /= np.pi
    return out

# SLP with pressure null-space correction; fixing scale to eval at 0th target point
def Pressure_SLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N+1])
    out[:-1,:-1] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    out[0*trg.N:1*trg.N, -1] = trg.normal_x*trg.weights
    out[1*trg.N:2*trg.N, -1] = trg.normal_y*trg.weights
    return out

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
        # singular construction; not in use right now
        # until I make a version of QFS that is compatible with
        # pressure fixes and the singular operators
        if False:
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
        else: # the non-singular construction
            self.interface_qfs_g = QFS_Evaluator_Pressure(self.ebdy.interface_qfs,
                                    self.interior, b2c_funcs=[PSLP, PDLP], s2c_func=Pressure_SLP, form_b2c=True)
            self.interface_qfs_r = QFS_Evaluator_Pressure(self.ebdy.interface_qfs,
                                    not self.interior, b2c_funcs=[PSLP, PDLP], s2c_func=Pressure_SLP, form_b2c=True)
    def _define_layer_apply(self):
        def Layer_Apply(src, trg, f):
            s = src.get_stacked_boundary()
            t = trg.get_stacked_boundary()
            out = SFMM(source=s, target=t, forces=f*src.weights, compute_target_velocity=True, compute_target_stress=True)
            u = out['target']['u']
            v = out['target']['v']
            p = out['target']['p']
            return u, v, p
        self.Layer_Apply = Layer_Apply
