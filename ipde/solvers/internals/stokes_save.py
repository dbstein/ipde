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

def eval_p1(src, px, py, slp, dlp=None):
    dx = px - src.x
    dy = py - src.y
    r2 = dx*dx + dy*dy
    ir2 = 1/r2
    sir2 = ir2*0.5/np.pi*src.weights
    p = np.sum(dx*sir2*slp[0]) + np.sum(dy*sir2*slp[1])
    if dlp is not None:
        rdotnir4 = (dx*src.normal_x + dy*src.normal_y)*ir2*ir2
        wx = (-src.normal_x*ir2 + 2*rdotnir4*dx)/np.pi*src.weights
        wy = (-src.normal_y*ir2 + 2*rdotnir4*dy)/np.pi*src.weights
        p += np.sum(wx*dlp[0]) + np.sum(wy*dlp[1])
    return p
def trs(f):
    if len(f.shape) == 1:
        N = int(f.shape[0]//2)
        return f.reshape(2, N)
    else:
        return f
class Stokes_Converter(object):
    def __init__(self, ebdy, interior=True, interior_point=None):
        qfs = ebdy.interface_qfs
        bdy = ebdy.interface
        h = bdy.speed[0]*bdy.dt
        self.bdy = bdy
        self.interior = interior
        if self.interior:
            self.src = qfs.interior_source_bdy
            sign = -1
        else:
            self.src = qfs.exterior_source_bdy
            sign = 1
        if interior_point is None:
            self.pressure_targ_x = bdy.x[0] + sign*6*bdy.normal_x[0]*h
            self.pressure_targ_y = bdy.y[0] + sign*6*bdy.normal_y[0]*h
        else:
            self.pressure_targ_x = interior_point[0]
            self.pressure_targ_y = interior_point[1]
        self.sh = [2,self.src.N]
        Singular_SLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifforce=True)
        Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) + sign*0.5*np.eye(2*src.N)
        Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
        def Stokes_Pressure_Fix(src, trg):
            Nxx = trg.normal_x[:,None]*src.normal_x
            Nxy = trg.normal_x[:,None]*src.normal_y
            Nyx = trg.normal_y[:,None]*src.normal_x
            Nyy = trg.normal_y[:,None]*src.normal_y
            NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
            return NN
        def Fixed_SLP(src, trg):
            return Naive_SLP(src, trg) + Stokes_Pressure_Fix(src, trg)
        self.qfs = QFS_Evaluator(qfs, self.interior, b2c_funcs=[Singular_SLP,Singular_DLP], s2c_func=Fixed_SLP, form_b2c=True, on_surface=True, vector=True)
    def __call__(self, taul):
        sigma = self.qfs(taul).reshape(self.sh)
        pe = eval_p1(self.src, self.pressure_targ_x, self.pressure_targ_y, sigma)
        pt = eval_p1(self.bdy, self.pressure_targ_x, self.pressure_targ_y, trs(taul[0]), trs(taul[1]))
        pd = pe - pt
        sigma[0] += self.src.normal_x*pd
        sigma[1] += self.src.normal_y*pd
        return sigma

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
            self.interface_qfs_g = Stokes_Converter(self.ebdy, self.interior)
            self.interface_qfs_r = Stokes_Converter(self.ebdy, not self.interior)
            # self.interface_qfs_g = QFS_Evaluator_Pressure(self.ebdy.interface_qfs,
            #                         self.interior, b2c_funcs=[PSLP, PDLP], s2c_func=Pressure_SLP, form_b2c=True)
            # self.interface_qfs_r = QFS_Evaluator_Pressure(self.ebdy.interface_qfs,
            #                         not self.interior, b2c_funcs=[PSLP, PDLP], s2c_func=Pressure_SLP, form_b2c=True)
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
