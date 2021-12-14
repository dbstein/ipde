import numpy as np
import pybie2d
from pyfmmlib2d import SFMM
from qfs.stokes_qfs import Stokes_QFS
from ipde.solvers.internals.vector import VectorHelper
from ipde.annular.stokes import AnnularStokesSolver

Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply

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
        self.interface_qfs_g = Stokes_QFS(self.ebdy.interface, 
                                                   self.interior, True, True)
        self.interface_qfs_r = Stokes_QFS(self.ebdy.interface, 
                                        not self.interior, True, True)
    def _define_layer_apply(self):
        def Layer_Apply(src, trg, f):
            s = src.get_stacked_boundary()
            t = trg.get_stacked_boundary()
            out = SFMM(source=s, target=t, forces=f*src.weights, \
                compute_target_velocity=True, compute_target_stress=True)
            u = out['target']['u']
            v = out['target']['v']
            p = out['target']['p']
            return u, v, p
        self.Layer_Apply = Layer_Apply
