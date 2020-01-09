import numpy as np
import scipy as sp
import scipy.linalg
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from ....annular.stokes import AnnularStokesSolver
from ....annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ....derivatives import fourier, fd_x_4, fd_y_4
from ....utilities import affine_transformation

PointSet = pybie2d.point_set.PointSet
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form

# SLP Function with fixed pressure nullspace
def Fixed_SLP(src, trg, Naive_SLP):
    Nxx = trg.normal_x[:,None]*src.normal_x
    Nxy = trg.normal_x[:,None]*src.normal_y
    Nyx = trg.normal_y[:,None]*src.normal_x
    Nyy = trg.normal_y[:,None]*src.normal_y
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
    MAT = Naive_SLP(src, trg) + NN
    return MAT

def v2f(x):
    return x.reshape(2, x.size//2)

class StokesSolver(object):
    """
    Inhomogeneous Stokes Solver on general domain
    """
    def __init__(self, ebdy, bump, bump_loc, solver_type='spectral', ASS=None):
        """
        Type can either be 'spectral' or 'fourth'
        """
        self.ebdy = ebdy
        self.type = solver_type
        self.interior = self.ebdy.interior
        self.solver_type = solver_type
        if ASS is None:
            AAG = ApproximateAnnularGeometry(self.ebdy.bdy.N, self.ebdy.M,
                self.ebdy.radial_width, self.ebdy.approximate_radius)
            ASS = AnnularStokesSolver(AAG, mu=1.0)
        self.annular_solver = ASS
        sp =  ebdy.bdy.speed
        cur = ebdy.bdy.curvature
        self.RAG = RealAnnularGeometry(sp, cur, self.annular_solver.AAG)
        self.sign = 1 if self.interior else -1
        self.Singular_SLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifforce=True)
        self.Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True)
        self.Singular_DLP_I = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True)  - 0.5*np.eye(2*self.ebdy.bdy.N)
        self.Singular_DLP_E = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True)  + 0.5*np.eye(2*self.ebdy.bdy.N)
        self.Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
        self.Fixed_SLP = lambda src, trg: Fixed_SLP(src, trg, self.Naive_SLP)
        self._get_qfs()
        kxv = np.fft.fftfreq(self.ebdy.grid.Nx, self.ebdy.grid.xh/(2*np.pi))
        kyv = np.fft.fftfreq(self.ebdy.grid.Ny, self.ebdy.grid.yh/(2*np.pi))
        self.kx, self.ky = np.meshgrid(kxv, kyv, indexing='ij')
        self.ikx, self.iky = 1j*self.kx, 1j*self.ky
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.lap[0,0] = np.Inf
        self.ilap = 1.0/self.lap
        self.lap[0,0] = 0.0
        if self.solver_type == 'spectral':
            self.dx = lambda x: fourier(x, self.ikx)
            self.dy = lambda x: fourier(x, self.iky)
        else:
            self.dx = lambda x: fd_x_4(x, self.ebdy.grid.xh)
            self.dy = lambda x: fd_y_4(x, self.ebdy.grid.yh)
        self.radp = PointSet(ebdy.radial_x.ravel(), ebdy.radial_y.ravel())
        self.gridp = PointSet(ebdy.grid.xg[ebdy.grid_not_in_annulus].ravel(), ebdy.grid.yg[ebdy.grid_not_in_annulus].ravel())
        self.gridpa = PointSet(ebdy.grid.xg[ebdy.phys].ravel(), ebdy.grid.yg[ebdy.phys].ravel())
        self.interpolation_order = 3 if self.solver_type == 'fourth' else np.Inf
        grr = np.sqrt((self.ebdy.grid.xg-bump_loc[0])**2 + (self.ebdy.grid.yg-bump_loc[1])**2)
        self.bumpy = bump(affine_transformation(grr, 0, self.ebdy.radial_width, 0, 1))
        self.bumpy /= np.sum(self.bumpy)*self.ebdy.grid.xh*self.ebdy.grid.yh
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        self.interface_qfs_1 = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [self.Singular_SLP, self.Singular_DLP_I],
                                self.Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
        self.interface_qfs_2 = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [self.Singular_SLP, self.Singular_DLP_E],
                                self.Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
    def get_bv(self, ur, vr):
        bu = self.annular_solver.AAG.CO.obc_dirichlet[0].dot(ur)
        bv = self.annular_solver.AAG.CO.obc_dirichlet[0].dot(vr)
        return bu, bv
    def get_bt(self, ur, vr):
        # STILL TO IMPLEMENT
        return self.annular_solver.AAG.CO.obc_neumann[0].dot(ur)
    def __call__(self, fu, fv, fur, fvr, **kwargs):
        ebdy = self.ebdy
        # get the grid-based solution
        fuc = fu*ebdy.grid_step
        fvc = fv*ebdy.grid_step
        fuc -= self.bumpy*np.sum(fuc)*ebdy.grid.xh*ebdy.grid.yh
        fvc -= self.bumpy*np.sum(fvc)*ebdy.grid.xh*ebdy.grid.yh
        fuch = np.fft.fft2(fuc)
        fvch = np.fft.fft2(fvc)
        cph = self.ilap*(self.ikx*fuch + self.iky*fvch)
        cuh = self.ilap*(self.ikx*cph - fuch)
        cvh = self.ilap*(self.iky*cph - fvch)
        uc = np.fft.ifft2(cuh).real
        vc = np.fft.ifft2(cvh).real
        pc = np.fft.ifft2(cph).real
        # evaluate this on the interface
        bu = ebdy.interpolate_grid_to_interface(uc, order=self.interpolation_order)
        bv = ebdy.interpolate_grid_to_interface(vc, order=self.interpolation_order)
        # evaluate the traction on the interface (optimziations here!)
        ucx, ucy = self.dx(uc), self.dy(uc)
        vcx, vcy = self.dx(vc), self.dy(vc)
        bucx = ebdy.interpolate_grid_to_interface(ucx, order=self.interpolation_order)
        bucy = ebdy.interpolate_grid_to_interface(ucy, order=self.interpolation_order)
        bvcx = ebdy.interpolate_grid_to_interface(vcx, order=self.interpolation_order)
        bvcy = ebdy.interpolate_grid_to_interface(vcy, order=self.interpolation_order)
        bpc = ebdy.interpolate_grid_to_interface(pc, order=self.interpolation_order)
        btxx = 2*bucx - bpc
        btxy = bucy + bvcx
        btyy = 2*bvcy - bpc
        btx = btxx*ebdy.interface.normal_x + btxy*ebdy.interface.normal_y
        bty = btxy*ebdy.interface.normal_x + btyy*ebdy.interface.normal_y
        # solve the radial problem (just use homogeneous 0 bcs)
        fr, ft = ebdy.convert_uv_to_rt(fur, fvr)
        zer = np.zeros(ebdy.bdy.N)
        rr, tr, pr = self.annular_solver.solve(self.RAG, fr, ft, zer, zer, zer, zer, verbose=True, tol=1e-12)
        ur, vr = ebdy.convert_rt_to_uv(rr, tr)
        # get the traction due to the radial solution (there's a better way to do this!)
        rux, ruy = ebdy.radial_grid_derivatives(ur)
        rvx, rvy = ebdy.radial_grid_derivatives(vr)
        EVAL = self.annular_solver.AAG.CO.ibc_dirichlet[0]
        brux, bruy = EVAL.dot(rux), EVAL.dot(ruy)
        brvx, brvy = EVAL.dot(rvx), EVAL.dot(rvy)
        brp = EVAL.dot(pr)
        rtxx = 2*brux - brp
        rtxy = bruy + brvx
        rtyy = 2*brvy - brp
        rtx = rtxx*ebdy.interface.normal_x + rtxy*ebdy.interface.normal_y
        rty = rtxy*ebdy.interface.normal_x + rtyy*ebdy.interface.normal_y
        # get the single/double layer that needs to be computed
        taus =  np.concatenate([ rtx-btx, rty-bty ])
        taud = -np.concatenate([ -bu,     -bv   ])
        # get effective layer potentials for this
        sigma1 = self.interface_qfs_1([taus, taud])
        sigma2 = self.interface_qfs_2([taus, taud])
        # evaluate these where they need to go
        iq = ebdy.interface_qfs
        rslp = Stokes_Layer_Apply(iq.exterior_source_bdy, self.radp, forces=v2f(sigma2), out_type='stacked')
        gslp = Stokes_Layer_Apply(iq.interior_source_bdy, self.gridp, forces=v2f(sigma1), out_type='stacked')
        # add these to the current solution
        uc[ebdy.grid_not_in_annulus] += gslp[0]
        vc[ebdy.grid_not_in_annulus] += gslp[1]
        ur += rslp[0].reshape(ur.shape)
        vr += rslp[1].reshape(ur.shape)
        # STILL NEED TO FIX P!
        # interpolate radial portions onto the grid
        ebdy.interpolate_radial_to_grid(ur, uc)
        ebdy.interpolate_radial_to_grid(vr, vc)
        ebdy.interpolate_radial_to_grid(pr, pc)
        uc *= ebdy.phys
        vc *= ebdy.phys
        pc *= ebdy.phys
        return uc, vc, pc, ur, vr, pr


