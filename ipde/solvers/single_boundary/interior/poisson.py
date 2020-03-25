import numpy as np
import scipy as sp
import scipy.linalg
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from ....annular.poisson import AnnularPoissonSolver
from ....annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ....derivatives import fourier, fd_x_4, fd_y_4
from ....utilities import affine_transformation

PointSet = pybie2d.point_set.PointSet
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Singular_SLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifcharge=True)
Singular_DLP = lambda src, _, sign: Laplace_Layer_Singular_Form(src, ifdipole=True) - sign*0.5*np.eye(src.N)
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)
Naive_DLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifdipole=True)

class PoissonSolver(object):
    """
    Inhomogeneous Laplace Solver on general domain
    """
    def __init__(self, ebdy, bump, bump_loc, solver_type='spectral', APS=None):
        """
        Type can either be 'spectral' or 'fourth'
        """
        self.ebdy = ebdy
        self.type = solver_type
        self.interior = self.ebdy.interior
        self.solver_type = solver_type
        if APS is None:
            AAG = ApproximateAnnularGeometry(self.ebdy.bdy.N, self.ebdy.M,
                self.ebdy.radial_width, self.ebdy.approximate_radius)
            APS = AnnularPoissonSolver(AAG)
        self.annular_solver = APS
        sp =  ebdy.bdy.speed
        cur = ebdy.bdy.curvature
        self.RAG = RealAnnularGeometry(sp, cur, APS.AAG)
        self.sign = 1 if self.interior else -1
        self.Singular_DLP = lambda src, _: Singular_DLP(src, _, self.sign)
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
                                self.interior, [Singular_SLP,],
                                Naive_SLP, on_surface=True, form_b2c=False)
        self.interface_qfs_2 = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [Singular_SLP,],
                                Naive_SLP, on_surface=True, form_b2c=False)
    def get_bv(self, ur):
        return self.annular_solver.AAG.CO.obc_dirichlet[0].dot(ur)
    def get_bn(self, ur):
        return self.annular_solver.AAG.CO.obc_neumann[0].dot(ur)
    def __call__(self, f, fr, **kwargs):
        ebdy = self.ebdy
        # get the grid-based solution
        fc = f*ebdy.grid_step
        fc -= self.bumpy*np.sum(fc)*ebdy.grid.xh*ebdy.grid.yh
        uc = np.fft.ifft2(np.fft.fft2(fc)*self.ilap).real
        # evaluate this on the interface
        bv = ebdy.interpolate_grid_to_interface(uc, order=self.interpolation_order)
        print(np.abs(bv).sum())
        # take the gradient of uc and evaluate on interface
        ucx, ucy = self.dx(uc), self.dy(uc)
        bx = ebdy.interpolate_grid_to_interface(ucx, order=self.interpolation_order)
        by = ebdy.interpolate_grid_to_interface(ucy, order=self.interpolation_order)
        ucn = bx*ebdy.bdy.normal_x + by*ebdy.bdy.normal_y
        # compute the radial solution
        ur = self.annular_solver.solve(self.RAG, fr, bv, bv, **kwargs)
        # evaluate the normal derivative of the radial solution
        DER = self.annular_solver.AAG.CO.ibc_neumann[0]
        urn = np.array(DER.dot(ur))
        # get the single layer to smooth the interface
        tau = urn - ucn
        # get effective layer potentials for this
        sigma1 = self.interface_qfs_1([tau,])
        sigma2 = self.interface_qfs_2([tau,])
        # evaluate these where they need to go
        iq = ebdy.interface_qfs
        rslp = Laplace_Layer_Apply(iq.exterior_source_bdy, self.radp, charge=sigma2)
        gslp = Laplace_Layer_Apply(iq.interior_source_bdy, self.gridp, charge=sigma1)
        bslp = Laplace_Layer_Apply(iq.exterior_source_bdy, ebdy.bdy, charge=sigma2)
        # add these to the current solution
        uc[ebdy.grid_not_in_annulus] += gslp
        ur += rslp.reshape(ur.shape)
        # interpolate ur onto uc
        ebdy.interpolate_radial_to_grid(ur, uc)
        uc *= ebdy.phys
        return uc, ur


