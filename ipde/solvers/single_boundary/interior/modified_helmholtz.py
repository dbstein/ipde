import numpy as np
import scipy as sp
import scipy.linalg
import pybie2d
from qfs.two_d_qfs import QFS_Evaluator
from ....annular.modified_helmholtz import AnnularModifiedHelmholtzSolver
from ....annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ....derivatives import fourier, fd_x_4, fd_y_4
from ....utilities import affine_transformation

PointSet = pybie2d.point_set.PointSet
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

class ModifiedHelmholtzSolver(object):
    """
    Inhomogeneous Modified-Helmholtz Solver on general domain
    """
    def __init__(self, ebdy, k, solver_type='spectral', AMHS=None):
        """
        Type can either be 'spectral' or 'fourth'
        """
        self.ebdy = ebdy
        self.type = solver_type
        self.interior = self.ebdy.interior
        self.solver_type = solver_type
        self.k = k
        if AMHS is None:
            AAG = ApproximateAnnularGeometry(self.ebdy.bdy.N, self.ebdy.M,
                self.ebdy.radial_width, self.ebdy.approximate_radius)
            AMHS = AnnularModifiedHelmholtzSolver(AAG, k=self.k)
        self.annular_solver = AMHS
        sp =  ebdy.bdy.speed
        cur = ebdy.bdy.curvature
        self.RAG = RealAnnularGeometry(sp, cur, self.annular_solver.AAG)
        self.sign = 1 if self.interior else -1
        self.Singular_SLP = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=self.k)
        self.Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, k=self.k, ifcharge=True)
        self._get_qfs()
        kxv = np.fft.fftfreq(self.ebdy.grid.Nx, self.ebdy.grid.xh/(2*np.pi))
        kyv = np.fft.fftfreq(self.ebdy.grid.Ny, self.ebdy.grid.yh/(2*np.pi))
        self.kx, self.ky = np.meshgrid(kxv, kyv, indexing='ij')
        self.ikx, self.iky = 1j*self.kx, 1j*self.ky
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self.helm = self.k**2 - self.lap
        self.ihelm = 1.0/self.helm
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
    def _get_qfs(self):
        # construct qfs evaluators for the interface
        self.interface_qfs_1 = QFS_Evaluator(self.ebdy.interface_qfs,
                                self.interior, [self.Singular_SLP,],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
        self.interface_qfs_2 = QFS_Evaluator(self.ebdy.interface_qfs,
                                not self.interior, [self.Singular_SLP,],
                                self.Naive_SLP, on_surface=True, form_b2c=False)
    def get_bv(self, ur):
        return self.annular_solver.AAG.CO.obc_dirichlet[0].dot(ur)
    def get_bn(self, ur):
        return self.annular_solver.AAG.CO.obc_neumann[0].dot(ur)
    def __call__(self, f, fr, **kwargs):
        ebdy = self.ebdy
        # get the grid-based solution
        fc = f*ebdy.grid_step
        uc = np.fft.ifft2(np.fft.fft2(fc)*self.ihelm).real
        # evaluate this on the interface
        bv = ebdy.interpolate_grid_to_interface(uc, order=self.interpolation_order)
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
        rslp = MH_Layer_Apply(iq.exterior_source_bdy, self.radp, charge=sigma2, k=self.k)
        gslp = MH_Layer_Apply(iq.interior_source_bdy, self.gridp, charge=sigma1, k=self.k)
        bslp = MH_Layer_Apply(iq.exterior_source_bdy, ebdy.bdy, charge=sigma2, k=self.k)
        # add these to the current solution
        uc[ebdy.grid_not_in_annulus] += gslp
        ur += rslp.reshape(ur.shape)
        # interpolate ur onto uc
        ebdy.interpolate_radial_to_grid(ur, uc)
        uc *= ebdy.phys
        return uc, ur


