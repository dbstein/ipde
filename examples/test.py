import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.poisson import PoissonSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Singular_DLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(src.N)
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)

nb = 1000
M = 16
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 2*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.0, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone, MOL.step)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (1.2-ebdy.radial_width, 1.2-ebdy.radial_width), ebdys[0].radial_width)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua = solution_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
uar = uars[0]
bcs2v = solution_func(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2l = ebdyc.v2l(bcs2v)

################################################################################
# Setup Poisson Solver

solver = PoissonSolver(ebdyc, solver_type=solver_type)
ue, uers = solver(f, frs, tol=1e-12, verbose=verbose)

qfs = solver.helpers[0].interface_qfs_g
M = qfs.s2c_mat

# solve M via circulant math
theta = np.linspace(0, 2*np.pi, bdy.N, endpoint=False)
bc = np.sin(theta*10)
LU = sp.linalg.lu_factor(M)
x1 = np.linalg.solve(M, bc)
x2 = sp.linalg.lu_solve(LU, bc)
x3 = np.fft.ifft(np.fft.fft(bc)/np.fft.fft(M[0])).real
from personal_utilities.scipy_gmres import gmres, right_gmres
def prec_func(x):
	return np.fft.ifft(np.fft.fft(x)/np.fft.fft(M[0])).real
prec = sp.sparse.linalg.LinearOperator((bdy.N, bdy.N), dtype=float, matvec=prec_func)
out = gmres(M, np.fft.fft(bc), verbose=True, tol=1e-5, convergence='resid')[0]


