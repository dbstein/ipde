import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.single_boundary.interior.modified_helmholtz import ModifiedHelmholtzSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

nb = 200
helmholtz_k = 2.0
M = 8
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 2*M

solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid, verbose=verbose)

################################################################################
# Get solution, forces, BCs

k = 1*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
fr = force_func(ebdy.radial_x, ebdy.radial_y)
ua = solution_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
uar = solution_func(ebdy.radial_x, ebdy.radial_y)
bc = solution_func(ebdy.bdy.x, ebdy.bdy.y)

################################################################################
# Setup Poisson Solver

solver = ModifiedHelmholtzSolver(ebdy, helmholtz_k, solver_type=solver_type)
ue, uer = solver(f, fr, tol=1e-12, verbose=verbose)
mue = np.ma.array(ue, mask=ebdy.ext)

if plot:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# this isn't correct yet because we haven't applied boundary conditions
A = bdy.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) - 0.5*np.eye(bdy.N)*helmholtz_k**2
bv = solver.get_bv(uer)
tau = np.linalg.solve(A, bc-bv)
Singular_DLP = lambda src, _: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) - 0.5*np.eye(src.N)*helmholtz_k**2
Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])
rslp = MH_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.radp, charge=sigma, k=helmholtz_k)
gslp = MH_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.gridpa, charge=sigma, k=helmholtz_k)
uer += rslp.reshape(uer.shape)
ue[ebdy.phys] += gslp

if plot:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# compute the error
rerr = np.abs(uer - uar)
gerr = np.abs(ue - ua)
gerrp = gerr[ebdy.phys]
mgerr = np.ma.array(gerr, mask=ebdy.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mgerr + 1e-15, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error in grid:    {:0.2e}'.format(gerrp.max()))
print('Error in annulus: {:0.2e}'.format(rerr.max()))
