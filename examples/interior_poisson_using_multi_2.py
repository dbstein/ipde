import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.poisson import PoissonSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Singular_DLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(src.N)
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)

nb = 1400
M = 18
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 2*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.2, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1.0, pad_zone, MOL.step)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (1.2-ebdy.radial_width, 1.2-ebdy.radial_width), ebdy.radial_width)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = EmbeddedFunction(ebdyc)
f.define_via_function(force_func)
ua = EmbeddedFunction(ebdyc)
ua.define_via_function(solution_func)
bcs = BoundaryFunction(ebdyc)
bcs.define_via_function(solution_func)

################################################################################
# Setup Poisson Solver

solver = PoissonSolver(ebdyc, solver_type=solver_type)
ue = solver(f, tol=1e-12, verbose=verbose)

if plot:
	fig, ax = plt.subplots()
	ue.plot(ax)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# this isn't correct yet because we haven't applied boundary conditions
A = Laplace_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(bdy.N)
bv = solver.get_boundary_values(ue)
ba = bcs - bv
tau = np.linalg.solve(A, ba.aggregate())
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])
out = Laplace_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigma)
ue_adj = ebdyc.divide_grid_and_radial2(out)
ue += ue_adj

if plot:
	fig, ax = plt.subplots()
	ue.plot(ax)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# compute the error
err = np.abs(ue - ua)
gerr = err.grid_max()
rerr = err.radial_max()

if plot:
	fig, ax = plt.subplots()
	clf = (err+1e-15).plot(ax, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error in grid:    {:0.2e}'.format(gerr.max()))
print('Error in annulus: {:0.2e}'.format(rerr.max()))
