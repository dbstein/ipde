import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.ebdy_collection import BoundaryFunction
from ipde.embedded_function import EmbeddedFunction
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

problem = 'easy'

nb = 800
M = 20
pad_zone = 0
verbose = True
plot = True
reparametrize = False
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral
solver_tol = 1e-14
coordinate_scheme = 'nufft'
coordinate_tolerance = 1e-14
qfs_tolerance = 1e-14
grid_upsample = 1

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.2, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
# get h to use for radial solvers and grid
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh/grid_upsample, pad_zone=0, heaviside=MOL.step, qfs_tolerance=qfs_tolerance, coordinate_tolerance=coordinate_tolerance, coordinate_scheme=coordinate_scheme)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
grid = ebdyc.generate_grid(bh/grid_upsample)
ebdyc.register_grid(grid, verbose=verbose)
ebdyc.ready_bump(MOL.bump, (grid.x_bounds[1]-ebdy.radial_width, grid.y_bounds[1]-ebdy.radial_width), ebdyc[0].radial_width)
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)

################################################################################
# Get solution, forces, BCs

if problem == 'easy':
	solution_func = lambda x, y: -np.cos(x)*np.exp(np.sin(x))*np.sin(y)
	force_func = lambda x, y: (2.0*np.cos(x)+3.0*np.cos(x)*np.sin(x)-np.cos(x)**3)*np.exp(np.sin(x))*np.sin(y)
else:
	k = 10*np.pi/3
	solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
	force_func = lambda x, y: k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)

bc = solution_func(ebdy.bdy.x, ebdy.bdy.y)

f = EmbeddedFunction(ebdyc)
f.define_via_function(force_func)
ua = EmbeddedFunction(ebdyc)
ua.define_via_function(solution_func)
bc = BoundaryFunction(ebdyc)
bc.define_via_function(solution_func)

################################################################################
# Setup Poisson Solver

# generate inhomogeneous solver and solve that problem
solver = PoissonSolver(ebdyc, solver_type=solver_type)
ue = solver(f, tol=solver_tol, verbose=verbose, maxiter=100, restart=20)

# this isn't correct yet because we haven't applied boundary conditions
A = Laplace_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(bdy.N)
bv = solver.get_boundary_values(ue)
tau = np.linalg.solve(A, np.concatenate((bc-bv).bdy_value_list))
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])
out = Laplace_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigma)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)
ue[0] += rslpl[0].reshape(ebdyc[0].radial_shape)
ue['grid'] += gslp

# compute the error
err = np.abs(ue - ua)
max_err = err.max()

# make plots if plotting
if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, ue.get_grid_value(masked=True))
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, (err+1e-16).get_grid_value(masked=True), norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error: {:0.2e}'.format(max_err))
