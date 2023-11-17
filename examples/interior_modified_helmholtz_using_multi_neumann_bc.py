import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.modified_helmholtz import ModifiedHelmholtzSolver
from qfs.modified_helmholtz_qfs import Modified_Helmholtz_QFS
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

from ipde.grid_evaluators.modified_helmholtz_grid_evaluator import ModifiedHelmholtzFreespaceGridEvaluator, ModifiedHelmholtzGridBackend

nb_base = 100
adj = 3
nb = nb_base * adj
M = max(4, min(3*adj, 20))
helmholtz_k = 1.0
verbose = True
plot = False
reparametrize = False
slepian_r = 1.5*M
evaluation_backend = 'pybie2d'
solver_type = 'spectral'
coordinate_scheme = 'nufft'
tols = 1e-12
solver_tol = tols
coordinate_tol = tols
qfs_tolerance = tols

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone=0, heaviside=MOL.step)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# construct and register a confining grid
print('\nRegistering the grid')
grid = ebdyc.generate_grid(force_square=True)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_dx = lambda x, y: k*np.cos(k*x)*np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_dy = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = EmbeddedFunction(ebdyc, function=force_func)
ua = EmbeddedFunction(ebdyc, function=solution_func)
bcs2v_x = BoundaryFunction(ebdyc, function=solution_func_dx)
bcs2v_y = BoundaryFunction(ebdyc, function=solution_func_dy)
bcs2v = bcs2v_x*bdy.normal_x + bcs2v_y*bdy.normal_y

################################################################################
# Setup  Solver

if evaluation_backend == 'grid':
	MHGB = ModifiedHelmholtzGridBackend(grid.xh, 20, helmholtz_k)
	MHGE = ModifiedHelmholtzFreespaceGridEvaluator(MHGB, grid.xv, grid.yv)
	grid_backend = MHGE
else:
	grid_backend = evaluation_backend

import time

# construct inhomogeneous solver
solver = ModifiedHelmholtzSolver(ebdyc, helmholtz_k, solver_type=solver_type, grid_backend=grid_backend)
ue = solver(f, tol=solver_tol, verbose=verbose)

# this isn't correct yet because we haven't applied boundary conditions
d_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
half_eye    = lambda src: np.eye(src.N)*0.5
s_singular = lambda src: -(d_singular(src)/src.weights).T*src.weights
A = s_singular(bdy) - half_eye(bdy)
bvn = solver.get_boundary_normal_derivatives(ue)
tau = -np.linalg.solve(A, bcs2v-bvn)
# get QFS evaluator
qfs_options = {
	'tol'        : qfs_tolerance,
	'shift_type' : 'complex',
}
qfs = Modified_Helmholtz_QFS(bdy, True, True, False, helmholtz_k, qfs_options)
sigma = qfs([tau,])
out = MH_Layer_Apply(qfs.source, ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k)
ue += out

# get the error
err = np.abs(ue - ua)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, ue.get_grid_value(masked=True), cmap=mpl.cm.jet)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	fig.colorbar(clf)

	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, (err+1e-16).get_grid_value(masked=True), norm=mpl.colors.LogNorm(), cmap=mpl.cm.jet)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	fig.colorbar(clf)

print('Maximum Error:    {:0.2e}'.format(err.max()))

# k^2 = 1; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k1      = [9.26e-01, 1.09e-02, 1.20e-04, 2.28e-06, 4.48e-08, 7.80e-09, 9.82e-10, 1.96e-09, 7.33e-10, 1.35e-09]
# k^2 = 10; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k10     = [4.26e-01, 1.01e-02, 1.06e-04, 1.85e-06, 3.98e-08, 5.04e-09, 6.43e-10, 1.23e-09, 4.55e-10, 8.45e-10]
# k^2 = 100; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k100    = [1.95e-01, 1.01e-02, 1.15e-04, 1.74e-06, 3.59e-08, 4.46e-09, 5.90e-10, 9.96e-10, 3.89e-10, 6.62e-10]
# k^2 = 1000; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k1000   = [8.88e-01, 6.90e-02, 6.90e-04, 8.69e-06, 1.15e-07, 1.11e-08, 1.58e-09, 2.21e-09, 1.01e-09, 1.44e-09]
# k^2 = 10000; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k10000  = [1.84e+00, 7.21e-01, 4.50e-02, 3.09e-03, 9.81e-05, 2.65e-06, 9.29e-08, 1.82e-08, 3.95e-09, 4.10e-09]
# k^2 = 100000; nb = 100*adj; M = 3*adj; adj=1,2,3...
errors_k100000 = [np.nan,   2.00e+00, 7.96e-01, 2.41e-01, 4.07e-02, 7.80e-03, 1.46e-03, 6.24e-04, 3.05e-04, 1.50e-04]

