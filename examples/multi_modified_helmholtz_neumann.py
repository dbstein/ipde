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
from ipde.grid_evaluators.modified_helmholtz_grid_evaluator import ModifiedHelmholtzFreespaceGridEvaluator, ModifiedHelmholtzGridBackend
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
BoundaryCollection = pybie2d.boundaries.collection.BoundaryCollection

nb_base = 40
adj = 5
nb = nb_base * adj
M = max(4, min(3*adj, 20))
helmholtz_k = 5.0
verbose = True
plot = False
reparametrize = False
slepian_r = 1.5*M
evaluation_backend = 'grid'
solver_type = 'spectral'
coordinate_scheme = 'nufft'
tols = 1e-12
solver_tol = tols
coordinate_tol = tols
qfs_tolerance = tols

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy1 = GSB(c=star(12*nb, a=0.0, r=3, f=11))
bdy2 = GSB(c=squish(2*nb, x=-1.2, y=-0.7, b=0.05, rot=-np.pi/4))
bdy3 = GSB(c=star(3*nb, x=1, y=0.5, a=0.3, f=3))
if reparametrize:
	bdy1 = GSB(*arc_length_parameterize(bdy1.x, bdy1.y))
	bdy2 = GSB(*arc_length_parameterize(bdy2.x, bdy2.y))
	bdy3 = GSB(*arc_length_parameterize(bdy3.x, bdy3.y))
bh1 = bdy1.dt*bdy1.speed.min()
bh2 = bdy2.dt*bdy2.speed.min()
bh3 = bdy3.dt*bdy3.speed.min()
bh = min(bh1, bh2, bh3)
# construct embedded boundary
bdys = [bdy1, bdy2, bdy3]
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, heaviside=MOL.step) for bdy in bdys]
ebdyc = EmbeddedBoundaryCollection(ebdys)
# register the grid
print('\nRegistering the grid')
grid = ebdyc.generate_grid(force_square=True)

################################################################################
# Get solution, forces, BCs

k = 20*np.pi/7

print('Evaluating analytic solution')
solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_x = lambda x, y: k*np.cos(k*x)*np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_y = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)
solution_func_n = lambda x, y, nx, ny: solution_func_x(x, y)*nx + solution_func_y(x, y)*ny
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = EmbeddedFunction(ebdyc, function=force_func)
ua = EmbeddedFunction(ebdyc, function=solution_func)
all_nx = np.concatenate([ebdy.bdy.normal_x for ebdy in ebdys])
all_ny = np.concatenate([ebdy.bdy.normal_y for ebdy in ebdys])
bcs2v_x = BoundaryFunction(ebdyc, function=solution_func_x)
bcs2v_y = BoundaryFunction(ebdyc, function=solution_func_y)
bcs2v = bcs2v_x*all_nx + bcs2v_y*all_ny

################################################################################
# Setup Solver

if evaluation_backend == 'grid':
	MHGB = ModifiedHelmholtzGridBackend(grid.xh, 20, helmholtz_k)
	MHGE = ModifiedHelmholtzFreespaceGridEvaluator(MHGB, grid.xv, grid.yv)
	grid_backend = MHGE
else:
	grid_backend = evaluation_backend

import time

print('Creating inhomogeneous solver')
solver = ModifiedHelmholtzSolver(ebdyc, helmholtz_k, solver_type=solver_type, grid_backend=grid_backend)
print('Solving inhomogeneous problem')
ue = solver(f, tol=solver_tol, verbose=verbose)

print('Solving homogeneous problem')
# this isn't correct yet because we haven't applied boundary conditions
def two_d_split(MAT, hsplit, vsplit):
	vsplits = np.vsplit(MAT, vsplit)
	return [np.hsplit(vsplit, hsplit) for vsplit in vsplits]

Ns = [ebdy.bdy.N for ebdy in ebdyc.ebdys]
CNs = np.cumsum(Ns)
CN = CNs[:-1]
total_N = CNs[-1]
MAT = np.zeros([total_N, total_N], dtype=float)
MATS = two_d_split(MAT,CN,CN)

d_only  = lambda src, trg: MH_Layer_Form(src, trg, ifdipole=True, k=helmholtz_k)
c_only  = lambda src, trg: MH_Layer_Form(src, trg, ifcharge=True, k=helmholtz_k)
c_and_d = lambda src, trg: MH_Layer_Form(src, trg, ifcharge=True, ifdipole=True, k=helmholtz_k)
d_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
c_singular  = lambda src: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
cd_singular = lambda src: c_singular(src) + d_singular(src)
half_eye    = lambda src: np.eye(src.N)*0.5

s_singular = lambda src: -(d_singular(src)/src.weights).T*src.weights
s_only = lambda src, trg: -(d_only(trg, src)/trg.weights).T*src.weights

MATS[0][0][:] = s_singular(bdy1) - half_eye(bdy1)
MATS[0][1][:] = s_only(bdy2, bdy1)
MATS[0][2][:] = s_only(bdy3, bdy1)

MATS[1][0][:] = s_only(bdy1, bdy2)
MATS[1][1][:] = s_singular(bdy2) + half_eye(bdy2)
MATS[1][2][:] = s_only(bdy3, bdy2)

MATS[2][0][:] = s_only(bdy1, bdy3)
MATS[2][1][:] = s_only(bdy2, bdy3)
MATS[2][2][:] = s_singular(bdy3) + half_eye(bdy3)

# get the inhomogeneous solution on the boundary
bvs = solver.get_boundary_normal_derivatives(ue)

# solve for density
tau = -np.linalg.solve(MAT, bcs2v - bvs)

# separate this into pieces
taul = ebdyc.v2l(tau)

# get effective sources
qfs_options = {
	'tol'        : qfs_tolerance,
	'shift_type' : 'complex',
}
qfs_list = []
qfs_sources = BoundaryCollection()
for ebdy in ebdys:
	interior = ebdy is ebdys[0]
	qfs = Modified_Helmholtz_QFS(ebdy.bdy, interior, True, False, helmholtz_k, qfs_options)
	qfs_list.append(qfs)
	qfs_sources.add(qfs.source, 'i' if interior else 'e')
qfs_sources.amass_information()

# compute sigmas
sigmal = [qfs([tau]) for qfs, tau in zip(qfs_list, taul)]
sigmav = np.concatenate(sigmal)

print('Evaluating homogeneous solution')
# evaluate this onto all gridpoints and radial points
out = MH_Layer_Apply(qfs_sources, ebdyc.grid_and_radial_pts, charge=sigmav, k=helmholtz_k)
ue += out

if plot:
	fig, ax = plt.subplots()
	clf = ue.plot(ax, cmap=mpl.cm.jet)
	for ebdy in ebdyc:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black')
	plt.colorbar(clf)

# compute the error
err = np.abs(ue-ua)

if plot:
	fig, ax = plt.subplots()
	clf = (err+1e-16).plot(ax, norm=mpl.colors.LogNorm(), cmap=mpl.cm.jet)
	for ebdy in ebdyc:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black')
	plt.colorbar(clf)

print('\nMaximum Error: {:0.2e}'.format(err.max()))
