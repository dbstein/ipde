import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.modified_helmholtz import ModifiedHelmholtzSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

nb = 300
helmholtz_k = 5.0
M = 8
pad_zone = 0
verbose = False
plot = True
reparametrize = False
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy1 = GSB(c=star(4*nb, a=0.0, r=3, f=11))
bdy2 = GSB(c=squish(nb, x=-1.2, y=-0.7, b=0.05, rot=-np.pi/4))
bdy3 = GSB(c=star(nb, x=1, y=0.5, a=0.3, f=3))
if reparametrize:
	bdy1 = GSB(*arc_length_parameterize(bdy1.x, bdy1.y))
	bdy2 = GSB(*arc_length_parameterize(bdy2.x, bdy2.y))
	bdy3 = GSB(*arc_length_parameterize(bdy3.x, bdy3.y))
bh1 = bdy1.dt*bdy1.speed.min()
bh2 = bdy2.dt*bdy2.speed.min()
bh3 = bdy3.dt*bdy3.speed.min()
bh = min(bh1, bh2, bh3)
if False:
	fig, ax = plt.subplots()
	ax.scatter(bdy1.x, bdy1.y, color='black')
	ax.scatter(bdy2.x, bdy2.y, color='blue')
	ax.scatter(bdy3.x, bdy3.y, color='red')
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*7//bh)
# construct a grid
grid = Grid([-3.5, 3.5], ng, [-3.5, 3.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
bdys = [bdy1, bdy2, bdy3]
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, pad_zone, MOL.step) for bdy in bdys]
ebdyc = EmbeddedBoundaryCollection(ebdys)
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# make some plots
if False:
	fig, ax = plt.subplots()
	colors = ['black', 'blue', 'red', 'purple', 'purple']
	for ebdy in ebdys:
		q = ebdy.bdy_qfs
		q1 = q.interior_source_bdy if ebdy.interior else q.exterior_source_bdy
		q = ebdy.interface_qfs
		q2 = q.interior_source_bdy
		q3 = q.exterior_source_bdy
		bbs = [ebdy.bdy, ebdy.interface, q1, q2, q3]
		for bi, bb in enumerate(bbs):
			ax.plot(bb.x, bb.y, color=colors[bi])

################################################################################
# Get solution, forces, BCs

k = 20*np.pi/7

print('Evaluating analytic solution')
solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_x = lambda x, y: k*np.cos(k*x)*np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_y = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)
solution_func_n = lambda x, y, nx, ny: solution_func_x(x, y)*nx + solution_func_y(x, y)*ny
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(grid.xg, grid.yg)
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua = solution_func(grid.xg, grid.yg)
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
all_nx = np.concatenate([ebdy.bdy.normal_x for ebdy in ebdys])
all_ny = np.concatenate([ebdy.bdy.normal_y for ebdy in ebdys])
bcs2v = solution_func_n(ebdyc.all_bvx, ebdyc.all_bvy, all_nx, all_ny)
bcs2l = ebdyc.v2l(bcs2v)

################################################################################
# Setup Poisson Solver

print('Creating inhomogeneous solver')
solver = ModifiedHelmholtzSolver(ebdyc, solver_type=solver_type, k=helmholtz_k)
print('Solving inhomogeneous problem')
ue, uers = solver(f, frs, tol=1e-12, verbose=verbose)

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
bvs = solver.get_boundary_normal_derivatives(uers)

# solve for density
tau = -np.linalg.solve(MAT, bcs2v - bvs)

# separate this into pieces
taul = ebdyc.v2l(tau)

# get effective sources
qfs_list = []
Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, ifcharge=True, k=helmholtz_k)
for ebdy in ebdys:
	def Kernel_Function(src, trg):
		return src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
	qfs = QFS_Evaluator(ebdy.bdy_qfs, ebdy.interior, [Kernel_Function,], Naive_SLP, on_surface=True, form_b2c=False)
	qfs_list.append(qfs)

# compute sigmas
sigmal = [qfs([tau]) for qfs, tau in zip(qfs_list, taul)]
sigmav = np.concatenate(sigmal)

print('Evaluating homogeneous solution')
# evaluate this onto all gridpoints and radial points
out = MH_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigmav, k=helmholtz_k)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)

ue[ebdyc.phys] += gslp
for uer, rslp in zip(uers, rslpl):
	uer += rslp

if plot:
	print('Plotting solution')
	mue = np.ma.array(ue, mask=ebdyc.ext)
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mue)
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)

# compute the error
rerrs = [np.abs(uer - uar).max() for uer, uar in zip(uers, uars)]
gerr = np.abs(ue - ua)
gerrp = gerr[ebdyc.phys]
mgerr = np.ma.array(gerr, mask=ebdyc.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mgerr + 1e-15, norm=mpl.colors.LogNorm())
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)

print('\nError in grid:         {:0.2e}'.format(gerrp.max()))
for ri, rerr in enumerate(rerrs):
	print('Error in annulus', ri+1, 'is: {:0.2e}'.format(rerr))
