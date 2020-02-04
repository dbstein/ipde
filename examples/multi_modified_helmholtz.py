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

import time

nb = 800
helmholtz_k = 5.0
M = 20
pad_zone = 0
verbose = True
plot = True
reparametrize = True
slepian_r = 1.5*M
solver_type = 'fourth' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy1 = GSB(c=star(4*nb, a=0.05, r=3, f=7))
bdy2 = GSB(c=squish(nb, x=-1.2, y=-0.7, b=0.05, rot=-np.pi/4))
bdy3 = GSB(c=star(nb, x=1, y=0.5, a=0.2, f=3))
if reparametrize:
	bdy1 = GSB(*arc_length_parameterize(bdy1.x, bdy1.y))
	bdy2 = GSB(*arc_length_parameterize(bdy2.x, bdy2.y))
	bdy3 = GSB(*arc_length_parameterize(bdy3.x, bdy3.y))
bh1 = bdy1.dt*bdy1.speed.min()
bh2 = bdy2.dt*bdy2.speed.min()
bh3 = bdy3.dt*bdy3.speed.min()
bh = min(bh1, bh2, bh3)
if plot:
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
kwa = {'pad_zone':pad_zone, 'heaviside':MOL.step, 'interpolation_scheme':'polyi'}
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, **kwa) for bdy in bdys]
ebdyc = EmbeddedBoundaryCollection(ebdys)
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)

# timing for grid registration!
for i in range(3):
	st = time.time()
	ebdys[i].register_grid(grid)
	print((time.time()-st)*1000)

# make some plots
if plot:
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

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(grid.xg, grid.yg)
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua = solution_func(grid.xg, grid.yg)
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
bcs2v = solution_func(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2l = ebdyc.v2l(bcs2v)

################################################################################
# Setup Poisson Solver

solver = ModifiedHelmholtzSolver(ebdyc, solver_type=solver_type, k=helmholtz_k)
ue, uers = solver(f, frs, tol=1e-12, verbose=verbose)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, ue)
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)
	ax.set(xticks=[], xticklabels=[])
	ax.set(yticks=[], yticklabels=[])

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

MATS[0][0][:] = d_singular(bdy1) - half_eye(bdy1)
MATS[0][1][:] = c_and_d(bdy2, bdy1)
MATS[0][2][:] = c_and_d(bdy3, bdy1)

MATS[1][0][:] = d_only(bdy1, bdy2)
MATS[1][1][:] = cd_singular(bdy2) + half_eye(bdy2)
MATS[1][2][:] = c_and_d(bdy3, bdy2)

MATS[2][0][:] = d_only(bdy1, bdy3)
MATS[2][1][:] = c_and_d(bdy2, bdy3)
MATS[2][2][:] = cd_singular(bdy3) + half_eye(bdy3)

# get the inhomogeneous solution on the boundary
bvs = solver.get_boundary_values(uers)

# solve for density
tau = np.linalg.solve(MAT, bcs2v - bvs)

# separate this into pieces
taul = ebdyc.v2l(tau)

# get effective sources
qfs_list = []
Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, ifcharge=True, k=helmholtz_k)
for ebdy in ebdys:
	if ebdy.interior:
		def Kernel_Function(src, trg):
			return src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) - 0.5*np.eye(src.N)
	else:
		def Kernel_Function(src, trg):
			return src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) + src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k) + 0.5*np.eye(src.N)
	qfs = QFS_Evaluator(ebdy.bdy_qfs, ebdy.interior, [Kernel_Function,], Naive_SLP, on_surface=True, form_b2c=False)
	qfs_list.append(qfs)

# compute sigmas
sigmal = [qfs([tau]) for qfs, tau in zip(qfs_list, taul)]
sigmav = np.concatenate(sigmal)

# evaluate this onto all gridpoints and radial points
out = MH_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigmav, k=helmholtz_k)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)

ue[ebdyc.phys] += gslp
for uer, rslp in zip(uers, rslpl):
	uer += rslp

if plot:
	mue = np.ma.array(ue, mask=ebdyc.ext)
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mue)
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)
	ax.set(xticks=[], xticklabels=[])
	ax.set(yticks=[], yticklabels=[])

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
	ax.set(xticks=[], xticklabels=[])
	ax.set(yticks=[], yticklabels=[])

print('Error in grid:         {:0.2e}'.format(gerrp.max()))
for ri, rerr in enumerate(rerrs):
	print('Error in annulus', ri+1, 'is: {:0.2e}'.format(rerr))
