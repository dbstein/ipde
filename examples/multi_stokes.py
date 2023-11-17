import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.stokes import StokesSolver
from qfs.stokes_qfs import Stokes_QFS
from personal_utilities.arc_length_reparametrization import arc_length_parameterize

squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
BoundaryCollection = pybie2d.boundaries.collection.BoundaryCollection

nb = 500
M = 10
pad_zone = 0
verbose = True
plot = True
reparametrize = True
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy1 = GSB(c=star(4*nb, a=0.1, r=3, f=11))
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
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*7//bh)
# construct a grid
grid = Grid([-3.5, 3.5], ng, [-3.5, 3.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
bdys = [bdy1, bdy2, bdy3]
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, heaviside=MOL.step) for bdy in bdys]
ebdyc = EmbeddedBoundaryCollection(ebdys)
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (3.5-ebdys[0].radial_width, 3.5-ebdys[0].radial_width), ebdys[0].radial_width)

################################################################################
# Extract radial information from ebdy and construct annular solver

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
a = 8.0
b = 7.0
p_a = 3.0
p_b = 1.0
sin = np.sin
cos = np.cos
esin = lambda x: np.exp(sin(x))
psix = lambda x, y: esin(a*x)*cos(b*y)
psiy = lambda x, y: esin(a*x)*sin(b*y)
u_function  = lambda x, y: psix(x, y)
v_function  = lambda x, y: -a/b*cos(a*x)*psiy(x, y)
p_function  = lambda x, y: cos(p_a*x) + esin(p_b*y)
fu_function = lambda x, y: (a**2*(sin(a*x)-cos(a*x)**2) + b**2)*psix(x, y) - p_a*sin(p_a*x)
fv_function = lambda x, y: -a*b*cos(a*x)*psiy(x, y)*(1 + (a/b)**2*sin(a*x)*(3+sin(a*x))) + p_b*cos(p_b*y)*esin(p_b*y)
fu = EmbeddedFunction(ebdyc, function=fu_function)
fv = EmbeddedFunction(ebdyc, function=fv_function)
ua = EmbeddedFunction(ebdyc, function=u_function)
va = EmbeddedFunction(ebdyc, function=v_function)
pa = EmbeddedFunction(ebdyc, function=p_function)
# demean pa
area = EmbeddedFunction(ebdyc, function=lambda x, y: np.ones_like(x)).integrate()
pa_mean = pa.integrate() / area
pa -= pa_mean
# gather boundaries
bdys = BoundaryCollection()
for ebdy in ebdyc:
	bdys.add(ebdy.bdy, 'i' if ebdy.interior else 'e')
bdys.amass_information()
# get boundary conditions
bdy_u = u_function(bdys.x, bdys.y)
bdy_v = v_function(bdys.x, bdys.y)

# setup the solver
solver = StokesSolver(ebdyc, solver_type=solver_type)
uc, vc, pc = solver(fu, fv, tol=1e-12, verbose=verbose)

if plot:
	mue = uc.get_grid_value(masked=True)
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)

# this isn't correct yet because we haven't applied boundary conditions
def two_d_split(MAT, hsplit, vsplit):
	vsplits = np.vsplit(MAT, vsplit)
	return [np.hsplit(vsplit, hsplit) for vsplit in vsplits]

Ns = [ebdy.bdy.N for ebdy in ebdyc.ebdys]
CNs = np.cumsum(Ns)
CN = CNs[:-1]
total_N = CNs[-1]
Ns2 = 2*Ns
CNs2 = 2*CNs
CN2 = 2*CN
total_N2 = 2*total_N
MAT = np.zeros([total_N2, total_N2], dtype=float)
MATS = two_d_split(MAT,CN2,CN2)

def Stokes_Pressure_Fix(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x*src.weights
    Nxy = trg.normal_x[:,None]*src.normal_y*src.weights
    Nyx = trg.normal_y[:,None]*src.normal_x*src.weights
    Nyy = trg.normal_y[:,None]*src.normal_y*src.weights
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))/np.sum(src.speed*src.weights)
    return NN

d_only  = lambda src, trg: Stokes_Layer_Form(src, trg, ifdipole=True)
c_only  = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
c_and_d = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True, ifdipole=True)
d_singular  = lambda src: Stokes_Layer_Singular_Form(src, ifdipole=True)
c_singular  = lambda src: Stokes_Layer_Singular_Form(src, ifforce=True)
cd_singular = lambda src: Stokes_Layer_Singular_Form(src, ifforce=True, ifdipole=True)
half_eye    = lambda src: np.eye(2*src.N)*0.5

MATS[0][0][:] = d_singular(bdy1) - half_eye(bdy1) + Stokes_Pressure_Fix(bdy1, bdy1)
MATS[0][1][:] = c_and_d(bdy2, bdy1) + Stokes_Pressure_Fix(bdy2, bdy1)*0.0
MATS[0][2][:] = c_and_d(bdy3, bdy1) + Stokes_Pressure_Fix(bdy3, bdy1)*0.0

MATS[1][0][:] = d_only(bdy1, bdy2) + Stokes_Pressure_Fix(bdy1, bdy2)
MATS[1][1][:] = cd_singular(bdy2) + half_eye(bdy2) + Stokes_Pressure_Fix(bdy2, bdy2)*0.0
MATS[1][2][:] = c_and_d(bdy3, bdy2) + Stokes_Pressure_Fix(bdy3, bdy2)*0.0

MATS[2][0][:] = d_only(bdy1, bdy3) + Stokes_Pressure_Fix(bdy1, bdy3)
MATS[2][1][:] = c_and_d(bdy2, bdy3) + Stokes_Pressure_Fix(bdy2, bdy3)*0.0
MATS[2][2][:] = cd_singular(bdy3) + half_eye(bdy3) + Stokes_Pressure_Fix(bdy3, bdy3)*0.0

bu = solver.get_boundary_values(uc)
bv = solver.get_boundary_values(vc)
bu_adj = bdy_u - bu
bv_adj = bdy_v - bv
bu_adj = ebdyc.v2l(bu_adj)
bv_adj = ebdyc.v2l(bv_adj)
bc_adj = np.concatenate([np.concatenate([bu_a, bv_a]) for bu_a, bv_a in zip(bu_adj, bv_adj)])
tau = np.linalg.solve(MAT, bc_adj)
taul = ebdyc.v2l2(tau)

def v2f(x):
    return x.reshape(2, x.size//2)

# get effective sources
qfs_list = []
for ebdy in ebdys:
	qfs = Stokes_QFS(ebdy.bdy, ebdy.interior, not ebdy.interior, True)
	qfs_list.append(qfs)

sigmal = []
for qfs, tau in zip(qfs_list, taul):
	sigmal.append(qfs([tau,tau] if qfs.interior else [tau,]))
sigmav = np.column_stack([v2f(sigma) for sigma in sigmal])
# collect sources
sources = BoundaryCollection()
for qfs in qfs_list:
	sources.add(qfs.source, 'i' if qfs.interior else 'e')
sources.amass_information()

out = Stokes_Layer_Apply(sources, ebdyc.grid_and_radial_pts, forces=sigmav, out_type='stacked')
uc += out[0]
vc += out[1]

# gslpu, rslplu = ebdyc.divide_grid_and_radial(outu)
# gslpv, rslplv = ebdyc.divide_grid_and_radial(outv)

# uc[ebdyc.phys] += gslpu
# vc[ebdyc.phys] += gslpv
# for ur, rslp in zip(urs, rslplu):
# 	ur += rslp
# for vr, rslp in zip(vrs, rslplv):
# 	vr += rslp

if plot:
	mue = uc.get_grid_value(masked=True)
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mue)
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)

# compute the error
u_err = np.abs(uc - ua)
v_err = np.abs(vc - va)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, u_err.get_grid_value(masked=True) + 1e-15, norm=mpl.colors.LogNorm())
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, v_err.get_grid_value(masked=True) + 1e-15, norm=mpl.colors.LogNorm())
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error, u {:0.2e}'.format(u_err.max()))
print('Error, v {:0.2e}'.format(v_err.max()))
