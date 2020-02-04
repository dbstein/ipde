import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.stokes import StokesSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
squish = pybie2d.misc.curve_descriptions.squished_circle
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply

nb = 2000
M = 20
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
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, pad_zone, MOL.step) for bdy in bdys]
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
fu = fu_function(grid.xg, grid.yg)
fv = fv_function(grid.xg, grid.yg)
ua = u_function(grid.xg, grid.yg)
va = v_function(grid.xg, grid.yg)
fu_rs = [fu_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
fv_rs = [fv_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua_rs = [u_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
va_rs = [v_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
pa_rs = [p_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
pa_rs = [pa_r - np.mean(pa_r) for pa_r in pa_rs]
upper_u = np.concatenate([u_function(ebdy.bdy.x, ebdy.bdy.y) for ebdy in ebdys])
upper_v = np.concatenate([v_function(ebdy.bdy.x, ebdy.bdy.y) for ebdy in ebdys])

# setup the solver
solver = StokesSolver(ebdyc, solver_type=solver_type)
uc, vc, pc, urs, vrs = solver(fu, fv, fu_rs, fv_rs, tol=1e-12, verbose=verbose)
ur = urs[0]
vr = vrs[0]

if plot:
	mue = np.ma.array(uc, mask=ebdyc.ext)
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

bu = solver.get_boundary_values(urs)
bv = solver.get_boundary_values(vrs)
bu_adj = upper_u - bu
bv_adj = upper_v - bv
bu_adj = ebdyc.v2l(bu_adj)
bv_adj = ebdyc.v2l(bv_adj)
bc_adj = np.concatenate([np.concatenate([bu_a, bv_a]) for bu_a, bv_a in zip(bu_adj, bv_adj)])
tau = np.linalg.solve(MAT, bc_adj)
taul = ebdyc.v2l2(tau)

def v2f(x):
    return x.reshape(2, x.size//2)

# get effective sources
qfs_list = []
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
Fixed_SLP = lambda src, trg: Naive_SLP(src, trg) + Stokes_Pressure_Fix(src, trg)
for ebdy in ebdys:
	if ebdy.interior:
		def Kernel_Function(src, trg):
			return Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(2*src.N)
	else:
		def Kernel_Function(src, trg):
			return Stokes_Layer_Singular_Form(src, ifdipole=True, ifforce=True) + 0.5*np.eye(2*src.N)
	print(ebdy.interior)
	qfs = QFS_Evaluator(ebdy.bdy_qfs, ebdy.interior, [Kernel_Function,], Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
	qfs_list.append(qfs)

sigmal = [qfs([tau,]) for qfs, tau in zip(qfs_list, taul)]
sigmav = np.column_stack([v2f(sigma) for sigma in sigmal])

out = Stokes_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, forces=sigmav)
outu, outv = v2f(out)
gslpu, rslplu = ebdyc.divide_grid_and_radial(outu)
gslpv, rslplv = ebdyc.divide_grid_and_radial(outv)

uc[ebdyc.phys] += gslpu
vc[ebdyc.phys] += gslpv
for ur, rslp in zip(urs, rslplu):
	ur += rslp
for vr, rslp in zip(vrs, rslplv):
	vr += rslp

if plot:
	mue = np.ma.array(uc, mask=ebdyc.ext)
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mue)
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=1, alpha=0.1)
	plt.colorbar(clf)

# compute the error
u_r_errs = [np.abs(uer-uar).max() for uer, uar in zip(urs, ua_rs)]
v_r_errs = [np.abs(ver-var).max() for ver, var in zip(vrs, va_rs)]
u_gerr = np.abs(uc - ua)
u_gerrp = u_gerr[ebdyc.phys]
m_u_gerr = np.ma.array(u_gerr, mask=ebdyc.ext)
v_gerr = np.abs(vc - va)
v_gerrp = v_gerr[ebdyc.phys]
m_v_gerr = np.ma.array(v_gerr, mask=ebdyc.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, m_u_gerr + 1e-15, norm=mpl.colors.LogNorm())
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, m_v_gerr + 1e-15, norm=mpl.colors.LogNorm())
	for ebdy in ebdys:
		ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
		ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error, u, in grid:    {:0.2e}'.format(u_gerrp.max()))
print('Error, v, in grid:    {:0.2e}'.format(v_gerrp.max()))
print('Error, u, in annulus: {:0.2e}'.format(max(u_r_errs)))
print('Error, v, in annulus: {:0.2e}'.format(max(v_r_errs)))
