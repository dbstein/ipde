import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.single_boundary.interior.stokes import StokesSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(2*src.N)
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)

nb = 300
M = 8
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 1.5*M
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
ebdy.register_grid(grid)

################################################################################
# Extract radial information from ebdy and construct annular solver

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
a = 7.0
b = 8.0
p_a = 2.0
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
fu_radial = fu_function(ebdy.radial_x, ebdy.radial_y)
fv_radial = fv_function(ebdy.radial_x, ebdy.radial_y)
ua_radial =  u_function(ebdy.radial_x, ebdy.radial_y)
va_radial =  v_function(ebdy.radial_x, ebdy.radial_y)
pa_radial =  p_function(ebdy.radial_x, ebdy.radial_y)
pa_radial -= np.mean(pa_radial)
lower_u = u_function(ebdy.interface.x, ebdy.interface.y)
lower_v = v_function(ebdy.interface.x, ebdy.interface.y)
upper_u = u_function(ebdy.bdy.x, ebdy.bdy.y)
upper_v = v_function(ebdy.bdy.x, ebdy.bdy.y)

# setup the solver
try:
	ASS = solver.annular_solver
except:
	ASS = None
solver = StokesSolver(ebdy, MOL.bump, bump_loc=(1.2-ebdy.radial_width, 1.2-ebdy.radial_width), ASS=ASS)
uc, vc, pc, ur, vr, pr = solver(fu, fv, fu_radial, fv_radial, tol=1e-12, verbose=verbose)

if plot:
	mue = np.ma.array(uc, mask=ebdy.ext)
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3, alpha=0.2)

# this isn't correct yet because we haven't applied boundary conditions
def Stokes_Pressure_Fix(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x*src.weights
    Nxy = trg.normal_x[:,None]*src.normal_y*src.weights
    Nyx = trg.normal_y[:,None]*src.normal_x*src.weights
    Nyy = trg.normal_y[:,None]*src.normal_y*src.weights
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))/np.sum(src.speed*src.weights)
    return NN
def Fixed_SLP(src, trg):
    return Naive_SLP(src, trg) + Stokes_Pressure_Fix(src, trg)
A = Stokes_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(2*bdy.N) + Stokes_Pressure_Fix(bdy, bdy)
bu, bv = solver.get_bv(ur, vr)
buc = np.concatenate([bu, bv])
bc = np.concatenate([upper_u, upper_v])
tau = np.linalg.solve(A, bc-buc)

qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
sigma = qfs([tau,])
rslp = Stokes_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.radp, forces=sigma, out_type='stacked')
gslp = Stokes_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.gridpa, forces=sigma, out_type='stacked')
ur += rslp[0].reshape(ur.shape)
vr += rslp[1].reshape(ur.shape)
uc[ebdy.phys] += gslp[0]
vc[ebdy.phys] += gslp[1]

if plot:
	mue = np.ma.array(uc, mask=ebdy.ext)
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3, alpha=0.2)

# compute the error
u_rerr = np.abs(ur - ua_radial)
u_gerr = np.abs(uc - ua)
u_gerrp = u_gerr[ebdy.phys]
m_u_gerr = np.ma.array(u_gerr, mask=ebdy.ext)

v_rerr = np.abs(vr - va_radial)
v_gerr = np.abs(vc - va)
v_gerrp = v_gerr[ebdy.phys]
m_v_gerr = np.ma.array(v_gerr, mask=ebdy.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, m_u_gerr + 1e-15, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, m_v_gerr + 1e-15, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error, u, in grid:    {:0.2e}'.format(u_gerrp.max()))
print('Error, v, in grid:    {:0.2e}'.format(v_gerrp.max()))
print('Error, u, in annulus: {:0.2e}'.format(u_rerr.max()))
print('Error, v, in annulus: {:0.2e}'.format(v_rerr.max()))