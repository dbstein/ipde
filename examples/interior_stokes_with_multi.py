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
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(2*src.N)
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)

ns =     [100,      200,      300,      400,      500,      600,      700,      800,      900,      1000     ]
errs_s = [2.81e-01, 9.03e-03, 2.88e-04, 8.17e-06, 2.48e-07, 8.25e-09, 2.16e-10, 1.22e-11, 1.36e-11, 1.17e-11 ]

nb = 100
M = min(30, 4*int(nb/100))
pad_zone = 0
verbose = False
plot = False
reparametrize = False
slepian_r = 2.0*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 6*int(0.25*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1.0, pad_zone, MOL.step)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (1.2-ebdy.radial_width, 1.2-ebdy.radial_width), ebdys[0].radial_width)

################################################################################
# Extract radial information from ebdy and construct annular solver

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
a = 7.0
b = 5.0
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
fu_rs = [fu_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
fv_rs = [fv_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua_rs = [u_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
va_rs = [v_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
pa_rs = [p_function(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
pa_rs = [pa_r - np.mean(pa_r) for pa_r in pa_rs]
upper_u = u_function(ebdy.bdy.x, ebdy.bdy.y)
upper_v = v_function(ebdy.bdy.x, ebdy.bdy.y)

# setup the solver
solver = StokesSolver(ebdyc, solver_type=solver_type)
uc, vc, pc, urs, vrs = solver(fu, fv, fu_rs, fv_rs, tol=1e-12, verbose=verbose)
ur = urs[0]
vr = vrs[0]

if plot:
	mue = np.ma.array(uc, mask=ebdy.ext)
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3, alpha=0.2)

# this isn't correct yet because we haven't applied boundary conditions
def Stokes_Pressure_Fix(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x
    Nxy = trg.normal_x[:,None]*src.normal_y
    Nyx = trg.normal_y[:,None]*src.normal_x
    Nyy = trg.normal_y[:,None]*src.normal_y
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
    return NN
def Fixed_SLP(src, trg):
    return Naive_SLP(src, trg) + Stokes_Pressure_Fix(src, trg)
A = Stokes_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(2*bdy.N) + Stokes_Pressure_Fix(bdy, bdy)
bu = solver.get_boundary_values(urs)
bv = solver.get_boundary_values(vrs)
buc = np.concatenate([bu, bv])
bc = np.concatenate([upper_u, upper_v])
tau = np.linalg.solve(A, bc-buc)

qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
sigma = qfs([tau,])
out = Stokes_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, ebdyc.grid_and_radial_pts, forces=sigma, out_type='stacked')
ugslp, urslpl = ebdyc.divide_grid_and_radial(out[0])
vgslp, vrslpl = ebdyc.divide_grid_and_radial(out[1])
urslp = urslpl[0]
vrslp = vrslpl[0]
ur += urslp.reshape(ur.shape)
vr += vrslp.reshape(ur.shape)
uc[ebdy.phys] += ugslp
vc[ebdy.phys] += vgslp

if plot:
	mue = np.ma.array(uc, mask=ebdy.ext)
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3, alpha=0.2)

# compute the error
u_rerr = np.abs(ur - ua_rs[0])
u_gerr = np.abs(uc - ua)
u_gerrp = u_gerr[ebdy.phys]
m_u_gerr = np.ma.array(u_gerr, mask=ebdy.ext)

v_rerr = np.abs(vr - va_rs[0])
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

errs = np.array([u_gerrp.max(), v_gerrp.max(), u_rerr.max(), v_rerr.max()])
print('Maximum error: {:0.2e}'.format(errs.max()))
