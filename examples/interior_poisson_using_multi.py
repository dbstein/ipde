import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.poisson import PoissonSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Singular_DLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(src.N)
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)

nb = 600
M =  20
M = min(30, M)
pad_zone = 0
verbose = True
plot = True
reparametrize = False
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral
coordinate_scheme = 'nufft'
coordinate_tolerance = 1e-14
qfs_tolerance = 1e-14

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
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone=pad_zone, heaviside=MOL.step, qfs_tolerance=qfs_tolerance, coordinate_tolerance=coordinate_tolerance, coordinate_scheme=coordinate_scheme)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (1.2-ebdy.radial_width, 1.2-ebdy.radial_width), ebdyc[0].radial_width)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: -np.cos(x)*np.exp(np.sin(x))*np.sin(y)
force_func = lambda x, y: (2.0*np.cos(x)+3.0*np.cos(x)*np.sin(x)-np.cos(x)**3)*np.exp(np.sin(x))*np.sin(y)

# solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
# force_func = lambda x, y: k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(ebdyc.grid.xg, ebdyc.grid.yg)*ebdyc.phys
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdyc]
ua = solution_func(ebdyc.grid.xg, ebdyc.grid.yg)*ebdyc.phys
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdyc]
uar = uars[0]
bcs2v = solution_func(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2l = ebdyc.v2l(bcs2v)

################################################################################
# Setup Poisson Solver

solver = PoissonSolver(ebdyc, solver_type=solver_type)
ue, uers = solver(f, frs, tol=1e-14, verbose=verbose)

uer = uers[0]
mue = np.ma.array(ue, mask=ebdyc.ext)

if plot:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# this isn't correct yet because we haven't applied boundary conditions
A = Laplace_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(bdy.N)
bv = solver.get_boundary_values(uers)
tau = np.linalg.solve(A, bcs2v-bv)
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])
out = Laplace_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigma)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)
rslp = rslpl[0]
uer += rslp.reshape(uer.shape)
ue[ebdyc.phys] += gslp

if plot:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# compute the error
rerr = np.abs(uer - uar)
gerr = np.abs(ue - ua)
gerrp = gerr[ebdyc.phys]
mgerr = np.ma.array(gerr, mask=ebdyc.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mgerr + 1e-15, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error in grid:    {:0.2e}'.format(gerrp.max()))
print('Error in annulus: {:0.2e}'.format(rerr.max()))
