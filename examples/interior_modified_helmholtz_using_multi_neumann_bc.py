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

base_nb = 100
adjust = 6
nb = base_nb * adjust
M = min(max(4, 10*adjust), 50)
helmholtz_k = np.sqrt(100000)
pad_zone = 3
verbose = True
plot = True
reparametrize = False
slepian_r = 1.0*M
qfs_tolerance = 1e-14
qfs_fsuf = 8

solver_type = 'fourth' # fourth or spectral

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
tolerances = {
	'qfs'      : qfs_tolerance,
	'qfs_fsuf' : qfs_fsuf,
}
ebdy = EmbeddedBoundary(bdy, True, M, bh*0.5, pad_zone, MOL.step, tolerances=tolerances)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_dx = lambda x, y: k*np.cos(k*x)*np.exp(np.sin(k*x))*np.sin(k*y)
solution_func_dy = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
f = force_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua = solution_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
uar = uars[0]
bcs2v_x = solution_func_dx(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2v_y = solution_func_dy(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2v = bcs2v_x*bdy.normal_x + bcs2v_y*bdy.normal_y

################################################################################
# Setup Poisson Solver

# THE TWO IMPORTANT LINES
solver = ModifiedHelmholtzSolver(ebdyc, solver_type=solver_type, k=helmholtz_k)
ue, uers = solver(f, frs, tol=1e-15, verbose=verbose)

uer = uers[0]
mue = np.ma.array(ue, mask=ebdy.ext)

# this isn't correct yet because we haven't applied boundary conditions
d_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
half_eye    = lambda src: np.eye(src.N)*0.5
s_singular = lambda src: -(d_singular(src)/src.weights).T*src.weights
A = s_singular(bdy) - half_eye(bdy)
bvn = solver.get_boundary_normal_derivatives(uers)
tau = -np.linalg.solve(A, bcs2v-bvn)

Singular_SLP = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_SLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])

out = MH_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)
rslp = rslpl[0]
uer += rslp.reshape(uer.shape)
ue[ebdy.phys] += gslp

# compute the error
rerr = np.abs(uer - uar)
gerr = np.abs(ue - ua)
gerrp = gerr[ebdy.phys]
mgerr = np.ma.array(gerr, mask=ebdy.ext)

if plot:
	fig, ax = plt.subplots()
	clf = ax.pcolormesh(grid.xg, grid.yg, mgerr + 1e-15, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)

print('Error in grid:    {:0.2e}'.format(gerrp.max()))
print('Error in annulus: {:0.2e}'.format(rerr.max()))
print('Max error:        {:0.2e}'.format(max(gerrp.max(), rerr.max())))

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

