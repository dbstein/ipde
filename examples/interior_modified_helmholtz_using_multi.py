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

ns =     [200,     300,     400,     500,     600,     700,      800,      900,      1000,     1100,     1200,     1300,     1400,     1500,     1600     ]
errs4  = [7.06e-2, 2.03e-2, 7.52e-3, 3.33e-3, 1.71e-3, 9.71e-4,  5.90e-4,  3.76e-4,  2.48e-4,  1.74e-04, 1.25e-04, 9.21e-05, 6.89e-05, 5.22e-05, 4.10e-05 ]
errs8  = [1.77e-3, 9.10e-5, 1.23e-5, 2.11e-6, 4.96e-7, 1.44e-7,  4.83e-8,  2.04e-8,  1.08e-8,  9.73e-09, 1.08e-08, 1.20e-08, 1.30e-08, 1.30e-08, 1.14e-08 ]
errs12 = [np.nan,  9.45e-5, 1.00e-6, 7.22e-8, 1.02e-8, 1.85e-9,  4.00e-10, 1.11e-10, 4.09e-11, 3.37e-11, 3.37e-11, 2.82e-11, 1.83e-11, 1.80e-11, 1.56e-11 ]
errs16 = [np.nan,  np.nan,  5.16e-6, 3.61e-8, 1.33e-9, 1.30e-10, 1.51e-11, 2.25e-12, 4.95e-13, 4.02e-13, 2.69e-13, 3.10e-13, 4.43e-13, 4.47e-13, 2.64e-13 ]

ns =     [100,     200,     300,     400,     500,     600,      700,      800,      900,      1000     ]
errs_s = [2.21e-1, 5.10e-4, 7.01e-6, 1.38e-7, 3.36e-9, 9.34e-11, 3.09e-12, 4.36e-13, 4.14e-13, 4.47e-13 ]

def tornberg_bdy(N, n=1, R=1, cl=[], cv=[], dl=[], dv=[], a=0, b=0):
	theta = np.linspace(0, 2*np.pi, N, endpoint=False)
	clen = len(cl)
	dlen = len(dl)
	out = np.zeros(N, dtype=complex)
	for i in range(clen):
		out += cv[i]*np.cos(cl[i]*theta)
	for i in range(dlen):
		out += dv[i]*np.sin(dl[i]*theta)
	out *= R*np.exp(1j*n*theta)
	out += (a + 1j*b)
	return out

nb = 200
# M = min(30, 4*int(nb/100))
M = 8
helmholtz_k = 2.0
pad_zone = 0
verbose = True
plot = False
reparametrize = False
slepian_r = 2.0*M

solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
# bdy = GSB(c=tornberg_bdy(nb, 1, 1, [0, -5], [1, 0.2], [-1], [0.2]))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(1.0*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone, MOL.step)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)

################################################################################
# Get solution, forces, BCs

k = 8*np.pi/3

solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y) - k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
# def solution_func_i(x, y, i):
# 	return np.exp(-np.sqrt(2**i))*(np.cos((2**i)*x) + np.cos((2**i)*y))
# def solution_func(x, y):
# 	out = np.zeros_like(x)
# 	for i in range(6):
# 		out += solution_func_i(x, y, i)
# 	return out
# def force_func(x, y):
# 	out = np.zeros_like(x)
# 	for i in range(6):
# 		out += 2**(2*i)*solution_func_i(x, y, i)
# 	return out + 2**2*solution_func(x, y)
f = force_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
frs = [force_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
ua = solution_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
uars = [solution_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
uar = uars[0]
bcs2v = solution_func(ebdyc.all_bvx, ebdyc.all_bvy)
bcs2l = ebdyc.v2l(bcs2v)

################################################################################
# Setup Poisson Solver

# THE TWO IMPORTANT LINES
solver = ModifiedHelmholtzSolver(ebdyc, solver_type=solver_type, k=helmholtz_k)
ue, uers = solver(f, frs, tol=1e-15, verbose=verbose)

uer = uers[0]
mue = np.ma.array(ue, mask=ebdy.ext)

if False:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

# this isn't correct yet because we haven't applied boundary conditions
A = bdy.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) - 0.5*np.eye(bdy.N)*helmholtz_k**2
bv = solver.get_boundary_values(uers)
tau = np.linalg.solve(A, bcs2v-bv)
Singular_DLP = lambda src, _: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k) - 0.5*np.eye(src.N)*helmholtz_k**2
Naive_SLP = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])

out = MH_Layer_Apply(ebdyc.bdy_inward_sources, ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k)
gslp, rslpl = ebdyc.divide_grid_and_radial(out)
rslp = rslpl[0]
uer += rslp.reshape(uer.shape)
ue[ebdy.phys] += gslp

if False:
	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

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


if False:
	mpl.rc('text', usetex=True)
	mpl.rcParams.update({'font.size': 16})
	ns = np.array(ns)
	colo = mpl.cm.rainbow(np.linspace(0,1,4))
	fig, ax = plt.subplots()
	ax.plot(ns, errs4,   color=colo[0], linewidth=3)
	ax.plot(ns, errs8,   color=colo[1], linewidth=3)
	ax.plot(ns, errs12,  color=colo[2], linewidth=3)
	ax.plot(ns, errs16,  color=colo[3], linewidth=3)
	ax.plot(ns[3:-1], (10**2.2/ns[3:-1])**4, color='gray', linewidth=2)
	ax.plot(ns[1:-5], (10**2.05/ns[1:-5])**8, color='gray', linewidth=2)
	ax.plot(ns[3:9], (10**2.15/ns[3:9])**12, color='gray', linewidth=2)
	ax.plot(ns[3:8], (10**2.2/ns[3:8])**16, color='gray', linewidth=2)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('Number of boundary points')
	ax.set_ylabel(r'Error, in $L^\infty(\Omega)$')
	ax.minorticks_off()
	ax.set(xticks=[200, 400, 800, 1600], xticklabels=['200', '400', '800', '1600'])
	ax.text(800, 4e-3, r'$\mathcal{O}(h^4)$', color='gray')
	ax.text(800, 4e-7, r'$\mathcal{O}(h^8)$', color='gray')
	ax.text(900, 4e-10, r'$\mathcal{O}(h^{12})$', color='gray')
	ax.text(500, 2e-11, r'$\mathcal{O}(h^{16})$', color='gray')
	fig.tight_layout()
	ax.set(xticks=[], xticklabels=[])
	ax.set(yticks=[], yticklabels=[])
	ax.minorticks_off()

	wrap = lambda f: np.pad(f, (0,1), mode='wrap')

	fig, ax = plt.subplots()
	ax.plot(wrap(bdy.x), wrap(bdy.y), color='black')
	ax.plot(wrap(ebdy.interface.x), wrap(ebdy.interface.y), color='gray')
	ax.scatter(ebdy.radial_x, ebdy.radial_y, color='blue', s=10)
	ax.set(xticks=[], xticklabels=[])
	ax.set(yticks=[], yticklabels=[])
	ax.minorticks_off()

	mpl.rc('text', usetex=True)
	mpl.rcParams.update({'font.size': 16})
	ns = np.array(ns)
	colo = mpl.cm.rainbow(np.linspace(0,1,4))
	fig, ax = plt.subplots()
	ax.plot(ns, errs_s,   color=colo[0], linewidth=3)
	ax.set_yscale('log')
	ax.set_xlabel('Number of boundary points')
	ax.set_ylabel(r'Error, in $L^\infty(\Omega)$')
	ax.minorticks_off()
	ax.set(xticks=[200, 400, 600, 800, 1000], xticklabels=['200', '400', '600', '800', '1000'])
	fig.tight_layout()
