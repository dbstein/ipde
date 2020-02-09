import os
import time
import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.single_boundary.interior.poisson import PoissonSolver
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

nb = 200
M = 4
adj = 3

nb *= adj
M *= adj
M = int(np.floor(M))
M = max(4, min(20, M))

verbose = True
reparametrize = False
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral

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

################################################################################
# Get solution, forces, BCs

solution_func = lambda x, y: -np.cos(x)*np.exp(np.sin(x))*np.sin(y)
force_func = lambda x, y: (2.0*np.cos(x)+3.0*np.cos(x)*np.sin(x)-np.cos(x)**3)*np.exp(np.sin(x))*np.sin(y)

################################################################################
# Setup Poisson Solver

st = time.time()
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone=0, heaviside=MOL.step)
print(time.time()-st)
ebdy.register_grid(grid, verbose=verbose)
print(time.time()-st)
solver = PoissonSolver(ebdy, MOL.bump, bump_loc=(1.2-ebdy.radial_width, 1.2-ebdy.radial_width), solver_type=solver_type)
print(time.time()-st)
time_setup = time.time() - st

f = force_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
fr = force_func(ebdy.radial_x, ebdy.radial_y)
ua = solution_func(ebdy.grid.xg, ebdy.grid.yg)*ebdy.phys
uar = solution_func(ebdy.radial_x, ebdy.radial_y)
bc = solution_func(ebdy.bdy.x, ebdy.bdy.y)

st = time.time()
ue, uer = solver(f, fr, tol=1e-12, verbose=verbose)
time_inhomogeneous_solve = time.time() - st

mue = np.ma.array(ue, mask=ebdy.ext)

st = time.time()
A = Laplace_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(bdy.N)
A_LU = sp.linalg.lu_factor(A)
time_homogeneous_form = time.time() - st

st = time.time()
bv = solver.get_bv(uer)
tau = sp.linalg.lu_solve(A_LU, bc-bv)
qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Naive_SLP, on_surface=True, form_b2c=False)
sigma = qfs([tau,])
rslp = Laplace_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.radp, charge=sigma)
gslp = Laplace_Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, solver.gridpa, charge=sigma)
uer += rslp.reshape(uer.shape)
ue[ebdy.phys] += gslp
time_homogeneous_correction = time.time() - st

# compute the error
rerr = np.abs(uer - uar)
gerr = np.abs(ue - ua)
gerrp = gerr[ebdy.phys]
mgerr = np.ma.array(gerr, mask=ebdy.ext)

print('Error, maximum:               {:0.4e}'.format(max(gerrp.max(), rerr.max())))
print('Time, setup:                  {:0.4f}'.format(time_setup*1000))
print('Time, inhomogeneous solve:    {:0.4f}'.format(time_inhomogeneous_solve*1000))
print('Time, homogeneous form:       {:0.4f}'.format(time_homogeneous_form*1000))
print('Time, homogeneous correction: {:0.4f}'.format(time_homogeneous_correction*1000))
print('Degrees of freedom:           {:0.0f}'.format(solver.radp.N + solver.gridpa.N))

if False:
	ns   = 200*np.arange(1, 21)
	uscale = 1.238
	# this is for reparm...
	errs_M1p5  = np.array([5.5635e-04, 7.2616e-05, 1.9321e-05, 2.5564e-07, 1.9425e-08, 1.0209e-09, 1.2751e-10, 2.3578e-11, 2.4486e-12, 2.2293e-13, 1.3101e-13, 2.5702e-14, 3.7081e-14, 3.5971e-14, 1.0042e-13, 1.1147e-13, 9.3620e-14, 4.2411e-14, 4.7296e-14, 9.9587e-14 ])/uscale
	gmres_M1p5 = np.array([31,         18,         16,         17,         16,         17,         17,         17,         17,         17,         17,         17,         17,         17,         16,         16,         15,         15,         15,         14         ])
	errs_M2    = np.array([5.5635e-04, 7.2616e-05, 9.6542e-07, 2.3782e-08, 8.2043e-10, 2.5122e-11, 1.3433e-12, 1.3078e-13, 7.1609e-14, 1.0364e-13, 7.4385e-14, 8.3267e-14, 4.8975e-14, 3.5971e-14, 1.0042e-13, 1.1147e-13, 9.3620e-14, 4.2411e-14, 4.7296e-14, 9.9587e-14 ])/uscale
	gmres_M2   = np.array([31,         18,         20,         20,         20,         20,         20,         20,         20,         20,         18,         18,         17,         17,         16,         16,         15,         15,         15,         14         ])
	errs_M3    = np.array([5.5635e-04, 7.3761e-06, 1.0056e-07, 1.1997e-09, 1.8841e-11, 2.5424e-13, 9.3370e-14, 6.6558e-14, 4.8045e-14, 1.0364e-13, 7.4385e-14, 8.3267e-14, 4.8975e-14, 3.5971e-14, 1.0042e-13, 1.1147e-13, 9.3620e-14, 4.2411e-14, 4.7296e-14, 9.9587e-14 ])/uscale
	gmres_M3   = np.array([31,         26,         30,         31,         31,         31,         29,         25,         23,         20,         18,         18,         17,         17,         16,         16,         15,         15,         15,         14         ])
	# for not reparmed
	_errs_M1p5  = np.array([1.7102e-04, 1.9008e-05, 4.6032e-06, 3.7857e-08, 5.7047e-09, 5.1529e-10, 1.0177e-10, 7.6548e-12, 3.5099e-12, 4.1889e-13, 3.6504e-13, 3.0553e-13, 2.8244e-13, 2.4847e-13, 2.6668e-13, 2.5580e-13, 2.4225e-13, 2.3270e-13, 1.9718e-13, 1.9051e-13 ])/uscale
	_gmres_M1p5 = np.array([16,         14,         14,         14,         14,         12,         12,         11,         11,         11,         11,         11,         11,         11,         11,         11,         10,         10,         10,         10         ])
	_errs_M2    = np.array([1.7102e-04, 1.9008e-05, 1.4241e-07, 3.1137e-09, 1.7697e-10, 1.0479e-11, 8.8130e-13, 4.2877e-13, 4.3010e-13, 3.8125e-13, 3.6504e-13, 3.1308e-13, 2.8622e-13, 2.4847e-13, 2.6668e-13, 2.5580e-13, 2.4225e-13, 2.3270e-13, 1.9718e-13, 1.9051e-13 ])/uscale
	_gmres_M2   = np.array([16,         14,         14,         13,         13,         12,         12,         12,         12,         12,         12,         12,         11,         11,         11,         11,         10,         10,         10,         10         ])
	_errs_M3    = np.array([1.7102e-04, 8.7080e-07, 3.8560e-09, 2.1663e-11, 8.4466e-13, 5.8087e-13, 6.0374e-13, 4.3054e-13, 4.3299e-13, 3.8125e-13, 3.6504e-13, 3.1308e-13, 2.8622e-13, 2.4847e-13, 2.6668e-13, 2.5580e-13, 2.4225e-13, 2.3270e-13, 1.9718e-13, 1.9051e-13 ])/uscale
	_gmres_M3   = np.array([16,         14,         14,         14,         14,         14,         14,         13,         13,         12,         12,         12,         11,         11,         11,         11,         10,         10,         10,         10         ])
	_errs_M4    = np.array([1.7102e-04, 1.3314e-07, 2.3136e-10, 9.3747e-13, 8.5176e-13, 5.8442e-13, 6.0374e-13, 4.3054e-13, 4.3299e-13, 3.8125e-13, 3.6504e-13, 3.1308e-13, 2.8622e-13, 2.4847e-13, 2.6668e-13, 2.5580e-13, 2.4225e-13, 2.3270e-13, 1.9718e-13, 1.9051e-13 ])/uscale
	_gmres_M4   = np.array([16,         16,         17,         17,         17,         14,         14,         13,         13,         12,         12,         12,         11,         11,         11,         11,         10,         10,         10,         10         ])
	# timings are for M2
	time_setup = [154.9306,   161.8423,   231.6511,   279.5694,   365.5703,   441.3369,   567.2545,   675.1618,   776.0653,   988.2708,   1086.0043,  1109.0827,  1292.7971,  1514.4458,  1566.3459,  1731.6649,  2112.1962,  1997.4332,  2350.7431,  2493.5691  ]
	time_solve = [54.4164,    61.3074,    130.9922,   165.1330,   278.5511,   328.9678,   463.7773,   569.0830,   677.9559,   847.0654,   891.0797,   1108.5746,  1211.3173,  1395.2336,  1724.3528,  1821.3904,  2059.0310,  2340.1973,  2725.8365,  3026.1867  ]
	time_hform = [3.7887,     4.4360,     8.2417,     17.3187,    27.6716,    59.0572,    101.1767,   111.8031,   116.6139,   160.0151,   158.3679,   191.6876,   212.4312,   273.3963,   310.6759,   316.4508,   333.5750,   423.6121,   439.6663,   491.7672   ]
	time_happ  = [9.2492,     16.8111,    18.9607,    45.0113,    118.3691,   132.4646,   216.0077,   270.1962,   388.2639,   487.4279,   533.3376,   623.7583,   746.6176,   875.3183,   1029.5680,  1295.3801,  1378.9132,  1613.9953,  1811.7838,  2182.0726  ]
	dof        = [2937,       10153,      23278,      41176,      64142,      93065,      126337,     164660,     209371,     257995,     308865,     362634,     420616,     484843,     551565,     622557,     700186,     779826,     866465,     954829     ]

	os.chdir('/Users/dstein/Documents/Writing/Spectrally Accurate Poisson/images/')

	mpl.rc('text', usetex=True)
	mpl.rcParams.update({'text.latex.preamble' : [r'\usepackage{amsmath}']})
	mpl.rcParams.update({'font.size': 18})

	bx = np.pad(bdy.x, (0,1), mode='wrap')
	by = np.pad(bdy.y, (0,1), mode='wrap')
	ix = np.pad(ebdy.interface.x, (0,1), mode='wrap')
	iy = np.pad(ebdy.interface.y, (0,1), mode='wrap')
	xbds = [bdy.x.min(), bdy.x.max()]
	ybds = [bdy.y.min(), bdy.y.max()]
	fig, ax = plt.subplots()
	ax.plot(bx, by, color='black')
	ax.plot(ix, iy, color='black', linestyle='--')
	ax.fill(bx, by, color='pink', zorder=-15, alpha=0.9, edgecolor='none')
	ax.fill(ix, iy, color='white', zorder=-10, edgecolor='none')
	ax.fill(ix, iy, color='blue', zorder=-5, alpha=0.4, edgecolor='none')
	ax.set(xlim=xbds, ylim=ybds)
	ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
	ax.text(0.9,0.9,r'$\mathcal{C}$')
	ax.text(0.51,0.51,r'$\mathcal{A}$')
	ax.text(0.0,0.0,r'$\Omega_0$')
	ax.text(-0.4,0.8,r'$\Gamma$')
	ax.text(-0.4,0.48,r'$\overline{\Gamma}$')
	ax.set_aspect('equal')
	fig.tight_layout()
	fig.savefig('domain_decomposition.pdf', format='pdf', bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.plot(ns, errs_M1p5, color='blue', linewidth=2, marker='^', label=r'$\zeta=1.5$')
	ax.plot(ns, errs_M2, color='black', linewidth=2, marker='o', label=r'$\zeta=2$')
	ax.plot(ns, errs_M3, color='red', linewidth=2, marker='d', label=r'$\zeta=3$')
	ax.set_yscale('log')
	ax.set_xlabel(r'$n_\text{boundary}$')
	ax.set_ylabel(r'$\|u\|_{L^\infty(\Omega)}$')
	ax.axhline(1e-12, color='gray', linestyle='--')
	fig.tight_layout()
	plt.legend()
	fig.savefig('poisson_refinement.pdf', format='pdf', bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.plot(ns, _errs_M1p5, color='blue', linewidth=2, marker='^', label=r'$\zeta=1.5$')
	ax.plot(ns, _errs_M2, color='black', linewidth=2, marker='o', label=r'$\zeta=2$')
	ax.plot(ns, _errs_M3, color='red', linewidth=2, marker='d', label=r'$\zeta=3$')
	ax.plot(ns, _errs_M4, color='orange', linewidth=2, marker='s', label=r'$\zeta=4$')
	ax.set_yscale('log')
	ax.set_xlabel(r'$n_\text{boundary}$')
	ax.set_ylabel(r'$\|u\|_{L^\infty(\Omega)}$')
	ax.axhline(1e-12, color='gray', linestyle='--')
	fig.tight_layout()
	plt.legend()
	fig.savefig('_poisson_refinement.pdf', format='pdf', bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.plot((dof + ns)/1000000, time_solve, color='black',  linewidth=2, marker='^', label='Inhomogeneous solve')
	ax.plot((dof + ns)/1000000, time_setup, color='blue',   linewidth=2, marker='o', label='Inhomogeneous setup')
	ax.plot((dof + ns)/1000000, time_hform, color='purple', linewidth=2, marker='d', label='Homogeneous setup')
	ax.plot((dof + ns)/1000000, time_happ,  color='red',    linewidth=2, marker='s', label='Homogeneous solve')
	ax.set_xlabel(r'$N_\text{dof}$ (millions)')
	ax.set_ylabel(r'Time (ms)')
	plt.legend()
	fig.tight_layout()
	fig.savefig('poisson_time.pdf', format='pdf', bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.plot(ns, gmres_M1p5, color='blue',   linewidth=2, marker='^', label=r'$\zeta=1.5$')
	ax.plot(ns, gmres_M2,   color='black', linewidth=2, marker='^', label=r'$\zeta=2$')
	ax.plot(ns, gmres_M3,   color='red',  linewidth=2, marker='^', label=r'$\zeta=3$')
	ax.set_xlabel(r'$n_\text{boundary}$')
	ax.set_ylabel(r'GMRES Iteration Count')
	plt.legend()
	fig.tight_layout()
	fig.savefig('poisson_gmres.pdf', format='pdf', bbox_inches='tight')

	fig, ax = plt.subplots()
	ax.pcolormesh(grid.xg, grid.yg, mue)
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)

	fig, ax = plt.subplots()
	clf = ax.imshow(mgerr.T[::-1]+1e-16, extent=grid.x_bounds+grid.y_bounds, vmin=1e-15, norm=mpl.colors.LogNorm())
	# clf = ax.pcolormesh(grid.xg, grid.yg, mgerr+1e-16, vmin=1e-11, norm=mpl.colors.LogNorm())
	ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
	ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
	plt.colorbar(clf)
	ax.set_aspect('equal')
	ax.set(xticks=[], yticks=[], xticklabels=[], yticklabels=[])
	ax.set(xlim=[-1.1,1.3], ylim=[-1.2,1.2])
	fig.tight_layout()
	fig.savefig('poisson_error.pdf', format='pdf', bbox_inches='tight')
