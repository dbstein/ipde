import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import finufftpy
import time
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.modified_helmholtz import ModifiedHelmholtzSolver
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from fast_interp import interp1d
from pybie2d.point_set import PointSet
from qfs.two_d_qfs import QFS_Evaluator
import scipy as sp
import scipy.linalg
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 0.5
reaction_value = 0.0  # value for reaction term
const_factor   = 0.0  # constant advection velocity
diff_factor    = 0.0  # non-constant advection velocity
nu             = 0.1 # diffusion coefficient
nonlinearity = lambda x: x*np.cos(x)
scaled_nonlinearity = lambda x: reaction_value*nonlinearity(x)

qfs_tolerance = 1e-14
qfs_fsuf = 4

# set timestep
dt =  0.1
tsteps = max_time / dt
if np.abs(int(np.round(tsteps)) - tsteps)/np.abs(tsteps) > 1e-14:
	raise Exception
tsteps = int(np.round(tsteps))
# number of boundary points
nb = 400
# number of grid points
ngx = int(nb/2)*4
ngy = int(nb/2)*4
# number of chebyshev modes
M = 32
# padding zone
pad_zone = 0
# smoothness of rolloff functions
slepian_r = 1.0*M

# stuff for the modified helmholtz equation
if nu > 0:
	first_zeta = nu*dt
	first_helmholtz_k = np.sqrt(1.0/first_zeta)
	second_zeta = (2/3)*nu*dt
	second_helmholtz_k = np.sqrt(1.0/second_zeta)
	solver_type = 'spectral'
	d1_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=first_helmholtz_k)
	half_eye    = lambda src: np.eye(src.N)*0.5
	s1_singular = lambda src: -(d1_singular(src)/src.weights).T*src.weights
	d2_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=second_helmholtz_k)
	s2_singular = lambda src: -(d2_singular(src)/src.weights).T*src.weights
	Singular_SLP1 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=first_helmholtz_k)
	Naive_SLP1 = lambda src, trg: MH_Layer_Form(src, trg, k=first_helmholtz_k, ifcharge=True)
	Singular_SLP2 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=second_helmholtz_k)
	Naive_SLP2 = lambda src, trg: MH_Layer_Form(src, trg, k=second_helmholtz_k, ifcharge=True)

# generate a velocity field
kk = 2*np.pi/3

c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)

xmin = -1.0
xmax =  2.0
ymin = -1.5
ymax =  1.5

################################################################################
# Get truth via just using Adams-Bashforth 2 on periodic domain

# generate a grid
xv,  h = np.linspace(xmin, xmax, ngx, endpoint=False, retstep=True)
yv, _h = np.linspace(ymin, ymax, ngy, endpoint=False, retstep=True)
if _h != h:
	raise Exception('Need grid to be isotropic')
del _h
x, y = np.meshgrid(xv, yv, indexing='ij')
# fourier modes
kvx = np.fft.fftfreq(ngx, h/(2*np.pi))
kvy = np.fft.fftfreq(ngy, h/(2*np.pi))
kvx[int(ngx/2)] = 0.0
kvy[int(ngy/2)] = 0.0
kx, ky = np.meshgrid(kvx, kvy, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# get heaviside function
MOL = SlepianMollifier(slepian_r)

# construct boundary and reparametrize
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.0, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
tolerances = {
	'qfs'      : qfs_tolerance,
	'qfs_fsuf' : qfs_fsuf,
}
ebdy = EmbeddedBoundary(bdy, True, M, 0.25*bh, pad_zone, MOL.step, tolerances=tolerances)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# get a grid
grid = Grid([xmin, xmax], ngx, [ymin, ymax], ngy, x_endpoints=[True, False], y_endpoints=[True, False])
ebdyc.register_grid(grid)

# initial c field
c0 = EmbeddedFunction(ebdyc)
c0.define_via_function(c0_function)

def xder(f):
	return np.fft.ifft2(np.fft.fft2(f)*ikx).real
def yder(f):
	return np.fft.ifft2(np.fft.fft2(f)*iky).real

# now timestep
c = c0.copy()
t = 0.0

x_tracers = []
y_tracers = []
ts = []

new_ebdyc = ebdyc

qfs1 = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP1,], Naive_SLP1, on_surface=True, form_b2c=False)
qfs2 = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP2,], Naive_SLP2, on_surface=True, form_b2c=False)
A1 = s1_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
A2 = s2_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
A1_lu = sp.linalg.lu_factor(A1)
A2_lu = sp.linalg.lu_factor(A2)

while t < max_time-1e-10:

	# step of the first order method to get things started
	if t == 0:
		# generate a holding ground for the new c
		c_new = c.copy()

		# now solve diffusion equation
		solver1 = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=first_helmholtz_k)
		c_new_update = solver1(c_new/first_zeta, tol=1e-12, verbose=True)
		bvn = solver1.get_boundary_normal_derivatives(c_new_update.radial_value_list)
		tau = sp.linalg.lu_solve(A1_lu, bvn)
		sigma = qfs1([tau,])
		out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=first_helmholtz_k)
		gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
		rslp = rslpl[0]
		c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
		c_new_update.grid_value[new_ebdyc.phys] += gslp
		c_new = c_new_update

	else:  # the second order method
		c_new = (4/3)*c - (1/3)*c_old

		# now solve diffusion equation
		solver2 = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=second_helmholtz_k)
		c_new_update = solver2(c_new/second_zeta, tol=1e-12, verbose=True)
		bvn = solver2.get_boundary_normal_derivatives(c_new_update.radial_value_list)
		tau = sp.linalg.lu_solve(A2_lu, bvn)
		sigma = qfs2([tau,])
		out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=second_helmholtz_k)
		gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
		rslp = rslpl[0]
		c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
		c_new_update.grid_value[new_ebdyc.phys] += gslp
		c_new = c_new_update

	c_old = c
	c = c_new

	ts.append(t)
	t += dt
	print('   t = {:0.3f}'.format(t), 'of', max_time)

################################################################################
# Evaluate

fig, ax = plt.subplots()
clf = c.plot(ax)
plt.colorbar(clf)

try:
	fig, ax = plt.subplots()
	clf = c_save.plot(ax)
	plt.colorbar(clf)

	c_diff = c_save - c
	c_adiff = np.abs(c_diff)
	fig, ax = plt.subplots()
	clf = c_diff.plot(ax)
	plt.colorbar(clf)
	print('Difference is: {:0.2e}'.format(c_adiff.max()))
	c_save = c.copy()
except:
	pass
	c_save = c.copy()

if False:
	# for nu = 0.1; gh = 2bh; radh = 0.5*gh
	dts                         = 0.1 / 2**np.arange(5)
	diffs_200_8_only_diffusion  = [4.99e-03, 9.93e-04, 8.11e-03, 5.55e-02, 3.07e-01, 1.67e+00]
	# for nu = 0.1; gh = 4bh; radh = 0.25*gh
	dts                         = 0.1 / 2**np.arange(5)
	diffs_200_8_only_diffusion  = [5.01e-03, 1.03e-03, 2.24e-04, 2.13e-03, 1.71e-02, 1.19e-01]
	diffs_200_16_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.55e-05, 1.36e-05, 4.40e-05]
	diffs_300_12_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.53e-05, 1.44e-05, 1.20e-04]
	diffs_300_24_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.56e-05, 1.37e-05, 3.39e-06, 8.44e-07, 1.99e-06]
	diffs_400_16_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.55e-05, 1.36e-05, 3.35e-06, 3.86e-06]
	diffs_400_32_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.56e-05, 1.37e-05, 3.39e-06, 8.44e-07, 2.11e-07, 5.46e-08]
	# for nu = 1.0
	dts                         = 0.1 / 2**np.arange(5)
	diffs_200_8_only_diffusion  = [1.35e-02, 2.13e-03, 5.19e-04, 8.14e-05, 5.93e-04, 5.23e-03]
	diffs_300_12_only_diffusion = [1.35e-02, 2.06e-03, 4.66e-04, 1.12e-04, 2.58e-05, 1.50e-06, 3.04e-05]
	diffs_400_16_only_diffusion = [1.34e-02, 2.05e-03, 4.64e-04, 1.11e-04, 2.74e-05, 6.67e-06, 1.38e-06]


