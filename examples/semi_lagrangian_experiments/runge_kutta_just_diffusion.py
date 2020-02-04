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
dt =  0.1/2/2/2/2/2/2
tsteps = max_time / dt
if np.abs(int(np.round(tsteps)) - tsteps)/np.abs(tsteps) > 1e-14:
	raise Exception
tsteps = int(np.round(tsteps))
# number of boundary points
nb = 200
# number of grid points
ngx = int(nb/2)*4
ngy = int(nb/2)*4
# number of chebyshev modes
M = 16
bh_ratio = 0.25
# padding zone
pad_zone = 0
# smoothness of rolloff functions
slepian_r = 1.0*M

# stuff for the modified helmholtz equation
if nu > 0:
	solver_type = 'spectral'

	zero_zeta = nu*dt
	zero_helmholtz_k = np.sqrt(1.0/zero_zeta)
	half_eye    = lambda src: np.eye(src.N)*0.5
	d0_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=zero_helmholtz_k)
	s0_singular = lambda src: -(d0_singular(src)/src.weights).T*src.weights
	Singular_SLP0 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=zero_helmholtz_k)
	Naive_SLP0 = lambda src, trg: MH_Layer_Form(src, trg, k=zero_helmholtz_k, ifcharge=True)

	first_zeta = 0.5*nu*dt
	first_helmholtz_k = np.sqrt(1.0/first_zeta)
	d1_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=first_helmholtz_k)
	s1_singular = lambda src: -(d1_singular(src)/src.weights).T*src.weights
	Singular_SLP1 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=first_helmholtz_k)
	Naive_SLP1 = lambda src, trg: MH_Layer_Form(src, trg, k=first_helmholtz_k, ifcharge=True)

# generate a velocity field
kk = 2*np.pi/3

# c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)
c0_function = lambda x, y: np.cos(kk*x)*np.sin(kk*y)

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
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.1, f=5))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
tolerances = {
	'qfs'      : qfs_tolerance,
	'qfs_fsuf' : qfs_fsuf,
}
ebdy = EmbeddedBoundary(bdy, True, M, bh_ratio*bh, pad_zone, MOL.step, tolerances=tolerances)
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
xder = lambda f: fd_x_4(f, grid.xh)
yder = lambda f: fd_y_4(f, grid.yh)

# now timestep
c = c0.copy()
t = 0.0

x_tracers = []
y_tracers = []
ts = []

new_ebdyc = ebdyc

qfs0 = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP0,], Naive_SLP0, on_surface=True, form_b2c=False)
A0 = s0_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
A0_lu = sp.linalg.lu_factor(A0)
solver0 = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k= zero_helmholtz_k)

qfs1 = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP1,], Naive_SLP1, on_surface=True, form_b2c=False)
A1 = s1_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
A1_lu = sp.linalg.lu_factor(A1)
solver1 = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k= first_helmholtz_k)

while t < max_time-1e-10:

	# step of forward euler to get things started
	if t == 0:
		c = solver0(c/zero_zeta, tol=1e-12, verbose=True)
		bvn = solver0.get_boundary_normal_derivatives(c.radial_value_list)
		tau = sp.linalg.lu_solve(A0_lu, bvn)
		sigma = qfs0([tau,])
		out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=zero_helmholtz_k)
		gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
		rslp = rslpl[0]
		c.radial_value_list[0] += rslp.reshape(M, nb)
		c.grid_value[new_ebdyc.phys] += gslp

	else:
		# half step
		c2 = solver1(c/first_zeta, tol=1e-12, verbose=True)
		bvn = solver1.get_boundary_normal_derivatives(c2.radial_value_list)
		tau = sp.linalg.lu_solve(A1_lu, bvn)
		sigma = qfs1([tau,])
		out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=first_helmholtz_k)
		gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
		rslp = rslpl[0]
		c2.radial_value_list[0] += rslp.reshape(M, nb)
		c2.grid_value[new_ebdyc.phys] += gslp
		# get the final step
		c = 2*c2 - c

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
	### for nu = 0.1; gh = 4bh; radh = 0.25*gh
	dts                         = 0.1 / 2**np.arange(5)
	## BDF
	diffs_200_8_only_diffusion  = [5.01e-03, 1.03e-03, 2.24e-04, 2.13e-03, 1.71e-02, 1.19e-01]
	diffs_200_16_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.55e-05, 1.36e-05, 4.40e-05]
	diffs_300_12_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.53e-05, 1.44e-05, 1.20e-04]
	diffs_300_24_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.56e-05, 1.37e-05, 3.39e-06, 8.44e-07, 1.99e-06]
	diffs_400_16_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.55e-05, 1.36e-05, 3.35e-06, 3.86e-06]
	diffs_400_32_only_diffusion = [5.01e-03, 1.03e-03, 2.31e-04, 5.56e-05, 1.37e-05, 3.39e-06, 8.44e-07, 2.11e-07, ]
	## second order
	diffs_200_8_only_diffusion  = [2.77e-03, 5.93e-04, 3.24e-04, 2.64e-03]
	diffs_200_16_only_diffusion = [2.78e-03, 5.92e-04, 1.53e-04, 3.91e-05, 1.09e-05, 1.40e-04]

