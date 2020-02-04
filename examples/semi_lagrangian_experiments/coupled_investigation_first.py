import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import finufftpy
import time
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from fast_interp import interp1d
from pybie2d.point_set import PointSet
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 2.0
# set timestep
dt_lagrangian = 0.02
dt_eulerian =  0.002
# check that lagrangian is a simple multiple of eulerian
eulerian_multiple = dt_lagrangian / dt_eulerian
if np.abs(int(eulerian_multiple) - eulerian_multiple)/np.abs(eulerian_multiple) > 1e-14:
	raise Exception
eulerian_multiple = int(eulerian_multiple)
tsteps = max_time / dt_lagrangian
if np.abs(int(np.round(tsteps)) - tsteps)/np.abs(tsteps) > 1e-14:
	raise Exception
tsteps = int(np.round(tsteps))
# number of boundary points
nb = 200
# number of grid points
ngx = int(nb)
ngy = int(nb/2)
# number of chebyshev modes
M = 8
# padding zone
pad_zone = 0
# smoothness of rolloff functions
slepian_r = 2*M
# type of u differentiation
u_differentiation_type = 'defined'
# u_differentiation_type = 'spectral'
# u_differentiation_type = 'fourth'

# generate a velocity field
kk = 2*np.pi/3
const_factor = 1.0
diff_factor = 0.75

c0_function = lambda x, y:    np.exp(np.cos(kk*x))*np.sin(kk*y)

diff_u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)
diff_v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)
# for testing purposes
diff_ux_function = lambda x, y, t:  kk*np.cos(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)
diff_uy_function = lambda x, y, t: -kk*np.sin(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)
diff_vx_function = lambda x, y, t:  kk*np.sin(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)
diff_vy_function = lambda x, y, t: -kk*np.cos(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)
const_u_function = lambda x, y, t:  x*0.0 + 1.0
const_v_function = lambda x, y, t:  x*0.0
# for testing purposes
const_ux_function = lambda x, y, t:  x*0.0
const_uy_function = lambda x, y, t:  x*0.0
const_vx_function = lambda x, y, t:  x*0.0
const_vy_function = lambda x, y, t:  x*0.0

def combine_functions(df, cf):
	return lambda x, y, t: const_factor*cf(x, y, t) + diff_factor*df(x, y, t)
u_function = combine_functions(diff_u_function, const_u_function)
v_function = combine_functions(diff_v_function, const_v_function)
ux_function = combine_functions(diff_ux_function, const_ux_function)
vx_function = combine_functions(diff_vx_function, const_vx_function)
uy_function = combine_functions(diff_uy_function, const_uy_function)
vy_function = combine_functions(diff_vy_function, const_vy_function)

# gradient function
def gradient(f):
	fh = np.fft.fft2(f)
	fx = np.fft.ifft2(fh*ikx).real
	fy = np.fft.ifft2(fh*iky).real
	return fx, fy

if ngx == ngy:
	xmin = -1.0
	xmax =  2.0
	ymin = -1.5
	ymax = ymin + (xmax-xmin)
else:	
	xmin = -1.0
	xmax =  5.0
	ymin = -1.5
	ymax = ymin + (xmax-xmin)/2

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

# initial c field
c0e = c0_function(x, y)

# get heaviside function
MOL = SlepianMollifier(slepian_r)

# construct boundary and reparametrize
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.0, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh, pad_zone, MOL.step)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# get a grid
grid = Grid([xmin, xmax], ngx, [ymin, ymax], ngy, x_endpoints=[True, False], y_endpoints=[True, False])
ebdyc.register_grid(grid)

original_ebdy = ebdy

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
te = 0.0

dt = dt_lagrangian

x_tracers = []
y_tracers = []

bxe = bdy.x.copy()
bye = bdy.y.copy()
ce = c0e.copy()

dt = dt_lagrangian

errors = []
ts = []

while t < max_time-1e-10:

	for _ in range(eulerian_multiple):
		cxe, cye = gradient(ce)
		ue = u_function(x, y, te)
		ve = v_function(x, y, te)
		bue = u_function(bxe, bye, te)
		bve = v_function(bxe, bye, te)
		if te == 0:
			ce = ce - dt_eulerian*(ue*cxe + ve*cye)
			bxe = bxe + dt_eulerian*bue
			bye = bye + dt_eulerian*bve
		else:
			ce = ce - 0.5*dt_eulerian*(3*ue*cxe + 3*ve*cye - uoe*cxoe - voe*cyoe)
			bxe = bxe + 0.5*dt_eulerian*(3*bue-buoe)
			bye = bye + 0.5*dt_eulerian*(3*bve-bvoe)
		bxe, bye, new_te = arc_length_parameterize(bxe, bye, return_t=True)
		uoe = ue
		voe = ve
		coe  = ce
		cxoe = cxe
		cyoe = cye
		bue_interp = interp1d(0, 2*np.pi, bdy.dt, bue, p=True)
		bve_interp = interp1d(0, 2*np.pi, bdy.dt, bve, p=True)
		buoe = bue_interp(new_te)
		bvoe = bve_interp(new_te)
		te += dt_eulerian

	# get the velocity fields at this time
	u = EmbeddedFunction(ebdyc)
	u.define_via_function(lambda x, y: u_function(x, y, t))
	v = EmbeddedFunction(ebdyc)
	v.define_via_function(lambda x, y: v_function(x, y, t))
	# get the velocity fields on the boundary
	ub = ebdyc.interpolate_radial_to_boundary(u).bdy_value_list[0]
	vb = ebdyc.interpolate_radial_to_boundary(v).bdy_value_list[0]

	# step of the first order method to get things started

	bx = ebdy.bdy.x + dt*ub
	by = ebdy.bdy.y + dt*vb

	bx, by, new_t = arc_length_parameterize(bx, by, return_t=True)
	bu_interp = interp1d(0, 2*np.pi, bdy.dt, ub, p=True)
	bv_interp = interp1d(0, 2*np.pi, bdy.dt, vb, p=True)
	# old boundary velocity values have to be in the correct place
	ubo_new_parm = bu_interp(new_t)
	vbo_new_parm = bv_interp(new_t)
	ubo = ub.copy()
	vbo = vb.copy()

	x_tracers.append(bx)
	y_tracers.append(by)

	# take gradients of the velocity fields
	dx = lambda f: fd_x_4(f, grid.xh, periodic_fix=True)
	dy = lambda f: fd_y_4(f, grid.yh, periodic_fix=True)
	if u_differentiation_type == 'spectral':
		ux, uy = ebdyc.gradient2(u, xder, yder, cutoff=True)
		vx, vy = ebdyc.gradient2(v, xder, yder, cutoff=True)
	elif u_differentiation_type == 'fourth':
		ux, uy = ebdyc.gradient2(u, dx, dy, cutoff=False)
		vx, vy = ebdyc.gradient2(v, dx, dy, cutoff=False)
	else:
		ux = EmbeddedFunction(ebdyc)
		uy = EmbeddedFunction(ebdyc)
		vx = EmbeddedFunction(ebdyc)
		vy = EmbeddedFunction(ebdyc)
		ux.define_via_function(lambda x, y: ux_function(x, y, t))
		uy.define_via_function(lambda x, y: uy_function(x, y, t))
		vx.define_via_function(lambda x, y: vx_function(x, y, t))
		vy.define_via_function(lambda x, y: vy_function(x, y, t))

	# now generate a new ebdy based on the moved boundary
	new_bdy = GSB(x=bx, y=by)
	bh = new_bdy.dt*new_bdy.speed.min()
	# construct embedded boundary
	new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh, pad_zone, MOL.step)
	new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
	new_ebdyc.register_grid(grid)

	# let's get the points that need to be interpolated to
	gp = new_ebdyc.grid_phys
	ap = new_ebdyc.radial_pts

	aax = np.concatenate([gp.x, ap.x])
	aay = np.concatenate([gp.y, ap.y])
	aap = PointSet(x=aax, y=aay)
	AP_key  = ebdy.register_points(aap.x, aap.y)

	# now we need to interpolate onto things
	AEP = ebdy.registered_partitions[AP_key]

	# generate a holding ground for the new c
	c_new = EmbeddedFunction(new_ebdyc)
	c_new.zero()

	# get departure points
	xd_all = np.zeros(aap.N)
	yd_all = np.zeros(aap.N)

	# advect those in the annulus
	c1,  c2,  c3  = AEP.get_categories()
	c1n, c2n, c3n = AEP.get_category_Ns()
	# category 1 and 2
	c1_2 = np.logical_or(c1, c2)
	c1_2n = c1n + c2n
	uxh = ebdy.interpolate_to_points(ux, aap.x, aap.y)
	uyh = ebdy.interpolate_to_points(uy, aap.x, aap.y)
	vxh = ebdy.interpolate_to_points(vx, aap.x, aap.y)
	vyh = ebdy.interpolate_to_points(vy, aap.x, aap.y)
	uh = ebdy.interpolate_to_points(u, aap.x, aap.y)
	vh = ebdy.interpolate_to_points(v, aap.x, aap.y)
	SLM = np.zeros([c1_2n,] + [2,2], dtype=float)
	SLR = np.zeros([c1_2n,] + [2,], dtype=float)
	SLM[:,0,0] = 1 + dt*uxh[c1_2]
	SLM[:,0,1] = dt*uyh[c1_2]
	SLM[:,1,0] = dt*vxh[c1_2]
	SLM[:,1,1] = 1 + dt*vyh[c1_2]
	SLR[:,0] = dt*uh[c1_2]
	SLR[:,1] = dt*vh[c1_2]
	OUT = np.linalg.solve(SLM, SLR)
	xdt, ydt = OUT[:,0], OUT[:,1]
	xd, yd = aap.x[c1_2] - xdt, aap.y[c1_2] - ydt
	# seems we should now back check xd, yd to make sure they're actually in ebdy!
	test_key = ebdy.register_points(xd, yd)
	test_part = ebdy.registered_partitions[test_key]
	test1, test2, test3 = test_part.get_categories()
	# reclassify things that are in test3 as being in c3
	c3[c1_2] = test3
	# recount c3
	c3n = c3.sum()
	print(test3.sum())
	xd_all[c1_2] = xd
	yd_all[c1_2] = yd
	# category 3... this is the tricky one
	if c3n > 0:
		th = 2*np.pi/nb
		tk = np.fft.fftfreq(nb, th/(2*np.pi))
		def d1_der(f):
			return np.fft.ifft(np.fft.fft(f)*tk*1j).real
		interp = lambda f: interp1d(0, 2*np.pi, th, f, k=3, p=True)
		bx_interp  = interp(ebdy.bdy.x)
		by_interp  = interp(ebdy.bdy.y)
		bxs_interp = interp(d1_der(ebdy.bdy.x))
		bys_interp = interp(d1_der(ebdy.bdy.y))
		nx_interp  = interp(ebdy.bdy.normal_x)
		ny_interp  = interp(ebdy.bdy.normal_y)
		nxs_interp = interp(d1_der(ebdy.bdy.normal_x))
		nys_interp = interp(d1_der(ebdy.bdy.normal_y))
		urb = ebdy.interpolate_radial_to_boundary_normal_derivative(u.radial_value_list[0])
		vrb = ebdy.interpolate_radial_to_boundary_normal_derivative(v.radial_value_list[0])
		ub_interp   = interp(ub)
		vb_interp   = interp(vb)
		urb_interp  = interp(urb)
		vrb_interp  = interp(vrb)
		ubs_interp  = interp(d1_der(ub))
		vbs_interp  = interp(d1_der(vb))
		urbs_interp = interp(d1_der(urb))
		vrbs_interp = interp(d1_der(vrb))
		xo = aap.x[c3]
		yo = aap.y[c3]
		def objective(s, r):
			f = np.empty([s.size, 2])
			f[:,0] = bx_interp(s) + r*nx_interp(s) + dt*ub_interp(s) + dt*r*urb_interp(s) - xo
			f[:,1] = by_interp(s) + r*ny_interp(s) + dt*vb_interp(s) + dt*r*vrb_interp(s) - yo
			return f
		def Jac(s, r):
			J = np.empty([s.size, 2, 2])
			J[:,0,0] = bxs_interp(s) + r*nxs_interp(s) + dt*ubs_interp(s) + dt*r*urbs_interp(s)
			J[:,1,0] = bys_interp(s) + r*nys_interp(s) + dt*vbs_interp(s) + dt*r*vrbs_interp(s)
			J[:,0,1] = nx_interp(s) + dt*urb_interp(s)
			J[:,1,1] = ny_interp(s) + dt*vrb_interp(s)
			return J
		# take as guess inds our s, r
		# s = new_ebdy.radial_t.ravel()[c3[gp.N:]]
		# r = new_ebdy.radial_r.ravel()[c3[gp.N:]]
		# s = AEP.test_t
		# r = AEP.test_r
		s =   AEP.full_t[c3]
		r =   AEP.full_r[c3]
		# now solve for sd, rd
		res = objective(s, r)
		mres = np.hypot(res[:,0], res[:,1]).max()
		tol = 1e-12
		while mres > tol:
			J = Jac(s, r)
			d = np.linalg.solve(J, res)
			s -= d[:,0]
			r -= d[:,1]
			res = objective(s, r)
			mres = np.hypot(res[:,0], res[:,1]).max()
		# get the departure points
		xd = bx_interp(s) + nx_interp(s)*r
		yd = by_interp(s) + ny_interp(s)*r
		xd_all[c3] = xd
		yd_all[c3] = yd
	# now interpolate to c
	ch = ebdy.interpolate_to_points(c, xd_all, yd_all)
	# set the grid values
	c_new.grid_value[new_ebdyc.phys] = ch[:gp.N]
	# set the radial values
	c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
	# merge the grid and the radial values where they overlap
	new_ebdyc.merge_grids(c_new)
	# overwrite under grid under annulus by radial grid
	# _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)

	# visualize fc3n
	if False:
		fig, ax = plt.subplots()
		is_fc3 = EmbeddedFunction(new_ebdyc)
		is_fc3_grid = np.zeros(x.shape, dtype=bool)
		is_fc3_grid[new_ebdyc.phys] = fc3[:gp.N]
		is_fc3_radial = fc3[gp.N:].reshape(new_ebdy.radial_shape)
		is_fc3.load_data(is_fc3_grid, [is_fc3_radial,])
		clf = is_fc3.plot(ax)
		for bb, col in zip([new_ebdyc, ebdyc], ['red', 'pink', 'gray']):
			bbb = bb.ebdys[0].bdy
			ax.plot(bbb.x, bbb.y, color=col)
		ax.scatter(new_ebdyc.ebdys[0].radial_x, new_ebdyc.ebdys[0].radial_y, color='white')

	# now reset naming conventions
	old_ebdy = ebdy
	old_ebdyc = ebdyc
	ebdy = new_ebdy
	ebdyc = new_ebdyc
	c = c_new

	# assess error on underlying grid
	err = (np.abs(c.grid_value - ce)*ebdyc.phys).max()

	ts.append(t)
	errors.append(err)

	t += dt
	print('   t = {:0.3f}'.format(t), max_time, 'error: {:0.2e}'.format(err))


c_eulerian = ce

def plotto(u):
	fig, ax = plt.subplots()
	u.plot(ax)

################################################################################
# Evaluate

print('\n\nAssessing Errors:')
# assess error on underlying grid
err = c.grid_value - c_eulerian
err = np.zeros_like(x)
err = np.abs(c.grid_value - c_eulerian)*ebdyc.phys
err = np.ma.array(err, mask=new_ebdyc.ext)
merr = err.max()

print('On grid:            {:0.1e}'.format(merr))

# assess error in position
err_bx = np.abs(bxe - ebdy.bdy.x)
err_by = np.abs(bye - ebdy.bdy.y)
err_b = max(err_bx.max(), err_by.max())

print('Boundary motion is: {:0.1e}'.format(err_b))

# assess on radial grid (via NUFFT of eulerian solution)
cr = ebdy.interpolate_grid_to_radial(c_eulerian)
err_r = np.abs(cr - c.radial_value_list[0])
merr_r = err_r.max()

print('On radial grid:     {:0.1e}'.format(merr_r))

# plot the errors properly...
cerr = EmbeddedFunction(ebdyc)
cerr.load_data(err, [err_r,])

fig, ax = plt.subplots()
clf = (cerr+1e-15).plot(ax, norm=mpl.colors.LogNorm(), vmin=1e-6)
plt.colorbar(clf)
# for xtr, ytr in zip(x_tracers, y_tracers):
	# ax.plot(xtr, ytr, alpha=0.2)

fig, ax = plt.subplots()
ax.plot(ts, errors)
