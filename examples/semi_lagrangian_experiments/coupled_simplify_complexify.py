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
max_time = 1.0
reaction_value = 0.5  # value for reaction term
const_factor   = 0.5  # constant advection velocity
diff_factor    = 0.7  # non-constant advection velocity
implicit_term  = 1.0  # value for implicit term to be solved for
nonlinearity = lambda x: x*np.cos(x)
scaled_nonlinearity = lambda x: reaction_value*nonlinearity(x)
# set timestep
dt_lagrangian =  0.1/2/2/2/2/2
dt_eulerian =  0.001/2/2/2/2/2
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
ngy = int(nb)
# number of chebyshev modes
M = 8
# padding zone
pad_zone = 0
# smoothness of rolloff functions
slepian_r = 1.5*M
# type of u differentiation
u_differentiation_type = 'defined'
# u_differentiation_type = 'spectral'
# u_differentiation_type = 'fourth'

# generate a velocity field
kk = 2*np.pi/3

c0_function = lambda x, y:    np.exp(np.cos(kk*x))*np.sin(kk*y)

diff_u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)*(-1)
diff_v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)*(-1)
# for testing purposes
diff_ux_function = lambda x, y, t:  kk*np.cos(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)*(-1)
diff_uy_function = lambda x, y, t: -kk*np.sin(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)*(-1)
diff_vx_function = lambda x, y, t:  kk*np.sin(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)*(-1)
diff_vy_function = lambda x, y, t: -kk*np.cos(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)*(-1)
const_u_function = lambda x, y, t:  x*0.0 + 1.0*(-1)
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

# xmin = -1.0
# xmax =  2.0
xmin = -2.0
xmax =  1.0
ymin = -1.5
ymax = ymin + (xmax-xmin)

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

max_time += 1*dt
while t < max_time-1e-10:

	for _ in range(eulerian_multiple):
		cxe, cye = gradient(ce)
		ue = u_function(x, y, te)
		ve = v_function(x, y, te)
		bue = u_function(bxe, bye, te)
		bve = v_function(bxe, bye, te)
		if te == 0:
			ce_new = ce - dt_eulerian*(ue*cxe + ve*cye + scaled_nonlinearity(ce))
			bxe = bxe + dt_eulerian*bue
			bye = bye + dt_eulerian*bve
		else:
			ce_new = (1/3)*(4*ce-coe) - (2/3)*dt_eulerian*(2*ue*cxe + 2*ve*cye - uoe*cxoe - voe*cyoe + 2*scaled_nonlinearity(ce) - scaled_nonlinearity(coe))
			bxe = bxe + 0.5*dt_eulerian*(3*bue-buoe)
			bye = bye + 0.5*dt_eulerian*(3*bve-bvoe)
		bxe, bye, new_te = arc_length_parameterize(bxe, bye, return_t=True)
		uoe = ue
		voe = ve
		coe = ce
		cxoe = cxe
		cyoe = cye
		ce = ce_new
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
	if t == 0:

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
		xd_all[c1_2] = xd
		yd_all[c1_2] = yd
		# categroy 3... this is the tricky one
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
		ch = ebdy.interpolate_to_points(c - dt*scaled_nonlinearity(c), xd_all, yd_all)
		# set the grid values
		c_new.grid_value[new_ebdyc.phys] = ch[:gp.N]
		# set the radial values
		c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
		# overwrite under grid under annulus by radial grid
		# _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
		new_ebdy.full_merge(c_new.grid_value, c_new.radial_value_list[0], 5)

	else:  # the second order method

		bx = ebdy.bdy.x + 0.5*dt*(3*ub - ubo_new_parm)
		by = ebdy.bdy.y + 0.5*dt*(3*vb - vbo_new_parm)
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
		OAP_key = ebdy_old.register_points(aap.x, aap.y)

		# now we need to interpolate onto things
		AEP = ebdy.registered_partitions[AP_key]
		OAEP = ebdy_old.registered_partitions[OAP_key]

		# generate a holding ground for the new c
		c_new = EmbeddedFunction(new_ebdyc)
		c_new.zero()

		# get departure points
		xd_all = np.zeros(aap.N)
		yd_all = np.zeros(aap.N)
		xD_all = np.zeros(aap.N)
		yD_all = np.zeros(aap.N)

		# advect those in the annulus
		c1,  c2,  c3  = AEP.get_categories()
		oc1, oc2, oc3 = OAEP.get_categories()
		fc3 = np.logical_or(c3, oc3)
		fc12 = np.logical_not(fc3)
		fc3n = np.sum(fc3)
		fc12n = np.sum(fc12)

		# category 1 and 2
		uxh = ebdy.interpolate_to_points(ux, aap.x, aap.y)
		uyh = ebdy.interpolate_to_points(uy, aap.x, aap.y)
		vxh = ebdy.interpolate_to_points(vx, aap.x, aap.y)
		vyh = ebdy.interpolate_to_points(vy, aap.x, aap.y)
		uh =  ebdy.interpolate_to_points(u, aap.x, aap.y)
		vh =  ebdy.interpolate_to_points(v, aap.x, aap.y)

		uxoh = ebdy_old.interpolate_to_points(uxo, aap.x, aap.y)
		uyoh = ebdy_old.interpolate_to_points(uyo, aap.x, aap.y)
		vxoh = ebdy_old.interpolate_to_points(vxo, aap.x, aap.y)
		vyoh = ebdy_old.interpolate_to_points(vyo, aap.x, aap.y)
		uoh =  ebdy_old.interpolate_to_points(uo,  aap.x, aap.y)
		voh =  ebdy_old.interpolate_to_points(vo,  aap.x, aap.y)

		SLM = np.zeros([fc12n,] + [4,4], dtype=float)
		SLR = np.zeros([fc12n,] + [4,], dtype=float)

		# solve for departure points
		SLM[:,0,0] = uxh[fc12]
		SLM[:,0,1] = uyh[fc12]
		SLM[:,0,2] = 0.5/dt
		SLM[:,0,3] = 0.0
		SLM[:,1,0] = vxh[fc12]
		SLM[:,1,1] = vyh[fc12]
		SLM[:,1,2] = 0.0
		SLM[:,1,3] = 0.5/dt
		SLM[:,2,0] = 2.0/dt + 3*uxh[fc12]
		SLM[:,2,1] = 3*uyh[fc12]
		SLM[:,2,2] = -uxoh[fc12]
		SLM[:,2,3] = -uyoh[fc12]
		SLM[:,3,0] = 3*vxh[fc12]
		SLM[:,3,1] = 2.0/dt + 3*vyh[fc12]
		SLM[:,3,2] = -vxoh[fc12]
		SLM[:,3,3] = -vyoh[fc12]
		SLR[:,0] = uh[fc12]
		SLR[:,1] = vh[fc12]
		SLR[:,2] = 3*uh[fc12] - uoh[fc12]
		SLR[:,3] = 3*vh[fc12] - voh[fc12]
		OUT = np.linalg.solve(SLM, SLR)
		dx, dy, Dx, Dy = OUT[:,0], OUT[:,1], OUT[:,2], OUT[:,3]
		xd, yd = aap.x[fc12] - dx, aap.y[fc12] - dy
		xD, yD = aap.x[fc12] - Dx, aap.y[fc12] - Dy
		# seems we should now back check xd, yd to make sure they're actually in ebdy!
		test_key = ebdy.register_points(xd, yd)
		test_part = ebdy.registered_partitions[test_key]
		test1, test2, test3 = test_part.get_categories()
		### TEST KEYS FOR EBDY_OLD
		old_test_key = ebdy_old.register_points(xD, yD)
		old_test_part = ebdy_old.registered_partitions[old_test_key]
		old_test1, old_test2, old_test3 = old_test_part.get_categories()
		total_test3 = np.logical_or(test3, old_test3)
		# reclassify things that are in test3 as being in c3
		fc3[fc12] = total_test3
		# recount c3
		fc3n = fc3.sum()
		xd_all[fc12] = xd
		yd_all[fc12] = yd
		xD_all[fc12] = xD
		yD_all[fc12] = yD

		# categroy 3... this is the tricky one
		if fc3n > 0:
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
			urrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(u.radial_value_list[0])
			vrrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(v.radial_value_list[0])
			ub_interp   = interp(ub)
			vb_interp   = interp(vb)
			urb_interp  = interp(urb)
			vrb_interp  = interp(vrb)
			urrb_interp  = interp(urrb)
			vrrb_interp  = interp(vrrb)
			ubs_interp  = interp(d1_der(ub))
			vbs_interp  = interp(d1_der(vb))
			urbs_interp = interp(d1_der(urb))
			vrbs_interp = interp(d1_der(vrb))
			urrbs_interp = interp(d1_der(urrb))
			vrrbs_interp = interp(d1_der(vrrb))

			old_bx_interp  = interp(ebdy_old.bdy.x)
			old_by_interp  = interp(ebdy_old.bdy.y)
			old_bxs_interp = interp(d1_der(ebdy_old.bdy.x))
			old_bys_interp = interp(d1_der(ebdy_old.bdy.y))
			old_nx_interp  = interp(ebdy_old.bdy.normal_x)
			old_ny_interp  = interp(ebdy_old.bdy.normal_y)
			old_nxs_interp = interp(d1_der(ebdy_old.bdy.normal_x))
			old_nys_interp = interp(d1_der(ebdy_old.bdy.normal_y))
			old_ub = ebdy_old.interpolate_radial_to_boundary(uo.radial_value_list[0])
			old_vb = ebdy_old.interpolate_radial_to_boundary(vo.radial_value_list[0])
			old_urb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(uo.radial_value_list[0])
			old_vrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(vo.radial_value_list[0])
			old_urrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(uo.radial_value_list[0])
			old_vrrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(vo.radial_value_list[0])
			# i think the old parm is right, but should think about
			old_ub_interp   = interp(old_ub)
			old_vb_interp   = interp(old_vb)
			old_urb_interp  = interp(old_urb)
			old_vrb_interp  = interp(old_vrb)
			old_urrb_interp  = interp(old_urrb)
			old_vrrb_interp  = interp(old_vrrb)
			old_ubs_interp  = interp(d1_der(old_ub))
			old_vbs_interp  = interp(d1_der(old_vb))
			old_urbs_interp = interp(d1_der(old_urb))
			old_vrbs_interp = interp(d1_der(old_vrb))
			old_urrbs_interp = interp(d1_der(old_urrb))
			old_vrrbs_interp = interp(d1_der(old_vrrb))

			xx = aap.x[fc3]
			yy = aap.y[fc3]
			def objective(s, r, so, ro):
				f = np.empty([s.size, 4])
				f[:,0] = old_bx_interp(so) + ro*old_nx_interp(so) + 2*dt*ub_interp(s) + 2*dt*r*urb_interp(s) + dt*r**2*urrb_interp(s) - xx
				f[:,1] = old_by_interp(so) + ro*old_ny_interp(so) + 2*dt*vb_interp(s) + 2*dt*r*vrb_interp(s) + dt*r**2*vrrb_interp(s) - yy
				f[:,2] = bx_interp(s) + r*nx_interp(s) + 1.5*dt*ub_interp(s) + 1.5*dt*r*urb_interp(s) + 0.75*dt*r**2*urrb_interp(s) - 0.5*dt*old_ub_interp(so) - 0.5*dt*ro*old_urb_interp(so) - 0.25*dt*ro**2*old_urrb_interp(so) - xx
				f[:,3] = by_interp(s) + r*ny_interp(s) + 1.5*dt*vb_interp(s) + 1.5*dt*r*vrb_interp(s) + 0.75*dt*r**2*vrrb_interp(s) - 0.5*dt*old_vb_interp(so) - 0.5*dt*ro*old_vrb_interp(so) - 0.25*dt*ro**2*old_vrrb_interp(so) - yy
				return f
			def Jac(s, r, so, ro):
				J = np.empty([s.size, 4, 4])
				# derivative with respect to s
				J[:,0,0] = 2*dt*ubs_interp(s) + 2*dt*r*urbs_interp(s) + dt*r**2*urrbs_interp(s)
				J[:,1,0] = 2*dt*vbs_interp(s) + 2*dt*r*vrbs_interp(s) + dt*r**2*vrrbs_interp(s)
				J[:,2,0] = bxs_interp(s) + r*nxs_interp(s) + 1.5*dt*ubs_interp(s) + 1.5*dt*r*urbs_interp(s) + 0.75*dt*r**2*urrbs_interp(s)
				J[:,3,0] = bys_interp(s) + r*nys_interp(s) + 1.5*dt*vbs_interp(s) + 1.5*dt*r*vrbs_interp(s) + 0.75*dt*r**2*vrrbs_interp(s)
				# derivative with respect to r
				J[:,0,1] = 2*dt*urb_interp(s) + 2*dt*r*urrb_interp(s)
				J[:,1,1] = 2*dt*vrb_interp(s) + 2*dt*r*vrrb_interp(s)
				J[:,2,1] = nx_interp(s) + 1.5*dt*urb_interp(s) + 1.5*dt*r*urrb_interp(s)
				J[:,3,1] = ny_interp(s) + 1.5*dt*vrb_interp(s) + 1.5*dt*r*vrrb_interp(s)
				# derivative with respect to so
				J[:,0,2] = old_bxs_interp(so) + ro*old_nxs_interp(so)
				J[:,1,2] = old_bys_interp(so) + ro*old_nys_interp(so)
				J[:,2,2] = -0.5*dt*old_ubs_interp(so) - 0.5*dt*ro*old_urbs_interp(so) - 0.25*dt*ro**2*old_urrbs_interp(so)
				J[:,3,2] = -0.5*dt*old_vbs_interp(so) - 0.5*dt*ro*old_vrbs_interp(so) - 0.25*dt*ro**2*old_vrrbs_interp(so)
				# derivative with respect to ro
				J[:,0,3] = old_nx_interp(so)
				J[:,1,3] = old_ny_interp(so)
				J[:,2,3] = -0.5*dt*old_urb_interp(so) - 0.5*dt*ro*old_urrb_interp(so)
				J[:,3,3] = -0.5*dt*old_vrb_interp(so) - 0.5*dt*ro*old_vrrb_interp(so)
				return J
			# take as guess inds our s, r
			s =   AEP.full_t[fc3]
			r =   AEP.full_r[fc3]
			so = OAEP.full_t[fc3]
			ro = OAEP.full_r[fc3]
			# now solve for sd, rd
			res = objective(s, r, so, ro)
			mres1 = np.hypot(res[:,0], res[:,1]).max()
			mres2 = np.hypot(res[:,2], res[:,3]).max()
			mres = max(mres1, mres2)
			tol = 1e-12
			while mres > tol:
				J = Jac(s, r, so, ro)
				d = np.linalg.solve(J, res)
				s  -= d[:,0]
				r  -= d[:,1]
				so -= d[:,2]
				ro -= d[:,3]
				res = objective(s, r, so, ro)
				mres1 = np.hypot(res[:,0], res[:,1]).max()
				mres2 = np.hypot(res[:,2], res[:,3]).max()
				mres = max(mres1, mres2)
			r_fail_1 = r.max() > 0.0
			r_fail_2 = r.min() < -ebdy.radial_width
			ro_fail_1 = ro.max() > 0.0
			ro_fail_2 = ro.min() < -ebdy_old.radial_width
			r_fail = r_fail_1 or r_fail_2
			ro_fail = ro_fail_1 or ro_fail_2
			fail = r_fail or ro_fail
			fail_amount = 0.0
			if fail:
				if r_fail_1:
					fail_amount = max(fail_amount, r.max())
					r[r > 0.0] = 0.0
				if r_fail_2:
					fail_amount = max(fail_amount, (-r-ebdy.radial_width).max())
					r[r < -ebdy.radial_width] = -ebdy.radial_width
				if ro_fail_1:
					fail_amount = max(fail_amount, ro.max())
					ro[ro > 0.0] = 0.0
				if ro_fail_2:
					fail_amount = max(fail_amount, (-ro-ebdy_old.radial_width).max())
					ro[ro < -ebdy_old.radial_width] = -ebdy_old.radial_width
				print('Failure! by: {:0.2e}'.format(fail_amount))
			
			# get the departure points
			xd = bx_interp(s) + nx_interp(s)*r
			yd = by_interp(s) + ny_interp(s)*r
			xD = old_bx_interp(so) + old_nx_interp(so)*ro
			yD = old_by_interp(so) + old_ny_interp(so)*ro
			xd_all[fc3] = xd
			yd_all[fc3] = yd
			xD_all[fc3] = xD
			yD_all[fc3] = yD

		# now interpolate to c
		ch1 = ebdy.interpolate_to_points((4/3)*c - (4/3)*dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True)
		ch2 = ebdy_old.interpolate_to_points((1/3)*c_old - (2/3)*dt*scaled_nonlinearity(c_old), xD_all, yD_all, fix_r=True)
		ch = ch1 - ch2
		# set the grid values
		c_new.grid_value[new_ebdyc.phys] = ch[:gp.N]
		# set the radial values
		c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
		# overwrite under grid under annulus by radial grid
		# _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
		new_ebdy.full_merge(c_new.grid_value, c_new.radial_value_list[0], 5)

		# apply a filter to c_new
		ff = c_new.radial_value_list[0]
		ffh = np.fft.fft(ff)
		ffh[:,100] = 0.0
		c_new.radial_value_list[0][:] = np.fft.ifft(ffh).real

		EP = ebdy.registered_partitions[-1]
		OEP = ebdy_old.registered_partitions[-1]

		# get cross categories
		cc1 = EP.category1[:gp.N].sum()
		cc2 = EP.category2[gp.N:].sum()
		occ1 = OEP.category1[:gp.N].sum()
		occ2 = OEP.category2[gp.N:].sum()
		print(cc2, occ2)

		# plot crossover points
		if False:
			fig, ax = plt.subplots()
			ax.scatter(EP.x, OEP.y, color='gray')
			ax.scatter(EP.x[:gp.N][EP.category1[:gp.N]], EP.y[:gp.N][EP.category1[:gp.N]], color='blue')
			ax.scatter(EP.x[gp.N:][EP.category2[gp.N:]], EP.y[gp.N:][EP.category2[gp.N:]], color='red')
			ax.plot(ebdy.interface.x, ebdy.interface.y, color='purple')
			ax.set_title('current')

			fig, ax = plt.subplots()
			ax.scatter(OEP.x, OEP.y, color='gray')
			ax.scatter(OEP.x[:gp.N][OEP.category1[:gp.N]], OEP.y[:gp.N][OEP.category1[:gp.N]], color='blue')
			ax.scatter(OEP.x[gp.N:][OEP.category2[gp.N:]], OEP.y[gp.N:][OEP.category2[gp.N:]], color='red')
			ax.plot(ebdy_old.interface.x, ebdy_old.interface.y, color='purple')
			ax.plot(ebdy_old.bdy.x, ebdy_old.bdy.y, color='purple')
			ax.set_title('old')

		# ch = ch1
		# c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
		# # set the radial values
		# c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
		# aa = c_new.radial_value_list[0][-1].copy()

		# ch = ch2
		# c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
		# # set the radial values
		# c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
		# bb = c_new.radial_value_list[0][-1].copy()

		# visualize fc3n
		if False:
			fig, ax = plt.subplots()
			is_fc3 = EmbeddedFunction(new_ebdyc)
			is_fc3_grid = np.zeros(x.shape, dtype=bool)
			is_fc3_grid[new_ebdyc.phys] = fc3[:gp.N]
			is_fc3_radial = fc3[gp.N:].reshape(new_ebdy.radial_shape)
			is_fc3.load_data(is_fc3_grid, [is_fc3_radial,])
			clf = is_fc3.plot(ax)
			for bb, col in zip([new_ebdyc, ebdyc, ebdyc_old], ['red', 'pink', 'gray']):
				bbb = bb.ebdys[0].bdy
				ax.plot(bbb.x, bbb.y, color=col)
			ax.scatter(new_ebdyc.ebdys[0].radial_x, new_ebdyc.ebdys[0].radial_y, color='white')

	# now reset naming conventions
	try:
		ebdy_old_save = ebdy_old
		ebdyc_old_save = ebdyc_old
		c_old_save = c_old
	except:
		pass

	ebdy_old = ebdy
	ebdyc_old = ebdyc
	ebdy = new_ebdy
	ebdyc = new_ebdyc
	c_old = c
	c = c_new
	uxo = ux
	uyo = uy
	vxo = vx
	vyo = vy
	uo = u
	vo = v

	# assess error on underlying grid
	err = (np.abs(c.grid_value - ce)*ebdyc.phys).max()

	ts.append(t)
	errors.append(err)

	t += dt
	print('   t = {:0.3f}'.format(t), 'of', max_time, 'error: {:0.2e}'.format(err))


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
err = (c.grid_value - c_eulerian)*ebdyc.phys
err = np.ma.array(err, mask=new_ebdyc.ext)
merr = np.abs(err).max()

print('On grid:            {:0.1e}'.format(merr))

# assess error in position
err_bx = bxe - ebdy.bdy.x
err_by = bye - ebdy.bdy.y
err_b = max(np.abs(err_bx).max(), np.abs(err_by).max())

print('Boundary motion is: {:0.1e}'.format(err_b))

# assess on radial grid (via NUFFT of eulerian solution)
cr = ebdy.interpolate_grid_to_radial(c_eulerian)
err_r = cr - c.radial_value_list[0]
merr_r = np.abs(err_r).max()

print('On radial grid:     {:0.1e}'.format(merr_r))

# plot the errors properly...
cerr = EmbeddedFunction(ebdyc)
cerr.load_data(err, [err_r,])

fig, axs = plt.subplots(2)
ax = axs[0]
ax2 = axs[1]
clf = (np.abs(cerr)+1e-15).plot(ax, norm=mpl.colors.LogNorm(), vmin=1e-6)
ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='red')
ax.plot(ebdy.interface.x, ebdy.interface.y, color='red')
ax.plot(ebdy_old.bdy.x, ebdy_old.bdy.y, color='pink')
ax.plot(ebdy_old.interface.x, ebdy_old.interface.y, color='pink')
ax.plot(ebdy_old_save.bdy.x, ebdy_old_save.bdy.y, color='gray')
ax.plot(ebdy_old_save.interface.x, ebdy_old_save.interface.y, color='gray')
# ax.set_xlim(1.1,1.5)
# ax.set_ylim(-0.1,0.1)

ax2.plot(cerr.radial_value_list[0].T+1e-5)
ax2.set_xlim(80, 120)
# ax2.set_ylim(4e-6, 8e-5)




# fig, ax = plt.subplots()
# clf = (cerr+1e-15).plot(ax, norm=mpl.colors.LogNorm(), vmin=1e-6)
# plt.colorbar(clf)
# for xtr, ytr in zip(x_tracers, y_tracers):
# 	ax.plot(xtr, ytr, alpha=0.2)



# fig, ax = plt.subplots()
# clf = (cerr+1e-15).plot(ax, norm=mpl.colors.LogNorm(), vmin=1e-6)
# plt.colorbar(clf)
# ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='red')
# ax.plot(ebdy.interface.x, ebdy.interface.y, color='red')
# ax.plot(ebdy_old.bdy.x, ebdy_old.bdy.y, color='pink')
# ax.plot(ebdy_old.interface.x, ebdy_old.interface.y, color='pink')
# ax.plot(ebdy_old_save.bdy.x, ebdy_old_save.bdy.y, color='gray')
# ax.plot(ebdy_old_save.interface.x, ebdy_old_save.interface.y, color='gray')

