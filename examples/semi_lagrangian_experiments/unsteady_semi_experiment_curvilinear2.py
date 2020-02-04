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
max_time = 0.2
# set timestep
dt = 0.005
# number of boundary points
nb = 200
# number of grid points
ng = int(nb/2)
# number of chebyshev modes
M = 8
# padding zone
pad_zone = 0
# smoothness of rolloff functions
# slepian_r = 2*M
slepian_r = M

# generate a velocity field
kk = 2*np.pi/3
u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*(1+np.cos(2*np.pi*t))
v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*(1+np.cos(2*np.pi*t))
c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)
# for testing purposes
ux_function = lambda x, y, t: kk*np.cos(kk*x)*np.cos(kk*y)*(1+np.cos(2*np.pi*t))
uy_function = lambda x, y, t: -kk*np.sin(kk*x)*np.sin(kk*y)*(1+np.cos(2*np.pi*t))
vx_function = lambda x, y, t: kk*np.sin(kk*x)*np.sin(kk*y)*(1+np.cos(2*np.pi*t))
vy_function = lambda x, y, t: -kk*np.cos(kk*x)*np.cos(kk*y)*(1+np.cos(2*np.pi*t))

# gradient function
def gradient(f):
	fh = np.fft.fft2(f)
	fx = np.fft.ifft2(fh*ikx).real
	fy = np.fft.ifft2(fh*iky).real
	return fx, fy

################################################################################
# Get truth via just using Forward Euler on periodic domain

bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.1, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bx = bdy.x
by = bdy.y

# generate a grid
v, h = np.linspace(-1.5, 1.5, ng, endpoint=False, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
# fourier modes
kv = np.fft.fftfreq(ng, h/(2*np.pi))
kv[int(ng/2)] = 0.0
kx, ky = np.meshgrid(kv, kv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# initial c field
c0 = c0_function(x, y)

print('Testing Forward-Euler Method')
st = time.time()
t = 0.0
c = c0.copy()
while t < max_time-1e-10:
	cx, cy = gradient(c)
	u = u_function(x, y, t)
	v = v_function(x, y, t)
	bu = u_function(bx, by, t)
	bv = v_function(bx, by, t)
	if t == 0:
		c = c - dt*(u*cx + v*cy)
		bx = bx + dt*bu
		by = by + dt*bv
	else:
		c = c - 0.5*dt*(3*u*cx + 3*v*cy - uo*cxo - vo*cyo)
		bx = bx + 0.5*dt*(3*bu-buo)
		by = by + 0.5*dt*(3*bv-bvo)
	bx, by = arc_length_parameterize(bx, by)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
	uo = u
	vo = v
	co = c
	cxo = cx
	cyo = cy
	buo = bu
	bvo = bv
c_eulerian = c.copy()
time_eulerian = time.time() - st

bxe = bx
bye = by

################################################################################
# Semi-Lagrangian Non-linear departure point method

print('Testing Linear Departure Method')

# get heaviside function
MOL = SlepianMollifier(slepian_r)

# construct boundary and reparametrize
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.1, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh, pad_zone, MOL.step)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# get a grid
grid = Grid([-1.5, 1.5], ng, [-1.5, 1.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid)

original_ebdy = ebdy

# initial c field
c0 = EmbeddedFunction(ebdyc)
c0.define_via_function(c0_function)

fig, ax = plt.subplots()

# now timestep
c = c0.copy()
t = 0
while t < max_time-1e-10:

	st = time.time()
	# get the velocity fields
	u = EmbeddedFunction(ebdyc)
	u.define_via_function(lambda x, y: u_function(x, y, t))
	v = EmbeddedFunction(ebdyc)
	v.define_via_function(lambda x, y: v_function(x, y, t))
	print('Time to compute u:                            {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	# get the velocity fields on the boundary
	ub = ebdyc.interpolate_radial_to_boundary(u)
	vb = ebdyc.interpolate_radial_to_boundary(v)
	print('Time to interp u to bdy:                      {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	# first, move the boundary with the fluid velocity
	bx = ebdy.bdy.x + dt*ub.bdy_value_list[0]
	by = ebdy.bdy.y + dt*vb.bdy_value_list[0]
	print('Time to move bdy:                             {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	# take gradients of the velocity fields
	dx = lambda f: fd_x_4(f, grid.xh, periodic_fix=True)
	dy = lambda f: fd_y_4(f, grid.yh, periodic_fix=True)
	ux, uy = ebdyc.gradient2(u, dx, dy, cutoff=False)
	vx, vy = ebdyc.gradient2(v, dx, dy, cutoff=False)
	print('Time to compute u gradients:                  {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	# now generate a new ebdy based on the moved boundary
	new_bdy = GSB(*arc_length_parameterize(bx, by))
	bh = new_bdy.dt*new_bdy.speed.min()
	# construct embedded boundary
	new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh, pad_zone, MOL.step)
	new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
	new_ebdyc.register_grid(grid)
	print('Time to generate new ebdy and register grid:  {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	if True:
		ax.plot(new_bdy.x, new_bdy.y)
		plt.pause(0.05)

	# let's get the points that need to be interpolated to
	gp = new_ebdyc.grid_pna
	ap = new_ebdyc.radial_pts

	aax = np.concatenate([gp.x, ap.x])
	aay = np.concatenate([gp.y, ap.y])
	aap = PointSet(x=aax, y=aay)

	AP_key = ebdy.register_points(aap.x, aap.y)
	print('Time to register new points with old ebdy |A: {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

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

	# graph this
	if False:
		cat = np.zeros_like(x)
		catw = np.zeros(gp.N)
		catw[c1[:gp.N]] = 1.0
		catw[c2[:gp.N]] = 2.0
		catw[c3[:gp.N]] = 3.0
		cat[new_ebdyc.phys_not_in_annulus] = catw

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
	xd_all[c1_2] = xd
	yd_all[c1_2] = yd
	print('Time to find departure points, category 1:    {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()
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
		ub  = ebdy.interpolate_radial_to_boundary(u.radial_value_list[0])
		vb  = ebdy.interpolate_radial_to_boundary(v.radial_value_list[0])
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
		s = new_ebdy.radial_t.ravel()[c3[gp.N:]]
		r = new_ebdy.radial_r.ravel()[c3[gp.N:]]
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
			# print(mres)
		# get the departure points
		xd = bx_interp(s) + nx_interp(s)*r
		yd = by_interp(s) + ny_interp(s)*r
		xd_all[c3] = xd
		yd_all[c3] = yd
		print('Time to find departure points, category 3:    {:0.1f}'.format( (time.time()-st)*1000 ))
		st = time.time()
	# now interpolate to c
	ch = ebdy.interpolate_to_points(c, xd_all, yd_all)
	print('Time to do interpolation:                     {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()
	# set the grid values
	c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
	# set the radial values
	c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
	print('Time to set function values:                  {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()
	# overwrite under grid under annulus by radial grid
	_ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
	print('Time for grid interpolation:                  {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

	# plot things relating to separation of points
	if False:
		fig, ax = plt.subplots()
		ax.plot(ebdy.bdy.x,           ebdy.bdy.y,           color='black', linewidth=3)
		ax.plot(ebdy.interface.x,     ebdy.interface.y,     color='black', linewidth=3)
		ax.plot(new_ebdy.bdy.x,       new_ebdy.bdy.y,       color='blue' , linewidth=3)
		ax.plot(new_ebdy.interface.x, new_ebdy.interface.y, color='blue' , linewidth=3)
		ax.scatter(aap.x[AEP.category1], aap.y[AEP.category1], color='red')
		ax.scatter(aap.x[AEP.category2], aap.y[AEP.category2], color='purple')
		ax.scatter(aap.x[AEP.category3], aap.y[AEP.category3], color='green')

	# now reset naming conventions
	ebdy_old = ebdy
	ebdyc_old = ebdyc
	ebdy = new_ebdy
	ebdyc = new_ebdyc
	c = c_new

	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')

################################################################################
# Evaluate

print('\n\nAssessing Errors:')
# assess error on underlying grid
err = c.grid_value - c_eulerian
err = np.zeros_like(x)
err = np.abs(c.grid_value - c_eulerian)*ebdyc.phys
err = np.ma.array(err, mask=ebdyc.ext)
merr = err.max()

fig, ax = plt.subplots()
clf = ax.pcolor(x, y, np.abs(err), vmin=1e-5, norm=mpl.colors.LogNorm())
plt.plot(new_ebdy.bdy.x, ebdy.bdy.y, color='red')
plt.colorbar(clf)

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
cerr.plot(ax, vmin=1e-5, norm=mpl.colors.LogNorm())

