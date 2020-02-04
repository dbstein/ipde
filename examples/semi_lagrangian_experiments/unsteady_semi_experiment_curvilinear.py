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
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 0.05
# set timestep
dt = 0.05
# number of boundary points
nb = 300
# number of grid points
ng = int(nb/2)
# number of chebyshev modes
M = 8
# padding zone
pad_zone = 3
# smoothness of rolloff functions
slepian_r = 2*M

# generate a velocity field
kk = 2*np.pi/3
u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*(1+np.cos(2*np.pi*t))
v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*(1+np.cos(2*np.pi*t))
c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)

# gradient function
def gradient(f):
	fh = np.fft.fft2(f)
	fx = np.fft.ifft2(fh*ikx).real
	fy = np.fft.ifft2(fh*iky).real
	return fx, fy

################################################################################
# Get truth via just using Forward Euler on periodic domain

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
	c -= dt*(u*cx + v*cy)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
c_eulerian = c.copy()
time_eulerian = time.time() - st

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

# initial c field
c0 = EmbeddedFunction(ebdyc)
c0.define_via_function(c0_function)

# now timestep
c = c0.copy()
t = 0
# while t < max_time-1e-10:

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

# let's get the points that need to be interpolated to
gp = new_ebdyc.grid_pna
ap = new_ebdyc.radial_pts

from pybie2d.point_set import PointSet
aax = np.concatenate([gp.x, ap.x])
aay = np.concatenate([gp.y, ap.y])
aap = PointSet(x=aax, y=aay)

AP_key = ebdy.register_points(aap.x, aap.y)
print('Time to register new points with old ebdy |A: {:0.1f}'.format( (time.time()-st)*1000 ))
st = time.time()

# register these with ebdy
gp_key = ebdy.register_points(gp.x, gp.y)
print('Time to register new points with old ebdy |g: {:0.1f}'.format( (time.time()-st)*1000 ))
st = time.time()
ap_key = ebdy.register_points(ap.x, ap.y, nearly_radial=True)
print('Time to register new points with old ebdy |a: {:0.1f}'.format( (time.time()-st)*1000 ))
st = time.time()

# now we need to interpolate onto things
AEP = ebdy.registered_partitions[ap_key]
GEP = ebdy.registered_partitions[gp_key]

# generate a holding ground for the new c
c_new = EmbeddedFunction(new_ebdyc)
c_new.zero()

# advect those in the annulus
# category 1
c1,  c2,  c3  = AEP.get_categories()
c1n, c2n, c3n = AEP.get_category_Ns()
uxh = ebdy.interpolate_to_points(ux, ap.x, ap.y)
uyh = ebdy.interpolate_to_points(uy, ap.x, ap.y)
vxh = ebdy.interpolate_to_points(vx, ap.x, ap.y)
vyh = ebdy.interpolate_to_points(vy, ap.x, ap.y)
uh = ebdy.interpolate_to_points(u, ap.x, ap.y)
vh = ebdy.interpolate_to_points(v, ap.x, ap.y)
SLM = np.zeros([c1n,] + [2,2], dtype=float)
SLR = np.zeros([c1n,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*uxh[c1]
SLM[:,0,1] = dt*uyh[c1]
SLM[:,1,0] = dt*vxh[c1]
SLM[:,1,1] = 1 + dt*vyh[c1]
SLR[:,0] = dt*uh[c1]
SLR[:,1] = dt*vh[c1]
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0], OUT[:,1]
xd, yd = ap.x[c1] - xdt, ap.y[c1] - ydt
# udate c
ch = ebdy.interpolate_to_points(c, xd, yd)
c_new.radial_value_list[0][c1.reshape(ebdy.radial_shape)] = ch
print('Time for annular advection cat 1:             {:0.1f}'.format( (time.time()-st)*1000 ))
st = time.time()
# category 2
SLM = np.zeros([c2n,] + [2,2], dtype=float)
SLR = np.zeros([c2n,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*uxh[c2]
SLM[:,0,1] = dt*uyh[c2]
SLM[:,1,0] = dt*vxh[c2]
SLM[:,1,1] = 1 + dt*vyh[c2]
SLR[:,0] = dt*uh[c2]
SLR[:,1] = dt*vh[c2]
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0], OUT[:,1]
xd, yd = ap.x[c2] - xdt, ap.y[c2] - ydt
# udate c
ch = ebdy.interpolate_to_points(c, xd, yd)
c_new.radial_value_list[0][c2.reshape(ebdy.radial_shape)] = ch
print('Time for annular advection cat 2:             {:0.1f}'.format( (time.time()-st)*1000 ))
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
	xo = new_ebdy.radial_x.ravel()[c3]
	yo = new_ebdy.radial_y.ravel()[c3]
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
	s = new_ebdy.radial_t.ravel()[c3]
	r = new_ebdy.radial_r.ravel()[c3]
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
	# now get the c values
	ch = ebdy.interpolate_to_points(c, xd, yd)
	c_new.radial_value_list[0][c3.reshape(ebdy.radial_shape)] = ch
	print('Time for annular advection cat 3:             {:0.1f}'.format( (time.time()-st)*1000 ))
	st = time.time()

# advect those in the grid
# category 1
c1,  c2,  c3  = GEP.get_categories()
c1n, c2n, c3n = GEP.get_category_Ns()
uxh = ebdy.interpolate_to_points(ux, gp.x, gp.y)
uyh = ebdy.interpolate_to_points(uy, gp.x, gp.y)
vxh = ebdy.interpolate_to_points(vx, gp.x, gp.y)
vyh = ebdy.interpolate_to_points(vy, gp.x, gp.y)
uh = ebdy.interpolate_to_points(u, gp.x, gp.y)
vh = ebdy.interpolate_to_points(v, gp.x, gp.y)
SLM = np.zeros([c1n,] + [2,2], dtype=float)
SLR = np.zeros([c1n,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*uxh[c1]
SLM[:,0,1] = dt*uyh[c1]
SLM[:,1,0] = dt*vxh[c1]
SLM[:,1,1] = 1 + dt*vyh[c1]
SLR[:,0] = dt*uh[c1]
SLR[:,1] = dt*vh[c1]
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0], OUT[:,1]
xd, yd = gp.x[c1] - xdt, gp.y[c1] - ydt
# udate c
ch = ebdy.interpolate_to_points(c, xd, yd)
work = np.empty_like(gp.x)
work[c1] = ch
print('Time for grid advection cat 1:                {:0.1f}'.format( (time.time()-st)*1000 ))
st = time.time()
# category 2
SLM = np.zeros([c2n,] + [2,2], dtype=float)
SLR = np.zeros([c2n,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*uxh[c2]
SLM[:,0,1] = dt*uyh[c2]
SLM[:,1,0] = dt*vxh[c2]
SLM[:,1,1] = 1 + dt*vyh[c2]
SLR[:,0] = dt*uh[c2]
SLR[:,1] = dt*vh[c2]
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0], OUT[:,1]
xd, yd = gp.x[c2] - xdt, gp.y[c2] - ydt
ch = ebdy.interpolate_to_points(c, xd, yd)
work[c2] = ch
# set the new c values
c_new.grid_value[new_ebdyc.phys_not_in_annulus] = work
print('Time for grid advection cat 2:                {:0.1f}'.format( (time.time()-st)*1000 ))
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
	ax.scatter(ap.x[AEP.category1], ap.y[AEP.category1], color='red')
	ax.scatter(ap.x[AEP.category2], ap.y[AEP.category2], color='purple')
	ax.scatter(ap.x[AEP.category3], ap.y[AEP.category3], color='green')

	fig, ax = plt.subplots()
	ax.plot(ebdy.bdy.x,           ebdy.bdy.y,           color='black', linewidth=3)
	ax.plot(ebdy.interface.x,     ebdy.interface.y,     color='black', linewidth=3)
	ax.plot(new_ebdy.bdy.x,       new_ebdy.bdy.y,       color='blue' , linewidth=3)
	ax.plot(new_ebdy.interface.x, new_ebdy.interface.y, color='blue' , linewidth=3)
	ax.scatter(gp.x[GEP.category1], gp.y[GEP.category1], color='red')
	ax.scatter(gp.x[GEP.category2], gp.y[GEP.category2], color='purple')
	ax.scatter(gp.x[GEP.category3], gp.y[GEP.category3], color='green')

# now reset naming conventions
ebdy = new_ebdy
ebdyc = new_ebdyc
c = c_new

t += dt
print('   t = {:0.3f}'.format(t), max_time, '\r', end='')

################################################################################
# Evaluate

err = c.grid_value - c_eulerian
err = np.zeros_like(x)
err = (c.grid_value - c_eulerian)*ebdyc.phys

err = err[ebdyc.phys]
err = np.abs(err).max()
print(err)
