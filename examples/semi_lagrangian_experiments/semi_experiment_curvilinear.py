import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
import time
from ipde.utilities import affine_transformation
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ipde.annular.stokes import AnnularStokesSolver
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from near_finder.coordinate_routines import compute_local_coordinates
from near_finder.near_routines import points_near_points, points_near_curve
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary

nb = 400
ng = int(nb)
M = 20
pad_zone = 4
interior = True
slepian_r = 20
reparametrize = False

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, x=np.pi, y=np.pi, a=0.1, f=5))
# reparametrize if reparametrizing
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, interior, M, 0.25/M, pad_zone, MOL.step)

################################################################################
# generate data

# coordinates for curvilinear region
xr = ebdy.radial_x
yr = ebdy.radial_y
rr = ebdy.radial_r
tr = ebdy.radial_t

# generate a velocity field that's zero on both boundaries
fu_func = lambda r, t: r*np.cos(2*t)
fv_func = lambda r, t: np.exp(-r)*np.sin(t)
fur = fu_func(rr, tr)
fvr = fv_func(rr, tr)
lower_u = np.zeros(nb)
lower_v = np.zeros(nb)
upper_u = np.zeros(nb)
upper_v = np.zeros(nb)
# solve radial stokes problem
AAG = ApproximateAnnularGeometry(nb, M, ebdy.radial_width, ebdy.approximate_radius)
# construct the preconditioner (for Dirichlet BC)
solver = AnnularStokesSolver(AAG, 1.0)
print('Solving')
# get the real geometry
sp =  ebdy.bdy.speed     if interior else ebdy.interface.speed
cur = ebdy.bdy.curvature if interior else ebdy.interface.curvature
RAG = RealAnnularGeometry(sp, cur, AAG)
# convert (u,v) to (r,t)
fr, ft = ebdy.convert_uv_to_rt(fur, fvr)
lr, lt = ebdy.convert_uv_to_rt(lower_u, lower_v)
ur, ut = ebdy.convert_uv_to_rt(upper_u, upper_v)
re_radial, te_radial, pe_radial = solver.solve(RAG, fr, ft, lr, lt, ur, ut, verbose=False, tol=1e-12)
ur, vr = ebdy.convert_rt_to_uv(re_radial, te_radial)
U = np.hypot(ur, vr).max()
u = ur/U
v = vr/U
# get derivatives of the velocity field
ux, uy = ebdy.radial_grid_derivatives(u)
vx, vy = ebdy.radial_grid_derivatives(v)

# get a function c to test with
c_func = lambda x, y: np.exp(np.cos(x))*np.sin(y)
c = c_func(xr, yr)

################################################################################
# test advectors

dt = 0.01

### eulerian advector
st = time.time()
cx, cy = ebdy.radial_grid_derivatives(c)
cu1 = c - dt*(u*cx + v*cy)
time_eulerian = time.time() - st

### non-linear departure based semi-lagrangian advector
def objective_function(xx, yy):
	ox = xx + dt*ebdy.interpolate_radial_to_points(u, xx, yy) - xr
	oy = yy + dt*ebdy.interpolate_radial_to_points(v, xx, yy) - yr
	return ox, oy
def Jacobian(xx, yy):
	Jxx = 1.0 + dt*ebdy.interpolate_radial_to_points(ux, xx, yy)
	Jxy = dt*ebdy.interpolate_radial_to_points(uy, xx, yy)
	Jyx = dt*ebdy.interpolate_radial_to_points(vx, xx, yy)
	Jyy = 1.0 + dt*ebdy.interpolate_radial_to_points(vy, xx, yy)
	return Jxx, Jxy, Jyx, Jyy
gx = xr - dt*u
gy = yr - dt*v
xn = gx.copy()
yn = gy.copy()
resx, resy = objective_function(xn, yn)
res = np.hypot(resx, resy).max()
tol = 1e-12
i = 0
print('\nInverting departure point system')
while res > tol:
	Jxx, Jxy, Jyx, Jyy = Jacobian(xn, yn)
	det = Jxx*Jyy - Jxy*Jyx
	idet = 1.0/det
	dx = -idet*(Jyy*resx - Jyx*resy)
	dy = -idet*(-Jxy*resx + Jxx*resy)
	xn = xn + dx
	yn = yn + dy
	ebdy.register_points(xn, yn, nearly_radial=True)
	# xn += dx
	# yn += dy
	resx, resy = objective_function(xn, yn)
	res = np.hypot(resx, resy).max()
	i += 1
	print(i, '{:0.1e}'.format(res))
cu2 = ebdy.interpolate_radial_to_points(c, xn, yn)
time_departure = time.time()-st

### linearized departure based semi-lagrangian advector
st = time.time()
# get departure points
nt = np.prod(xr.shape)
SLM = np.zeros([nt,] + [2,2], dtype=float)
SLR = np.zeros([nt,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*ux.ravel()
SLM[:,0,1] = dt*uy.ravel()
SLM[:,1,0] = dt*vx.ravel()
SLM[:,1,1] = 1 + dt*vy.ravel()
SLR[:,0] = dt*u.ravel()
SLR[:,1] = dt*v.ravel()
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0].reshape(xr.shape), OUT[:,1].reshape(xr.shape)
xd, yd = xr - xdt, yr - ydt
# interpolate onto departure points
cu3 = ebdy.interpolate_radial_to_points(c, xd, yd, nearly_radial=True)
time_linear_departure = time.time()-st

print('')
print('Err, eul vs. departure  {:0.1e}'.format(np.abs(cu1-cu2).max()))
print('Err, eul vs. linear dep {:0.1e}'.format(np.abs(cu1-cu3).max()))
print('Err, dep vs. linear dep {:0.1e}'.format(np.abs(cu2-cu3).max()))

print('')
print('Time, eulerian   {:0.1f}'.format(time_eulerian*1000))
print('Time, departure  {:0.1f}'.format(time_departure*1000))
print('Time, linear dep {:0.1f}'.format(time_linear_departure*1000))

