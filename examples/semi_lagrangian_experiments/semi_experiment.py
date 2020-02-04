import numpy as np
import finufftpy
import time

"""
Test semi-lagrangian solve...
"""
# set timestep
dt = 0.01

# generate a grid
n = 128
v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
# fourier modes
kv = np.fft.fftfreq(n, h/(2*np.pi))
kx, ky = np.meshgrid(kv, kv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# generate a velocity field (assume steady in time)
u = np.sin(x)*np.cos(y)
v = -np.cos(x)*np.sin(y)

# gradient function
def gradient(u):
	uh = np.fft.fft2(u)
	ux = np.fft.ifft2(uh*ikx).real
	uy = np.fft.ifft2(uh*iky).real
	return ux, uy

# we assume we already will have to compute these anyway
ux, uy = gradient(u)
vx, vy = gradient(v)

class interp2d(object):
	def __init__(self, u):
		self.uh = np.fft.fft2(u).copy()
		self.scale = 1.0/np.prod(self.uh.shape)
	def __call__(self, x, y):
		sh = x.shape
		x, y = x.ravel(), y.ravel()
		work = np.empty(x.size, dtype=complex)
		info = finufftpy.nufft2d2(x, y, work, 1, 1e-14, self.uh, modeord=1)
		return (work.real*self.scale).reshape(sh)
def objective_function(xx, yy):
	ox = xx + dt*u_interpolater(xx, yy) - x
	oy = yy + dt*v_interpolater(xx, yy) - y
	return ox, oy
def Jacobian(xx, yy):
	Jxx = 1.0 + dt*ux_interpolater(xx, yy)
	Jxy = dt*uy_interpolater(xx, yy)
	Jyx = dt*vx_interpolater(xx, yy)
	Jyy = 1.0 + dt*vy_interpolater(xx, yy)
	return Jxx, Jxy, Jyx, Jyy

# initial c field
c = np.exp(np.cos(x))*np.sin(y)
### update c (via eulerian rep)
st = time.time()
cx, cy = gradient(c)
cu1 = c - dt*(u*cx + v*cy)
time_eulerian = time.time() - st
### update c (via non-linear departure pt)
st = time.time()
# define needed interpolaters
u_interpolater = interp2d(u)
v_interpolater = interp2d(v)
ux_interpolater = interp2d(ux)
uy_interpolater = interp2d(uy)
vx_interpolater = interp2d(vx)
vy_interpolater = interp2d(vy)
c_interpolater = interp2d(c)
# get initial guess for each gridpoint
gx = x - dt*u
gy = y - dt*v
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
	xn += dx
	yn += dy
	resx, resy = objective_function(xn, yn)
	res = np.hypot(resx, resy).max()
	i += 1
	print(i, '{:0.1e}'.format(res))
cu2 = c_interpolater(xn, yn)
time_departure = time.time()-st
### update c (via linear departure pt)
st = time.time()
# get departure points
SLM = np.zeros([n**2,] + [2,2], dtype=float)
SLR = np.zeros([n**2,] + [2,], dtype=float)
SLM[:,0,0] = 1 + dt*ux.ravel()
SLM[:,0,1] = dt*uy.ravel()
SLM[:,1,0] = dt*vx.ravel()
SLM[:,1,1] = 1 + dt*vy.ravel()
SLR[:,0] = dt*u.ravel()
SLR[:,1] = dt*v.ravel()
OUT = np.linalg.solve(SLM, SLR)
xdt, ydt = OUT[:,0].reshape(n,n), OUT[:,1].reshape(n,n)
xd, yd = x - xdt, y - ydt
# interpolate onto departure points
cu3 = c_interpolater(xd, yd)
time_linear_departure = time.time()-st
### update c (via arrival method)
xn = x + dt*u
yn = y + dt*v
# compute weights
st = time.time()
Jxx = 1 + dt*ux
Jxy = dt*uy
Jyx = dt*vx
Jyy = 1 + dt*vy
det = Jxx*Jyy - Jxy*Jyx
cw = c*det
ch = np.zeros([n,n], dtype=complex, order='F')
finufftpy.nufft2d1(xn, yn, cw.astype(complex), -1, 1e-14, n, n, ch, modeord=1)
cu4 = np.fft.ifft2(ch).real
time_arrival = time.time()-st

print('')
print('Err, eul vs. dep     {:0.1e}'.format(np.abs(cu1-cu2).max()))
print('Err, eul vs. lin dep {:0.1e}'.format(np.abs(cu1-cu3).max()))
print('Err, eul vs. arr     {:0.1e}'.format(np.abs(cu1-cu4).max()))
print('Err, dep vs. arr     {:0.1e}'.format(np.abs(cu2-cu4).max()))
print('Err, dep vs. lin dep {:0.1e}'.format(np.abs(cu2-cu3).max()))

print('')
print('Time, eulerian   {:0.1f}'.format(time_eulerian*1000))
print('Time, departure  {:0.1f}'.format(time_departure*1000))
print('Time, linear dep {:0.1f}'.format(time_linear_departure*1000))
print('Time, arrival    {:0.1f}'.format(time_arrival*1000))



