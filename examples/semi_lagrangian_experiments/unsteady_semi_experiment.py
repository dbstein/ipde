import numpy as np
import finufftpy
import time

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 1.0
# set timestep
dt = 0.01
# discretization size
n = 32

# generate a velocity field (assume steady in time)
u_function = lambda x, y, t:  np.sin(x)*np.cos(y)*(1+np.cos(2*np.pi*t))
v_function = lambda x, y, t: -np.cos(x)*np.sin(y)*(1+np.cos(2*np.pi*t))

# gradient function
def gradient(f):
	fh = np.fft.fft2(f)
	fx = np.fft.ifft2(fh*ikx).real
	fy = np.fft.ifft2(fh*iky).real
	return fx, fy

################################################################################
# Forward-Euler, Small Timestep, double grid for "Truth"

adj = 4

n *= adj
dt /= adj

# generate a grid
v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
# fourier modes
kv = np.fft.fftfreq(n, h/(2*np.pi))
kv[int(n/2)] = 0.0
kx, ky = np.meshgrid(kv, kv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# initial c field
c0 = np.exp(np.cos(x))*np.sin(y)

print('')
print("Getting 'truth'")
t = 0.0
c = c0.copy()
while t < max_time-1e-10:
	cx, cy = gradient(c)
	u = u_function(x, y, t)
	v = v_function(x, y, t)
	c -= dt*(u*cx + v*cy)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
c_true = c.copy()[::adj,::adj]

n = int(n/adj)
dt *= adj

################################################################################
# Forward-Euler Timestepper

# generate a grid
v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
# fourier modes
kv = np.fft.fftfreq(n, h/(2*np.pi))
kv[int(n/2)] = 0.0
kx, ky = np.meshgrid(kv, kv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# initial c field
c0 = np.exp(np.cos(x))*np.sin(y)

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

print('Testing Non-linear Departure Method')

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

st = time.time()
t = 0.0
c = c0.copy()
while t < max_time-1e-10:
	# get velocity fields and gradients
	u = u_function(x, y, t)
	v = v_function(x, y, t)
	ux, uy = gradient(u)
	vx, vy = gradient(v)
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
	c = c_interpolater(xn, yn)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
c_nonlinear_departure = c.copy()
time_nonlinear_departure = time.time() - st

################################################################################
# Semi-Lagrangian Linear departure point method

print('Testing Linear Departure Method')

st = time.time()
t = 0.0
c = c0.copy()
SLM = np.zeros([n**2,] + [2,2], dtype=float)
SLR = np.zeros([n**2,] + [2,], dtype=float)
while t < max_time-1e-10:
	# get velocity fields and gradients
	u = u_function(x, y, t)
	v = v_function(x, y, t)
	ux, uy = gradient(u)
	vx, vy = gradient(v)
	# define needed interpolaters
	c_interpolater = interp2d(c)
	# solve for departure points
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
	c = c_interpolater(xd, yd)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
c_linear_departure = c.copy()
time_linear_departure = time.time() - st

################################################################################
# Semi-Lagrangian arrival point method

print('Testing Arrival Method')

st = time.time()
t = 0.0
c = c0.copy()
while t < max_time-1e-10:
	# get velocity fields and gradients
	u = u_function(x, y, t)
	v = v_function(x, y, t)
	ux, uy = gradient(u)
	vx, vy = gradient(v)
	# get arrival points
	xn = x + dt*u
	yn = y + dt*v
	# compute weights
	Jxx = 1 + dt*ux
	Jxy = dt*uy
	Jyx = dt*vx
	Jyy = 1 + dt*vy
	det = Jxx*Jyy - Jxy*Jyx
	cw = c*det
	ch = np.zeros([n,n], dtype=complex, order='F')
	finufftpy.nufft2d1(xn, yn, cw.astype(complex), -1, 1e-14, n, n, ch, modeord=1)
	c = np.fft.ifft2(ch).real
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
c_arrival = c.copy()
time_arrival = time.time() - st

################################################################################
# Evaluate

print('\n')
print('Err, true vs. euler    {:0.1e}'.format(np.abs(c_true-c_eulerian).max()))
print('Err, eul vs. dep       {:0.1e}'.format(np.abs(c_true-c_nonlinear_departure).max()))
print('Err, true vs. lin dep  {:0.1e}'.format(np.abs(c_true-c_linear_departure).max()))
print('Err, true vs. arrival  {:0.1e}'.format(np.abs(c_true-c_arrival).max()))

print('')
print('Time, eulerian        {:0.1f}'.format(time_eulerian*1000))
print('Time, departure       {:0.1f}'.format(time_nonlinear_departure*1000))
print('Time, linear dep      {:0.1f}'.format(time_linear_departure*1000))
print('Time, arrival         {:0.1f}'.format(time_arrival*1000))

