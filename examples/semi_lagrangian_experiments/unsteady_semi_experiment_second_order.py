import numpy as np
import finufftpy
import time

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 0.5
# set timestep
dt = 0.1/2/2/2/2/2/10
# discretization size
n = 64

# generate a velocity field (assume steady in time)
u_function = lambda x, y, t: np.sin(x)*np.cos(y)*(1+np.cos(2*np.pi*t))
v_function = lambda x, y, t: np.sin(x)*np.cos(y)*(1+np.cos(2*np.pi*t))

# gradient function
def gradient(f):
	fh = np.fft.fft2(f)
	fx = np.fft.ifft2(fh*ikx).real
	fy = np.fft.ifft2(fh*iky).real
	return fx, fy

################################################################################
# Forward-Euler, Small Timestep, double grid for "Truth"

adj = 2

n *= adj
dt /= (2*adj)

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
	if t == 0:
		f = u*cx + v*cy
		c -= dt*f
	else:
		f = u*cx + v*cy
		c -= dt*(1.5*f-0.5*f_old)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
	f_old = f
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
	if t == 0:
		f = u*cx + v*cy
		c -= dt*f
	else:
		f = u*cx + v*cy
		c -= dt*(1.5*f-0.5*f_old)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
	f_old = f
c_eulerian = c.copy()
time_eulerian = time.time() - st

################################################################################
# Semi-Lagrangian Linear departure point method

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
	if t == 0:
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
		# make this bigger for all following timesteps
		SLM = np.zeros([n**2,] + [4,4], dtype=float)
		SLR = np.zeros([n**2,] + [4,], dtype=float)
	else:
		# solve for departure points
		SLM[:,0,0] = ux.ravel()
		SLM[:,0,1] = uy.ravel()
		SLM[:,0,2] = 0.5/dt
		SLM[:,0,3] = 0.0
		SLM[:,1,0] = vx.ravel()
		SLM[:,1,1] = vy.ravel()
		SLM[:,1,2] = 0.0
		SLM[:,1,3] = 0.5/dt
		SLM[:,2,0] = 2.0/dt + 3*ux.ravel()
		SLM[:,2,1] = 3*uy.ravel()
		SLM[:,2,2] = -uxo.ravel()
		SLM[:,2,3] = -uyo.ravel()
		SLM[:,3,0] = 3*vx.ravel()
		SLM[:,3,1] = 2.0/dt + 3*vy.ravel()
		SLM[:,3,2] = -vxo.ravel()
		SLM[:,3,3] = -vyo.ravel()
		SLR[:,0] = u.ravel()
		SLR[:,1] = v.ravel()
		SLR[:,2] = 3*u.ravel() - uo.ravel()
		SLR[:,3] = 3*v.ravel() - vo.ravel()
		OUT = np.linalg.solve(SLM, SLR)
		dx, dy, Dx, Dy = OUT[:,0].reshape(n,n), OUT[:,1].reshape(n,n), \
								OUT[:,2].reshape(n,n), OUT[:,3].reshape(n,n)
		xd, yd = x - dx, y - dy
	# interpolate onto departure points
	c = c_interpolater(xd, yd)
	t += dt
	print('   t = {:0.3f}'.format(t), max_time, '\r', end='')
	# save old things
	uo = u.copy()
	vo = v.copy()
	uxo = ux.copy()
	vxo = vx.copy()
	uyo = uy.copy()
	vyo = vy.copy()
c_linear_departure = c.copy()
time_linear_departure = time.time() - st

################################################################################
# Evaluate

print('\n')
print('Err, true vs. euler   {:0.1e}'.format(np.abs(c_true-c_eulerian).max()))
print('Err, true vs. lin dep {:0.1e}'.format(np.abs(c_true-c_linear_departure).max()))

print('')
print('Time, eulerian        {:0.1f}'.format(time_eulerian*1000))
print('Time, linear dep      {:0.1f}'.format(time_linear_departure*1000))

################################################################################
# Evaluate

print('\n')
print('Err, true vs. euler   {:0.1e}'.format(np.abs(c_true-c_eulerian).max()))
print('Err, true vs. lin dep {:0.1e}'.format(np.abs(c_true-c_linear_departure).max()))

print('')
print('Time, eulerian        {:0.1f}'.format(time_eulerian*1000))
print('Time, linear dep      {:0.1f}'.format(time_linear_departure*1000))

