import numpy as np
import finufftpy

"""
Test semi-lagrangian solve...
"""

# generate a grid
n = 48

v, h = np.linspace(0, 2*np.pi, n, endpoint=False, retstep=True)
x, y = np.meshgrid(v, v, indexing='ij')
# fourier modes
kv = np.fft.fftfreq(n, h/(2*np.pi))
kx, ky = np.meshgrid(kv, kv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# generate a velocity field
u = np.sin(x)*np.cos(y)
v = -np.cos(x)*np.sin(y)

# c field
c_func = lambda x, y: np.exp(np.cos(x))*np.sin(y)
c = c_func(x, y)

# compute updated positions
dt = 0.1
xx = x + dt*u
yy = y + dt*v

# evaluate c at these new positions
cc = c_func(xx, yy)

# gradient functionx
def gradient(u):
	uh = np.fft.fft2(u)
	ux = np.fft.ifft2(uh*ikx).real
	uy = np.fft.ifft2(uh*iky).real
	return ux, uy

ux, uy = gradient(u)
vx, vy = gradient(v)

# interpolate this back to x, y and assess the errors
Jxx = 1 + dt*ux
Jxy = dt*uy
Jyx = dt*vx
Jyy = 1 + dt*vy
det = Jxx*Jyy - Jxy*Jyx
cw = cc*det
ch = np.zeros([n,n], dtype=complex, order='F')
finufftpy.nufft2d1(xx, yy, cw.astype(complex), -1, 1e-14, n, n, ch, modeord=1)
cr = np.fft.ifft2(ch).real

# get error
print('Error is: {:0.2e}'.format(np.abs(cr-c).max()))



