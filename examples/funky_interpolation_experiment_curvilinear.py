import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.utilities import affine_transformation
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from near_finder.coordinate_routines import compute_local_coordinates
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 200
ng = int(nb)
M = 8
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
# construct a grid
grid = Grid([0.0, 2*np.pi], ng, [0.0, 2*np.pi], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, interior, M, grid.xh*0.75, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid, verbose=True)

################################################################################
# Test interpolation on "shifted" region

# coordinates for curvilinear region
x = grid.xg
y = grid.yg
xr = ebdy.radial_x
yr = ebdy.radial_y

# generate a velocity field
u_func = lambda x, y: np.sin(x)*np.cos(y)
v_func = lambda x, y: -np.cos(x)*np.sin(y)
u = u_func(x, y)
v = v_func(x, y)
ur = u_func(xr, yr)
vr = v_func(xr, yr)

# move xr, yr with the velocity field
dt = 0.01
xr2 = xr + dt*ur
yr2 = yr + dt*vr

# get the gradients of this
kxv = np.fft.fftfreq(grid.shape[0], grid.xh/(2*np.pi))
kyv = np.fft.fftfreq(grid.shape[1], grid.yh/(2*np.pi))
kx, ky = np.meshgrid(kxv, kyv, indexing='ij')
dx = lambda f: fourier(f, 1j*kx)
dy = lambda f: fourier(f, 1j*ky)
ux, uy, uxr, uyr = ebdy.gradient(u, ur, dx, dy)
vx, vy, vxr, vyr = ebdy.gradient(v, vr, dx, dy)
Jxx = 1+uxr
Jxy = uyr
Jyx = vxr
Jyy = 1+vyr
det = Jxx*Jyy - Jxy*Jyx

# get a function c
c = c_func(xr, yr)

# evaluate c at these new positions
c2 = c_func(xr2, yr2)

# get r, t coordinates for xr2, yr2
t, r = compute_local_coordinates(bdy.x, bdy.y, xr2, yr2, 1e-14, verbose=True)
lb = -ebdy.radial_width
ub = 0.0
starb = affine_transformation(r, lb, ub, 1.0, -1.0, use_numexpr=True)
rtransf = np.arccos(starb)

# interpolate from c2 back to (xr, yr) and check against c
c2w = c2*det
c2e = np.row_stack([c2w, c2w[::-1]])
ch = np.zeros([2*M,nb], dtype=complex, order='F')
finufftpy.nufft2d1(xr2, yr2, c2e.astype(complex), -1, 1e-14, 2*M, nb, ch, modeord=1)



cw = cc*det
ch = np.zeros([n,n], dtype=complex, order='F')
finufftpy.nufft2d1(xx, yy, cw.astype(complex), -1, 1e-14, n, n, ch, modeord=1)
cr = np.fft.ifft2(ch).
real




self.interpolation_hold[:self.M,:] = fr
self.interpolation_hold[self.M:,:] = fr[::-1]
funch = np.fft.fft2(self.interpolation_hold)*self.interpolation_modifier
funch[self.M] = 0.0
out = np.zeros(self.grid_ia_r.shape[0], dtype=complex)
diagnostic = finufftpy.nufft2d2(self.grid_ia_r_transf, self.grid_ia_t, out, 1, 1e-14, funch, modeord=1)
vals = out.real/np.prod(funch.shape)
if f is not None: f[self.grid_in_annulus] = vals



