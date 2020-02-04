import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 200
ng = int(nb/2)
M = 16
pad_zone = 4
interior = False
slepian_r = 20
reparametrize = False

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
# reparametrize if reparametrizing
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
# construct a grid
grid = Grid([-1.5, 1.5], ng, [-1.5, 1.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, interior, M, grid.xh*0.75, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid, verbose=True)

################################################################################
# Make basic plots

fig, ax = plt.subplots()
ax.pcolormesh(grid.xg, grid.yg, ebdy.phys)
ax.scatter(bdy.x, bdy.y, color='white', s=20)
ax.set_title('Phys')

fig, ax = plt.subplots()
ax.pcolormesh(grid.xg, grid.yg, ebdy.grid_in_annulus)
ax.scatter(bdy.x, bdy.y, color='white', s=20)
ax.set_title('In Annulus')

fig, ax = plt.subplots()
ax.pcolormesh(grid.xg, grid.yg, ebdy.grid_step)
ax.scatter(bdy.x, bdy.y, color='white', s=20)
ax.set_title('Heaviside')

fig, ax = plt.subplots()
ax.scatter(ebdy.radial_x, ebdy.radial_y, color='blue', s=10, label='special coordinates')
ax.scatter(ebdy.bdy.x, ebdy.bdy.y, color='black', s=10, label='boundary')
ax.scatter(ebdy.interface.x, ebdy.interface.y, color='gray', s=10, label='interface')
ax.legend()
ax.set_title('Special Coordinates')

################################################################################
# Test interpolation operations

k = 2*np.pi/3
test_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
test_func_x = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*x)*np.sin(k*y)
test_func_y = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)

# Interpolation of a globally smooth function on grid to radial
f = test_func(grid.xg, grid.yg)
fr = test_func(ebdy.radial_x, ebdy.radial_y)
fe = ebdy.interpolate_grid_to_radial(f, order=5)
err = np.abs(fe-fr).max()
print('Error in grid --> radial interpolation:    {:0.2e}'.format(err))

# Interpolation of a function to the interface
fr = test_func(ebdy.interface.x, ebdy.interface.y)
fe = ebdy.interpolate_grid_to_interface(f, order=5)
err = np.abs(fe-fr).max()
print('Error in grid --> interface interpolation: {:0.2e}'.format(err))

# Interpolation of a function from radial to grid
fr = test_func(ebdy.radial_x, ebdy.radial_y)
ft = ebdy.interpolate_radial_to_grid(fr)
fe = test_func(ebdy.grid_ia_x, ebdy.grid_ia_y)
err = np.abs(fe-ft).max()
print('Error in radial --> grid interpolation:    {:0.2e}'.format(err))

################################################################################
# Test derivatives

# radial gradient
frxe, frye = ebdy.radial_grid_derivatives(fr)
frxt = test_func_x(ebdy.radial_x, ebdy.radial_y)
fryt = test_func_y(ebdy.radial_x, ebdy.radial_y)
err_x = np.abs(frxt-frxe).max()
err_y = np.abs(fryt-frye).max()
err = max(err_x, err_y)
print('Error in radial grid differentiation:      {:0.2e}'.format(err))

# fourth order accurate gradient on whole domain
dx = lambda x: fd_x_4(x, grid.xh, periodic_fix=not interior)
dy = lambda x: fd_y_4(x, grid.yh, periodic_fix=not interior)
fxe, fye, fxre, fyre = ebdy.gradient(f, fr, dx, dy)
fxt = test_func_x(grid.xg, grid.yg)
fyt = test_func_y(grid.xg, grid.yg)
err_x = np.abs(fxt-fxe)[ebdy.phys].max()
err_y = np.abs(fyt-fye)[ebdy.phys].max()
err = max(err_x, err_y)
print('Error in gradient, 4th order FD:           {:0.2e}'.format(err))

# spectrally accurate gradient on whole domain
kxv = np.fft.fftfreq(grid.Nx, grid.xh/(2*np.pi))
kyv = np.fft.fftfreq(grid.Ny, grid.yh/(2*np.pi))
kx, ky = np.meshgrid(kxv, kyv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky
dx = lambda x: fourier(x, ikx)
dy = lambda x: fourier(x, iky)
fxe, fye, fxre, fyre = ebdy.gradient(f, fr, dx, dy)
err_x = np.abs(fxt-fxe)[ebdy.phys].max()
err_y = np.abs(fyt-fye)[ebdy.phys].max()
err = max(err_x, err_y)
print('Error in gradient, Fourier:                {:0.2e}'.format(err))

################################################################################
# Plot QFS Boundaries

fig, ax = plt.subplots()
ax.scatter(ebdy.bdy.x, ebdy.bdy.y, color='black', s=10, label='boundary')
ax.scatter(ebdy.interface.x, ebdy.interface.y, color='gray', s=10, label='interface')
bb = ebdy.bdy_qfs.interior_source_bdy if interior else ebdy.bdy_qfs.exterior_source_bdy
ax.scatter(bb.x, bb.y, color='blue', s=10, label='boundary effective')
bb = ebdy.interface_qfs.exterior_source_bdy
ax.scatter(bb.x, bb.y, color='red', s=10, label='interface effective 1')
bb = ebdy.interface_qfs.interior_source_bdy
ax.scatter(bb.x, bb.y, color='pink', s=10, label='interface effective 2')
ax.legend()
ax.set_title('QFS Boundaries')



