import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 600
M = 16
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 2*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy1 = GSB(c=star(3*nb, a=0.1, r=3, f=5))
bdy2 = GSB(c=squish(nb, x=-1.2, y=-0.7, b=0.4, rot=-np.pi/4))
bdy3 = GSB(c=star(2*nb, x=1, y=0.5, a=0.3, f=3))
if reparametrize:
	bdy1 = GSB(*arc_length_parameterize(bdy1.x, bdy1.y))
	bdy2 = GSB(*arc_length_parameterize(bdy2.x, bdy2.y))
	bdy3 = GSB(*arc_length_parameterize(bdy3.x, bdy3.y))
bh1 = bdy1.dt*bdy1.speed.min()
bh2 = bdy2.dt*bdy2.speed.min()
bh3 = bdy3.dt*bdy3.speed.min()
bh = min(bh1, bh2, bh3)
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*6.4//bh)
# construct a grid
grid = Grid([-3.0, 3.4], ng, [-3.2, 3.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
bdys = [bdy1, bdy2, bdy3]
ebdys = [EmbeddedBoundary(bdy, bdy is bdy1, M, bh, pad_zone, MOL.step) for bdy in bdys]
ebdyc = EmbeddedBoundaryCollection(ebdys)
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid, verbose=verbose)
# make some plots
if plot:
	fig, ax = plt.subplots()
	colors = ['black', 'blue', 'red', 'purple', 'purple']
	for ebdy in ebdys:
		q = ebdy.bdy_qfs
		q1 = q.interior_source_bdy if ebdy.interior else q.exterior_source_bdy
		q = ebdy.interface_qfs
		q2 = q.interior_source_bdy
		q3 = q.exterior_source_bdy
		bbs = [ebdy.bdy, ebdy.interface, q1, q2, q3]
		for bi, bb in enumerate(bbs):
			ax.plot(bb.x, bb.y, color=colors[bi])

################################################################################
# Test interpolation operations

k = 2*np.pi/(6.4)
test_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
test_func_x = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*x)*np.sin(k*y)
test_func_y = lambda x, y: k*np.exp(np.sin(k*x))*np.cos(k*y)

# Interpolation of a globally smooth function on grid to radial
f = test_func(grid.xg, grid.yg)
frs = [test_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
fev = ebdyc.interpolate_grid_to_radial(f, order=5)
fes = ebdyc.v2r(fev)
errs = [np.abs(fe-fr).max() for fe, fr in zip(fes, frs)]
err = max(errs)
print('Error in grid --> radial interpolation:    {:0.2e}'.format(err))

f *= ebdyc.phys

# Interpolation of a function to the interface (3rd order)
frs = [test_func(ebdy.interface.x, ebdy.interface.y) for ebdy in ebdys]
fev = ebdyc.interpolate_grid_to_interface(f, order=3, cutoff=False)
fes = ebdyc.v2l(fev)
errs = [np.abs(fe-fr).max() for fe, fr in zip(fes, frs)]
err = max(errs)
print('Error in grid --> interface interpolation: {:0.2e}'.format(err))

# Interpolation of a function to the interface (NUFFT)
frs = [test_func(ebdy.interface.x, ebdy.interface.y) for ebdy in ebdys]
fev = ebdyc.interpolate_grid_to_interface(f, order=np.Inf, cutoff=True)
fes = ebdyc.v2l(fev)
errs = [np.abs(fe-fr).max() for fe, fr in zip(fes, frs)]
err = max(errs)
print('Error in grid --> interface interpolation: {:0.2e}'.format(err))

# Interpolation of a function from radial to grid
frs = [test_func(ebdy.radial_x, ebdy.radial_y) for ebdy in ebdys]
fts = ebdyc.interpolate_radial_to_grid(frs)
fes = [test_func(ebdy.grid_ia_x, ebdy.grid_ia_y) for ebdy in ebdys]
errs = [np.abs(fe-ft).max() for fe, ft in zip(fes, fts)]
err = max(errs)
print('Error in radial --> grid interpolation:    {:0.2e}'.format(err))

################################################################################
# Test derivatives

# fourth order accurate gradient on whole domain
dx = lambda x: fd_x_4(x, grid.xh, periodic_fix=True)
dy = lambda x: fd_y_4(x, grid.yh, periodic_fix=True)
fxe, fye, fxres, fyres = ebdyc.gradient(f, frs, dx, dy, cutoff=False)
fxt = test_func_x(grid.xg, grid.yg)
fyt = test_func_y(grid.xg, grid.yg)
err_x = np.abs(fxt-fxe)[ebdyc.phys].max()
err_y = np.abs(fyt-fye)[ebdyc.phys].max()
err = max(err_x, err_y)
print('Error in gradient, 4th order FD:           {:0.2e}'.format(err))

# spectrally accurate gradient on whole domain
kxv = np.fft.fftfreq(grid.Nx, grid.xh/(2*np.pi))
kyv = np.fft.fftfreq(grid.Ny, grid.yh/(2*np.pi))
kx, ky = np.meshgrid(kxv, kyv, indexing='ij')
ikx, iky = 1j*kx, 1j*ky
dx = lambda x: fourier(x, ikx)
dy = lambda x: fourier(x, iky)
fxe, fye, fxres, fyres = ebdyc.gradient(f, frs, dx, dy, cutoff=True)
err_x = np.abs(fxt-fxe)[ebdyc.phys].max()
err_y = np.abs(fyt-fye)[ebdyc.phys].max()
err = max(err_x, err_y)
print('Error in gradient, Fourier:                {:0.2e}'.format(err))
