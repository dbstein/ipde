import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ipde.annular.modified_helmholtz import AnnularModifiedHelmholtzSolver
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 400
ng = int(nb/2)
M = 12
helmholtz_k = 10.0
pad_zone = 2
interior = True
slepian_r = 20

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
# construct a grid
grid = Grid([-1.5, 1.5], ng, [-1.5, 1.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, interior, M, grid.xh*0.75, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid)

################################################################################
# Extract radial information from ebdy and construct annular solver

# get the forces and BCs for the problem
x, y = ebdy.radial_x, ebdy.radial_y
k = 2*np.pi/3
solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y)-k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
force = force_func(x, y)
asol = solution_func(x, y)
ibc = solution_func(ebdy.interface.x, ebdy.interface.y)
ubc = solution_func(ebdy.bdy.x, ebdy.bdy.y)
if not interior: temp = ibc; ibc = ubc; ubc = temp

# get an approximate geometry
print('Generating preconditioner')
AAG = ApproximateAnnularGeometry(nb, M, ebdy.radial_width, ebdy.approximate_radius)
# construct the preconditioner (for Dirichlet BC)
APS = AnnularModifiedHelmholtzSolver(AAG, 10.0)
print('Solving')
# get the real geometry
sp =  ebdy.bdy.speed     if interior else ebdy.interface.speed
cur = ebdy.bdy.curvature if interior else ebdy.interface.curvature
RAG = RealAnnularGeometry(sp, cur, AAG)
# solve
esol = APS.solve(RAG, force, ibc, ubc, verbose=True, tol=1e-14)
# get the errors
err = np.abs(esol-asol).max()
print('\nError is: {:0.2e}'.format(err))
# plot the errors
fig, ax = plt.subplots()
clf = ax.pcolormesh(x, y, np.abs(esol-asol), norm=mpl.colors.LogNorm())
ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', label='boundary', linewidth=3)
ax.plot(ebdy.interface.x, ebdy.interface.y, color='gray', label='interface', linewidth=3)
plt.colorbar(clf)
ax.legend()
ax.set_title('Error')

