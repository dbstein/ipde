import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
from ipde.annular.stokes import AnnularStokesSolver
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 600
ng = int(nb/2)
M = 12
pad_zone = 2
interior = True
slepian_r = 10

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.15, f=5))
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
# k = 2*np.pi/3

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
a = 2.0
b = 1.0
p_a = 1.0
p_b = 2.0
sin = np.sin
cos = np.cos
esin = lambda x: np.exp(sin(x))
psix = lambda x, y: esin(a*x)*cos(b*y)
psiy = lambda x, y: esin(a*x)*sin(b*y)
u_function  = lambda x, y: psix(x, y)
v_function  = lambda x, y: -a/b*cos(a*x)*psiy(x, y)
p_function  = lambda x, y: cos(p_a*x) + esin(p_b*y)
fu_function = lambda x, y: (a**2*(sin(a*x)-cos(a*x)**2) + b**2)*psix(x, y) - p_a*sin(p_a*x)
fv_function = lambda x, y: -a*b*cos(a*x)*psiy(x, y)*(1 + (a/b)**2*sin(a*x)*(3+sin(a*x))) + p_b*cos(p_b*y)*esin(p_b*y)
fu_radial = fu_function(ebdy.radial_x, ebdy.radial_y)
fv_radial = fv_function(ebdy.radial_x, ebdy.radial_y)
ua_radial =  u_function(ebdy.radial_x, ebdy.radial_y)
va_radial =  v_function(ebdy.radial_x, ebdy.radial_y)
pa_radial =  p_function(ebdy.radial_x, ebdy.radial_y)
pa_radial -= np.mean(pa_radial)
lower_u = u_function(ebdy.interface.x, ebdy.interface.y)
lower_v = v_function(ebdy.interface.x, ebdy.interface.y)
upper_u = u_function(ebdy.bdy.x, ebdy.bdy.y)
upper_v = v_function(ebdy.bdy.x, ebdy.bdy.y)
if not interior: temp = lower_u; lower_u = upper_u; upper_u = temp
if not interior: temp = lower_v; lower_v = upper_v; upper_v = temp

# get an approximate geometry
print('Generating preconditioner')
AAG = ApproximateAnnularGeometry(nb, M, ebdy.radial_width, ebdy.approximate_radius)
# construct the preconditioner (for Dirichlet BC)
solver = AnnularStokesSolver(AAG, 1.0)
print('Solving')
# get the real geometry
sp =  ebdy.bdy.speed     if interior else ebdy.interface.speed
cur = ebdy.bdy.curvature if interior else ebdy.interface.curvature
RAG = RealAnnularGeometry(sp, cur, AAG)
# convert (u,v) to (r,t)
fr, ft = ebdy.convert_uv_to_rt(fu_radial, fv_radial)
lr, lt = ebdy.convert_uv_to_rt(lower_u, lower_v)
ur, ut = ebdy.convert_uv_to_rt(upper_u, upper_v)
re_radial, te_radial, pe_radial = solver.solve(RAG, fr, ft, lr, lt, ur, ut, verbose=True, tol=1e-12)
ue_radial, ve_radial = ebdy.convert_rt_to_uv(re_radial, te_radial)
# get error
error_u = np.abs(ue_radial-ua_radial).max()
error_v = np.abs(ve_radial-va_radial).max()
error_p = np.abs(pe_radial-pa_radial).max()
print('      Error, u: {:0.2e}'.format(error_u))
print('      Error, v: {:0.2e}'.format(error_v))
print('      Error, p: {:0.2e}'.format(error_p))

# plot the errors
fig, ax = plt.subplots()
clf = ax.pcolormesh(ebdy.radial_x, ebdy.radial_y, np.abs(ue_radial-ua_radial), norm=mpl.colors.LogNorm())
ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', label='boundary', linewidth=3)
ax.plot(ebdy.interface.x, ebdy.interface.y, color='gray', label='interface', linewidth=3)
plt.colorbar(clf)
ax.legend()
ax.set_title('Error, u')

fig, ax = plt.subplots()
clf = ax.pcolormesh(ebdy.radial_x, ebdy.radial_y, np.abs(ve_radial-va_radial), norm=mpl.colors.LogNorm())
ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', label='boundary', linewidth=3)
ax.plot(ebdy.interface.x, ebdy.interface.y, color='gray', label='interface', linewidth=3)
plt.colorbar(clf)
ax.legend()
ax.set_title('Error, v')

fig, ax = plt.subplots()
clf = ax.pcolormesh(ebdy.radial_x, ebdy.radial_y, np.abs(pe_radial-pa_radial), norm=mpl.colors.LogNorm())
ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', label='boundary', linewidth=3)
ax.plot(ebdy.interface.x, ebdy.interface.y, color='gray', label='interface', linewidth=3)
plt.colorbar(clf)
ax.legend()
ax.set_title('Error, p')
