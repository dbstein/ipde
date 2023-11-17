import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.stokes import StokesSolver
from qfs.stokes_qfs import Stokes_QFS
from personal_utilities.arc_length_reparametrization import arc_length_parameterize

star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form

from pyfmmlib2d import SFMM
def Layer_Apply(src, trg, f):
    s = src.get_stacked_boundary()
    t = trg.get_stacked_boundary()
    out = SFMM(source=s, target=t, forces=f*src.weights, compute_target_velocity=True, compute_target_stress=True)
    u = out['target']['u']
    v = out['target']['v']
    p = out['target']['p']
    return u, v, p

ns =     [100,      200,      300,      400,      500,      600,      700,      800,      900,      1000     ]
# velocity errors with no pressure correction (M= 4 int(nb/100) )
errs_U = [1.09e-02, 1.91e-04, 6.65e-07, 7.98e-09, 9.74e-11, 1.52e-12, 6.73e-13, 9.77e-13, 1.23e-12, 1.38e-12 ]
errs_p = [3.34e-01, 4.20e-02, 6.57e-05, 1.27e-06, 1.83e-08, 2.08e-10, 2.95e-10, 3.42e-10, 7.09e-10, 9.36e-10 ]
iters  = [      23,       28,       29,       30,       31,       24,       20,       18,       16,       15 ]

nb = 900
ng = int(nb/2)
M = 4*int(nb/100)
M = max(4, M)
M = min(20, M)
pad_zone = 0
verbose = True
plot = True
reparametrize = True
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral
coordinate_tolerance = 1e-14
coordinate_scheme = 'nufft'

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
if reparametrize:
    bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct a grid
grid = Grid([-np.pi/2, np.pi/2], ng, [-np.pi/2, np.pi/2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, grid.xh*0.75, pad_zone=pad_zone, heaviside=MOL.step, coordinate_tolerance=coordinate_tolerance, coordinate_scheme=coordinate_scheme)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
ebdyc.register_grid(grid)
# give ebdyc a bumpy function
ebdyc.ready_bump(MOL.bump, (np.pi/2-ebdy.radial_width, np.pi/2-ebdy.radial_width), ebdyc[0].radial_width)

################################################################################
# Extract radial information from ebdy and construct annular solver

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
a = 3.0
b = 2.0
p_a = 2.0
p_b = 1.0
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
fu = EmbeddedFunction(ebdyc, function=fu_function)
fv = EmbeddedFunction(ebdyc, function=fv_function)
ua = EmbeddedFunction(ebdyc, function=u_function)
va = EmbeddedFunction(ebdyc, function=v_function)
pa = EmbeddedFunction(ebdyc, function=p_function)
bcu = u_function(ebdyc[0].bdy.x, ebdyc[0].bdy.y)
bcv = v_function(ebdyc[0].bdy.x, ebdyc[0].bdy.y)

# setup the solver
solver = StokesSolver(ebdyc, solver_type=solver_type)
u, v, p = solver(fu, fv, tol=1e-14, verbose=verbose, maxiter=200, restart=50)

# this isn't correct yet because we haven't applied boundary conditions
def Stokes_Pressure_Fix(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x
    Nxy = trg.normal_x[:,None]*src.normal_y
    Nyx = trg.normal_y[:,None]*src.normal_x
    Nyy = trg.normal_y[:,None]*src.normal_y
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
    return NN
A = Stokes_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(2*bdy.N) + Stokes_Pressure_Fix(bdy, bdy)
bu = solver.get_boundary_values(u)
bv = solver.get_boundary_values(v)
buc = np.concatenate([bu, bv])
bc = np.concatenate([bcu, bcv])
tau = np.linalg.solve(A, bc-buc)

qfs = Stokes_QFS(bdy, True, False, True)
sigma = qfs([tau,])
out = Layer_Apply(qfs.source, ebdyc.grid_and_radial_pts, sigma.reshape(2, sigma.size//2))
u += out[0]
v += out[1]
p += out[2]

# normalize p/pa
area = EmbeddedFunction(ebdyc, function = lambda x, y: np.ones_like(x)).integrate()
pa_mean = pa.integrate() / area
p_mean = p.integrate() / area
pa -= pa_mean
p -= p_mean

# compute the error
u_err = np.abs(ua-u)
v_err = np.abs(va-v)
p_err = np.abs(pa-p)

print('Error, u:    {:0.2e}'.format(u_err.max()))
print('Error, v:    {:0.2e}'.format(v_err.max()))
print('Error, U:    {:0.2e}'.format(max(u_err.max(), v_err.max())))
print('Error, p:    {:0.2e}'.format(p_err.max()))

if plot:
    fig, ax = plt.subplots()
    clf = (u_err+1e-15).plot(ax, norm=mpl.colors.LogNorm())
    ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
    ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
    plt.colorbar(clf)

    fig, ax = plt.subplots()
    clf = (v_err+1e-15).plot(ax, norm=mpl.colors.LogNorm())
    ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
    ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
    plt.colorbar(clf)

    fig, ax = plt.subplots()
    clf = (p_err+1e-15).plot(ax, norm=mpl.colors.LogNorm())
    ax.plot(ebdy.bdy.x, ebdy.bdy.y, color='black', linewidth=3)
    ax.plot(ebdy.interface.x, ebdy.interface.y, color='white', linewidth=3)
    plt.colorbar(clf)
