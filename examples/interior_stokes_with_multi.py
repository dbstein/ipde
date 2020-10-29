import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.multi_boundary.stokes import StokesSolver
from qfs.two_d_qfs import QFS_Evaluator, QFS_Evaluator_Pressure
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(2*src.N)
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
Naive_DLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifdipole=True)

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
# velocity errors with no pressure correction
errs_u = [1.09e-02, 4.77e-05, 6.65e-07, 7.98e-09, 9.75e-11, 1.27e-12, 2.13e-12, 2.58e-12, 2.59e-12, 1.87e-12 ]
# velocity/pressure errors with pressure correction
errs_u = [1.09e-02, 4.77e-05, 6.65e-07, 7.99e-09, 9.74e-11, 1.91e-12, 4.02e-11, 1.86e-11, 2.58e-12, 1.29e-10 ]
errs_p = [3.17e-01, 3.15e-03, 6.55e-05, 1.26e-06, 1.83e-08, 3.25e-10, 3.45e-09, 3.62e-09, 5.55e-10, 1.58e-08 ]

nb = 400
ng = int(nb/2)
M = 4*int(nb/100)
M = max(4, M)
M = min(30, M)
pad_zone = 0
verbose = True
plot = True
reparametrize = False
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral
grid_upsample = 2
fix_pressure = True

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
ebdy = EmbeddedBoundary(bdy, True, M, grid.xh*0.75, pad_zone=pad_zone, heaviside=MOL.step, qfs_tolerance=1e-14, qfs_fsuf=2)
ebdys = [ebdy,]
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
fu = EmbeddedFunction(ebdyc)
fv = EmbeddedFunction(ebdyc)
ua = EmbeddedFunction(ebdyc)
va = EmbeddedFunction(ebdyc)
pa = EmbeddedFunction(ebdyc)
fu.define_via_function(fu_function)
fv.define_via_function(fv_function)
ua.define_via_function(u_function)
va.define_via_function(v_function)
pa.define_via_function(p_function)
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
# SLP with pressure evaluation at 0th target point
def PSLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    return out
# DLP with pressure evaluation at 0th target point
def PDLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_DLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    rdotn = dx*src.normal_x + dy*src.normal_y
    ir2 = 1.0/r2
    rdotnir4 = rdotn*ir2*ir2
    out[-1, 0*src.N:1*src.N] = (-src.normal_x*ir2 + 2*rdotnir4*dx)*src.weights
    out[-1, 1*src.N:2*src.N] = (-src.normal_y*ir2 + 2*rdotnir4*dy)*src.weights
    out[-1] /= np.pi
    return out
# SLP with pressure null-space correction; fixing scale to eval at 0th target point
def Pressure_SLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N+1])
    out[:-1,:-1] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    out[0*trg.N:1*trg.N, -1] = trg.normal_x*trg.weights
    out[1*trg.N:2*trg.N, -1] = trg.normal_y*trg.weights
    return out
def Fixed_SLP(src, trg):
    return Naive_SLP(src, trg) + Stokes_Pressure_Fix(src, trg)
A = Stokes_Layer_Singular_Form(bdy, ifdipole=True) - 0.5*np.eye(2*bdy.N) + Stokes_Pressure_Fix(bdy, bdy)
bu = solver.get_boundary_values(u)
bv = solver.get_boundary_values(v)
buc = np.concatenate([bu, bv])
bc = np.concatenate([bcu, bcv])
tau = np.linalg.solve(A, bc-buc)

if fix_pressure:
    qfs = QFS_Evaluator_Pressure(ebdy.bdy_qfs, True, [PDLP,], Pressure_SLP, form_b2c=False)
    sigma = qfs([tau,])
    nsigma2 = int(sigma.size/2)
    out = Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, ebdyc.grid_and_radial_pts, sigma.reshape(2, nsigma2))
else:
    qfs = QFS_Evaluator(ebdy.bdy_qfs, True, [Singular_DLP,], Fixed_SLP, on_surface=True, form_b2c=False, vector=True)
    sigma = qfs([tau,])
    nsigma2 = int(sigma.size/2)
    out = Layer_Apply(ebdy.bdy_qfs.interior_source_bdy, ebdyc.grid_and_radial_pts, sigma.reshape(2, nsigma2))
u += out[0]
v += out[1]
p += out[2]

# normalize p/pa
pd = ebdyc[0].interpolate_radial_to_boundary(p[0])
pad = ebdyc[0].interpolate_radial_to_boundary(pa[0])
p -= np.sum(pd*bdy.weights)/np.sum(bdy.weights)
pa -= np.sum(pad*bdy.weights)/np.sum(bdy.weights)

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
