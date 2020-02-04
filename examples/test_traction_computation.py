import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from ipde.solvers.internals.vector import VectorHelper
from ipde.solvers.single_boundary.interior.stokes import StokesSolver
from qfs.two_d_qfs import QFS_Evaluator
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*np.eye(2*src.N)
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)

nb = 800
M = 16
pad_zone = 0
verbose = False
plot = True
reparametrize = True
slepian_r = 1.5*M
solver_type = 'spectral' # fourth or spectral

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
if reparametrize:
	bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# get number of gridpoints to roughly match boundary spacing
ng = 2*int(0.5*2.4//bh)
# construct a grid
grid = Grid([-1.2, 1.2], ng, [-1.2, 1.2], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid)

################################################################################
# Extract radial information from ebdy and construct annular solver

# Testing the radial Stokes solver
print('   Testing Radial Stokes Solver')
# choose a, b so that things are periodic
w = 2.4
kk = 2*np.pi/w
a = 3*kk
b = kk
p_a = 2*kk
p_b = 5*kk
sin = np.sin
cos = np.cos
esin = lambda x: np.exp(sin(x))
esin_x = lambda x: np.exp(sin(x))*np.cos(x)
psix = lambda x, y: esin(a*x)*cos(b*y)
psiy = lambda x, y: esin(a*x)*sin(b*y)
psix_x = lambda x, y: a*esin_x(a*x)*cos(b*y)
psix_y = lambda x, y: -b*esin(a*x)*sin(b*y)
psiy_x = lambda x, y: a*esin_x(a*x)*sin(b*y)
psiy_y = lambda x, y: b*esin(a*x)*cos(b*y)
u_function  = lambda x, y: psix(x, y)
v_function  = lambda x, y: -a/b*cos(a*x)*psiy(x, y)
p_function  = lambda x, y: cos(p_a*x) + esin(p_b*y)
ux_function = lambda x, y: psix_x(x, y)
uy_function = lambda x, y: psix_y(x, y)
vx_function = lambda x, y: a**2/b*sin(a*x)*psiy(x, y) - a/b*cos(a*x)*psiy_x(x, y)
vy_function = lambda x, y: -a/b*cos(a*x)*psiy_y(x, y)

txx_function = lambda x, y: 2*ux_function(x, y) - p_function(x, y)
txy_function = lambda x, y: uy_function(x, y) + vx_function(x, y)
tyy_function = lambda x, y: 2*vy_function(x, y) - p_function(x, y)

eval_on_grid      = lambda func: func(grid.xg, grid.yg)
eval_on_radial    = lambda func: func(ebdy.radial_x, ebdy.radial_y)
eval_on_bdy       = lambda func: func(ebdy.bdy.x, ebdy.bdy.y)
eval_on_interface = lambda func: func(ebdy.interface.x, ebdy.interface.y)
multi_eval = lambda meta_func, funcs: [meta_func(func) for func in funcs]

# compute everything on the background and radial grids
# and bdy and interfaces
main_funcs = [u_function, v_function, p_function]
der_funcs = [ux_function, uy_function, vx_function, vy_function]
traction_funcs = [txx_function, txy_function, tyy_function]
all_funcs = main_funcs + der_funcs + traction_funcs
u,  v,  p,  ux,  uy,  vx,  vy,  txx,  txy,  tyy  = multi_eval(eval_on_grid,      all_funcs)
ru, rv, rp, rux, ruy, rvx, rvy, rtxx, rtxy, rtyy = multi_eval(eval_on_radial,    all_funcs)
bu, bv, bp, bux, buy, bvx, bvy, btxx, btxy, btyy = multi_eval(eval_on_bdy,       all_funcs)
iu, iv, ip, iux, iuy, ivx, ivy, itxx, itxy, ityy = multi_eval(eval_on_interface, all_funcs)
# calcuate traction on interace
itx = itxx*ebdy.interface.normal_x + itxy*ebdy.interface.normal_y
ity = itxy*ebdy.interface.normal_x + ityy*ebdy.interface.normal_y

# interpolate stress to the interface to get the traction
etxx = ebdy.interpolate_grid_to_interface(txx, order=np.Inf)
etxy = ebdy.interpolate_grid_to_interface(txy, order=np.Inf)
etyy = ebdy.interpolate_grid_to_interface(tyy, order=np.Inf)
etx = etxx*ebdy.interface.normal_x + etxy*ebdy.interface.normal_y
ety = etxy*ebdy.interface.normal_x + etyy*ebdy.interface.normal_y
# get error in this computation
e1 = np.abs(etx-itx).max()
e2 = np.abs(ety-ity).max()
err = max(e1, e2)
print('Error is: {:0.2e}'.format(err))

# now do this with radial coords using helper functions
solver = VectorHelper(ebdy)
etx, ety = solver.get_interface_traction_uvp(ru, rv, rp)
# get error in this computation
e1 = np.abs(etx-itx).max()
e2 = np.abs(ety-ity).max()
err = max(e1, e2)
print('Error is: {:0.2e}'.format(err))

# test out the method with the other coords
itr, itt = ebdy.convert_uv_to_rt(itx, ity)
rr, rt = ebdy.convert_uv_to_rt(ru, rv)
etr, ett = solver.get_interface_traction_rtp(rr, rt, rp)
e1 = np.abs(etr-itr).max()
e2 = np.abs(ett-itt).max()
err = max(e1, e2)
print('Error is: {:0.2e}'.format(err))

# test this on the boundary, instead
btx = btxx*ebdy.interface.normal_x + btxy*ebdy.interface.normal_y
bty = btxy*ebdy.interface.normal_x + btyy*ebdy.interface.normal_y
etx, ety = solver.get_boundary_traction_uvp(ru, rv, rp)
# get error in this computation
e1 = np.abs(etx-btx).max()
e2 = np.abs(ety-bty).max()
err = max(e1, e2)
print('Error is: {:0.2e}'.format(err))

# test out the method with the other coords
btr, btt = ebdy.convert_uv_to_rt(btx, bty)
etr, ett = solver.get_boundary_traction_rtp(rr, rt, rp)
e1 = np.abs(etr-btr).max()
e2 = np.abs(ett-btt).max()
err = max(e1, e2)
print('Error is: {:0.2e}'.format(err))

