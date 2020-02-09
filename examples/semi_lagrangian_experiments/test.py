import numpy as np
import time
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.modified_helmholtz import ModifiedHelmholtzSolver
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from fast_interp import interp1d
from pybie2d.point_set import PointSet
from qfs.two_d_qfs import QFS_Evaluator
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

"""
Semi-Lagrangian Test Solver, Cleaned for benchmarking

Makes use of:
    FFTs
    NUFFTs
    FMMs
    Numba
    Linear Algebra

And pegs my macbookpro at ~800% CPU usage during the bulk of the runtime
"""

max_time          = 1.0        # time to run to
nu                = 0.1        # diffusion coefficient
nb                = 1000       # number of points in boundary
grid_upsampling   = 2          # upsampling of grid relative to boundary
radial_upsampling = 2          # how much to upsample cheb grid by (relative to background)
M                 = 30         # number of chebyshev modes in radial grids
dt                = 0.05       # timestep

# smoothness of rolloff functions
slepian_r = 1.0*M
# Define the nonlinear reaction term
nonlinearity = lambda x: x*np.cos(x)
# Scale the nonlinear reaction term
scaled_nonlinearity = lambda x: 0.5*nonlinearity(x)
# Ensure the timestep works well with max_time
tsteps = max_time / dt
if np.abs(int(np.round(tsteps)) - tsteps)/np.abs(tsteps) > 1e-14:
    raise Exception
tsteps = int(np.round(tsteps))
# Define the number of grid points
ngx = int(nb/2)*grid_upsampling
ngy = int(nb/2)*grid_upsampling

# stuff for the modified helmholtz equation (for first timestep of Backward Euler)
zeta         = nu*dt
helmholtz_k  = np.sqrt(1.0/zeta)
d_singular   = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
s_singular   = lambda src: -(d_singular(src)/src.weights).T*src.weights
Singular_SLP = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
Naive_SLP    = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
half_eye      = lambda src: np.eye(src.N)*0.5

# generate a velocity field
kk = 2*np.pi/3
u_function = lambda x, y, t:  0.5 + 0.7*np.sin(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)
v_function = lambda x, y, t:  -0.7*np.cos(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)
# generate an initial concentration field
c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)

################################################################################
# Get truth via just using Adams-Bashforth 2 on periodic domain

# generate a grid
xv,  h = np.linspace(-1.2, 1.7, ngx, endpoint=False, retstep=True)
yv, _h = np.linspace(-1.5, 1.5, ngy, endpoint=False, retstep=True)
x, y = np.meshgrid(xv, yv, indexing='ij')
# fourier modes
kvx = np.fft.fftfreq(ngx, h/(2*np.pi))
kvy = np.fft.fftfreq(ngy, h/(2*np.pi))
kvx[int(ngx/2)] = 0.0
kvy[int(ngy/2)] = 0.0
kx, ky = np.meshgrid(kvx, kvy, indexing='ij')
ikx, iky = 1j*kx, 1j*ky

# get heaviside function
MOL = SlepianMollifier(slepian_r)

# construct boundary and reparametrize
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.0, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh/radial_upsampling, pad_zone=0, heaviside=MOL.step, qfs_fsuf=4, coordinate_scheme='nufft', coordinate_tolerance=1e-12)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# get a grid
grid = Grid([-1.2, 1.7], ngx, [-1.5, 1.5], ngy, x_endpoints=[True, False], y_endpoints=[True, False])
ebdyc.register_grid(grid)

# initial c field
c0 = EmbeddedFunction(ebdyc)
c0.define_via_function(c0_function)

# now timestep
c = c0.copy()
t = 0.0

st = time.time()

while t < max_time-1e-10:

    # get the velocity fields at this time
    u = EmbeddedFunction(ebdyc)
    u.define_via_function(lambda x, y: u_function(x, y, t))
    v = EmbeddedFunction(ebdyc)
    v.define_via_function(lambda x, y: v_function(x, y, t))
    # get the velocity fields on the boundary
    ub = ebdyc.interpolate_radial_to_boundary(u).bdy_value_list[0]
    vb = ebdyc.interpolate_radial_to_boundary(v).bdy_value_list[0]

    # move the boundary with Forward Euler
    bx = ebdy.bdy.x + dt*ub
    by = ebdy.bdy.y + dt*vb
    # repararmetrize the boundary
    bx, by, new_t = arc_length_parameterize(bx, by, return_t=True)
    bu_interp = interp1d(0, 2*np.pi, bdy.dt, ub, p=True)
    bv_interp = interp1d(0, 2*np.pi, bdy.dt, vb, p=True)
    # old boundary velocity interpolated to new parametrization
    ubo_new_parm = bu_interp(new_t)
    vbo_new_parm = bv_interp(new_t)
    # old boundary velocity at old parametrization
    ubo = ub.copy()
    vbo = vb.copy()

    # take gradients of the velocity fields
    ux, uy = ebdyc.gradient2(u)
    vx, vy = ebdyc.gradient2(v)

    # now generate a new ebdy based on the moved boundary
    new_bdy = GSB(x=bx, y=by)
    bh = new_bdy.dt*new_bdy.speed.min()
    # construct embedded boundary
    new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh/radial_upsampling, pad_zone=0, heaviside=MOL.step, qfs_fsuf=4, coordinate_scheme='nufft', coordinate_tolerance=1e-12)
    new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
    umax = np.sqrt((u*u + v*v).data, where=np.logical_not(u.mask)).max()
    new_ebdyc.register_grid(grid, danger_zone_distance=2*umax*dt)

    # let's get the points that need to be interpolated to
    gp = new_ebdyc.grid_pna
    ap = new_ebdyc.radial_pts

    aax = np.concatenate([gp.x, ap.x])
    aay = np.concatenate([gp.y, ap.y])
    aap = PointSet(x=aax, y=aay)
    AP_key  = ebdy.register_points(aap.x, aap.y, danger_zone=new_ebdy.in_danger_zone, gi=new_ebdy.guess_inds)

    # now we need to interpolate onto things
    AEP = ebdy.registered_partitions[AP_key]

    # generate a holding ground for the new c
    c_new = EmbeddedFunction(new_ebdyc)
    c_new.zero()

    # get departure points
    xd_all = np.zeros(aap.N)
    yd_all = np.zeros(aap.N)

    # advect those in the annulus
    c1,  c2,  c3  = AEP.get_categories()
    c1n, c2n, c3n = AEP.get_category_Ns()
    # category 1 and 2
    c1_2 = np.logical_or(c1, c2)
    c1_2n = c1n + c2n
    uxh = ebdy.interpolate_to_points(ux, aap.x, aap.y)
    uyh = ebdy.interpolate_to_points(uy, aap.x, aap.y)
    vxh = ebdy.interpolate_to_points(vx, aap.x, aap.y)
    vyh = ebdy.interpolate_to_points(vy, aap.x, aap.y)
    uh = ebdy.interpolate_to_points(u, aap.x, aap.y)
    vh = ebdy.interpolate_to_points(v, aap.x, aap.y)
    SLM = np.zeros([c1_2n,] + [2,2], dtype=float)
    SLR = np.zeros([c1_2n,] + [2,], dtype=float)
    SLM[:,0,0] = 1 + dt*uxh[c1_2]
    SLM[:,0,1] = dt*uyh[c1_2]
    SLM[:,1,0] = dt*vxh[c1_2]
    SLM[:,1,1] = 1 + dt*vyh[c1_2]
    SLR[:,0] = dt*uh[c1_2]
    SLR[:,1] = dt*vh[c1_2]
    OUT = np.linalg.solve(SLM, SLR)
    xdt, ydt = OUT[:,0], OUT[:,1]
    xd, yd = aap.x[c1_2] - xdt, aap.y[c1_2] - ydt
    # we now back check xd, yd to make sure they're actually in ebdy!
    test_key = ebdy.register_points(xd, yd, danger_zone=new_ebdy.in_danger_zone[c1_2], gi=new_ebdy.guess_inds[c1_2])
    test_part = ebdy.registered_partitions[test_key]
    test1, test2, test3 = test_part.get_categories()
    # reclassify things that are in test3 as being in c3
    c3[c1_2] = test3
    # recount c3
    c3n = c3.sum()
    xd_all[c1_2] = xd
    yd_all[c1_2] = yd
    # categroy 3... this is the tricky one
    if c3n > 0:
        th = 2*np.pi/nb
        tk = np.fft.fftfreq(nb, th/(2*np.pi))
        def d1_der(f):
            return np.fft.ifft(np.fft.fft(f)*tk*1j).real
        interp = lambda f: interp1d(0, 2*np.pi, th, f, k=3, p=True)
        bx_interp  = interp(ebdy.bdy.x)
        by_interp  = interp(ebdy.bdy.y)
        bxs_interp = interp(d1_der(ebdy.bdy.x))
        bys_interp = interp(d1_der(ebdy.bdy.y))
        nx_interp  = interp(ebdy.bdy.normal_x)
        ny_interp  = interp(ebdy.bdy.normal_y)
        nxs_interp = interp(d1_der(ebdy.bdy.normal_x))
        nys_interp = interp(d1_der(ebdy.bdy.normal_y))
        urb = ebdy.interpolate_radial_to_boundary_normal_derivative(u.radial_value_list[0])
        vrb = ebdy.interpolate_radial_to_boundary_normal_derivative(v.radial_value_list[0])
        ub_interp   = interp(ub)
        vb_interp   = interp(vb)
        urb_interp  = interp(urb)
        vrb_interp  = interp(vrb)
        ubs_interp  = interp(d1_der(ub))
        vbs_interp  = interp(d1_der(vb))
        urbs_interp = interp(d1_der(urb))
        vrbs_interp = interp(d1_der(vrb))
        xo = aap.x[c3]
        yo = aap.y[c3]
        def objective(s, r):
            f = np.empty([s.size, 2])
            f[:,0] = bx_interp(s) + r*nx_interp(s) + dt*ub_interp(s) + dt*r*urb_interp(s) - xo
            f[:,1] = by_interp(s) + r*ny_interp(s) + dt*vb_interp(s) + dt*r*vrb_interp(s) - yo
            return f
        def Jac(s, r):
            J = np.empty([s.size, 2, 2])
            J[:,0,0] = bxs_interp(s) + r*nxs_interp(s) + dt*ubs_interp(s) + dt*r*urbs_interp(s)
            J[:,1,0] = bys_interp(s) + r*nys_interp(s) + dt*vbs_interp(s) + dt*r*vrbs_interp(s)
            J[:,0,1] = nx_interp(s) + dt*urb_interp(s)
            J[:,1,1] = ny_interp(s) + dt*vrb_interp(s)
            return J
        # guess inds for Newton solver
        s =   AEP.full_t[c3]
        r =   AEP.full_r[c3]
        # now solve for sd, rd
        res = objective(s, r)
        mres = np.hypot(res[:,0], res[:,1]).max()
        tol = 1e-12
        while mres > tol:
            J = Jac(s, r)
            d = np.linalg.solve(J, res)
            s -= d[:,0]
            r -= d[:,1]
            res = objective(s, r)
            mres = np.hypot(res[:,0], res[:,1]).max()
        # get the departure points
        xd = bx_interp(s) + nx_interp(s)*r
        yd = by_interp(s) + ny_interp(s)*r
        xd_all[c3] = xd
        yd_all[c3] = yd
    # now interpolate to c
    ch = ebdy.interpolate_to_points(c - dt*scaled_nonlinearity(c), xd_all, yd_all, danger_zone=new_ebdy.in_danger_zone, gi=new_ebdy.guess_inds)
    # set the grid values
    c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
    # set the radial values
    c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
    # overwrite under grid under annulus by radial grid
    _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)

    # now solve diffusion equation
    solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type='spectral', k=helmholtz_k)
    c_new_update = solver(c_new/zeta, tol=1e-12, verbose=False)
    A = s_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
    bvn = solver.get_boundary_normal_derivatives(c_new_update.radial_value_list)
    tau = np.linalg.solve(A, bvn)
    qfs = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP,], Naive_SLP, on_surface=True, form_b2c=False)
    sigma = qfs([tau,])
    out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k)
    gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
    rslp = rslpl[0]
    c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
    c_new_update.grid_value[new_ebdyc.phys] += gslp
    c_new = c_new_update
    # overwrite under grid under annulus by radial grid
    _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)

    ebdy = new_ebdy
    ebdyc = new_ebdyc
    c = c_new

    t += dt
    print('   t = {:0.3f}'.format(t), 'of', max_time)

total_time = time.time() - st
timestep_time = total_time / tsteps

print('Total simulation took: {:0.1f}'.format(total_time),    'seconds.')
print('Time per timestep:     {:0.1f}'.format(timestep_time), 'seconds.')

import matplotlib as mpl
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
clf = c.plot(ax)
plt.colorbar(clf)

