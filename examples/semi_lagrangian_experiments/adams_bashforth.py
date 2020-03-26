import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import finufftpy
import time
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, EmbeddedFunction, BoundaryFunction
from ipde.heavisides import SlepianMollifier
from ipde.solvers.multi_boundary.modified_helmholtz import ModifiedHelmholtzSolver
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
from ipde.derivatives import fd_x_4, fd_y_4, fourier
from fast_interp import interp1d
from pybie2d.point_set import PointSet
from qfs.two_d_qfs import QFS_Evaluator
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid
MH_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
MH_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply

"""
Test semi-lagrangian solve...
"""

max_time          = 1.1        # time to run to
reaction_value    = 0.5        # value for reaction term
const_factor      = 0.5        # constant advection velocity
diff_factor       = 0.7        # non-constant advection velocity
nu                = 0.1        # diffusion coefficient
nb                = 300        # number of points in boundary
grid_upsampling   = 2          # upsampling of grid relative to boundary
radial_upsampling = 2          # how much to upsample cheb grid by (relative to background)
M                 = 24         # number of chebyshev modes
pad_zone          = 0          # padding in radial grid (typically set to 0 for spectral methods)
dt                = 0.1/2/2/2/2/2/2     # timestep
solver_type       = 'fourth'   # 'spectral' or 'fourth'
qfs_fsuf          = 4          # forced upsampling for QFS
coordinate_scheme = 'polyi'    # interpolation scheme for coordinate finding ('polyi' or 'nufft')
coordinate_tol    = 1e-8       # probablys should be set to be compatible with 
xmin              = -1.5       # domain definition
xmax              =  4.5       # domain definition
ymin              = -1.5       # domain definition
ymax              =  1.5       # domain definition
printjustment     =  40        # how much to adjust for printing
verbose           = False      # provide verbose output for radial solve
timing_marks      = True
use_danger_zone   = True
timestepper       = 'bdf'       # 'fbe' or 'bdf' or 'ab'; respectively: (forward-euler/backward-euler), (2nd order IMEX backward-differentiation formula), (2nd order adams-bashforth/crank-nicolson)
backcomputation   = True      # use back or direct computation of lapc if using ab timestepper

# smoothness of rolloff functions
slepian_r = 1.0*M
# Define the nonlinear reaction term
nonlinearity = lambda x: x*np.cos(x)
# Scale the nonlinear reaction term
scaled_nonlinearity = lambda x: reaction_value*nonlinearity(x)
# Ensure the timestep works well with max_time
tsteps = max_time / dt
if np.abs(int(np.round(tsteps)) - tsteps)/np.abs(tsteps) > 1e-14:
    raise Exception
tsteps = int(np.round(tsteps))
# Define the number of grid points
ngx = int(nb)*grid_upsampling
ngy = int(nb/2)*grid_upsampling

# stuff for the modified helmholtz equation (for first timestep of Backward Euler)
zeta0         = nu*dt
helmholtz_k0  = np.sqrt(1.0/zeta0)
d_singular0   = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k0)
s_singular0   = lambda src: -(d_singular0(src)/src.weights).T*src.weights
Singular_SLP0 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k0)
Naive_SLP0    = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k0, ifcharge=True)
# stuff for the modified helmholtz equation (for all other timesteps, using AB)
zeta          = 0.5*nu*dt
helmholtz_k   = np.sqrt(1.0/zeta)
d_singular    = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
s_singular    = lambda src: -(d_singular(src)/src.weights).T*src.weights
Singular_SLP  = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
Naive_SLP     = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
# stuff for the modified helmholtz equation (for either)
half_eye      = lambda src: np.eye(src.N)*0.5
# stuff for the modified helmholtz equation (for all other timesteps)
bdf_zeta          = (2/3)*nu*dt
bdf_helmholtz_k   = np.sqrt(1.0/bdf_zeta)
bdf_d_singular    = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=bdf_helmholtz_k)
bdf_s_singular    = lambda src: -(bdf_d_singular(src)/src.weights).T*src.weights
bdf_Singular_SLP  = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=bdf_helmholtz_k)
bdf_Naive_SLP     = lambda src, trg: MH_Layer_Form(src, trg, k=bdf_helmholtz_k, ifcharge=True)

# generate a velocity field
kk = 2*np.pi/3
const_u_function = lambda x, y, t:  x*0.0 + 1.0
const_v_function = lambda x, y, t:  x*0.0
diff_u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*np.cos(2*np.pi*t)
diff_v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*np.cos(2*np.pi*t)
def combine_functions(df, cf):
    return lambda x, y, t: const_factor*cf(x, y, t) + diff_factor*df(x, y, t)
u_function = combine_functions(diff_u_function, const_u_function)
v_function = combine_functions(diff_v_function, const_v_function)
# generate an initial concentration field
c0_function = lambda x, y: np.exp(np.cos(kk*x))*np.sin(kk*y)

################################################################################
# Get truth via just using Adams-Bashforth 2 on periodic domain

# generate a grid
xv,  h = np.linspace(xmin, xmax, ngx, endpoint=False, retstep=True)
yv, _h = np.linspace(ymin, ymax, ngy, endpoint=False, retstep=True)
if _h != h:
    raise Exception('Need grid to be isotropic')
del _h
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
ebdy = EmbeddedBoundary(bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=coordinate_tol)
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# get a grid
grid = Grid([xmin, xmax], ngx, [ymin, ymax], ngy, x_endpoints=[True, False], y_endpoints=[True, False])
ebdyc.register_grid(grid)

# initial c field
c0 = EmbeddedFunction(ebdyc)
c0.define_via_function(c0_function)

# now timestep
c = c0.copy()
t = 0.0

x_tracers = [ebdy.bdy.x,]
y_tracers = [ebdy.bdy.y,]
ts = []

def myprint(*args):
    if timing_marks:
        print(*args)

not_first_time = False
while t < max_time-1e-10:

    # get the velocity fields at this time
    u = EmbeddedFunction(ebdyc)
    u.define_via_function(lambda x, y: u_function(x, y, t))
    v = EmbeddedFunction(ebdyc)
    v.define_via_function(lambda x, y: v_function(x, y, t))
    # get the velocity fields on the boundary
    ub = ebdyc.interpolate_radial_to_boundary(u).bdy_value_list[0]
    vb = ebdyc.interpolate_radial_to_boundary(v).bdy_value_list[0]

    # step of the first order method to get things started
    if t == 0 or timestepper == 'fbe':

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

        # keep track of boundary motion
        x_tracers.append(bx)
        y_tracers.append(by)

        # take gradients of the velocity fields
        ux, uy = ebdyc.gradient(u)
        vx, vy = ebdyc.gradient(v)

        # now generate a new ebdy based on the moved boundary
        new_bdy = GSB(x=bx, y=by)
        bh = new_bdy.dt*new_bdy.speed.min()
        # construct embedded boundary
        new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=coordinate_tol)
        new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
        umax = np.sqrt((u*u + v*v).data, where=np.logical_not(u.mask)).max()
        ddd = 2*umax*dt
        if use_danger_zone:
            new_ebdyc.register_grid(grid, danger_zone_distance=ddd)
        else:
            new_ebdyc.register_grid(grid)

        # let's get the points that need to be interpolated to
        aap = new_ebdyc.pnar
        if use_danger_zone:
            AP_key  = ebdyc.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        else:
            AP_key  = ebdyc.register_points(aap.x, aap.y)

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[AP_key]

        # generate a holding ground for the new c
        c_new = EmbeddedFunction(new_ebdyc)
        c_new.zero()

        # get departure points
        xd_all = np.zeros(aap.N)
        yd_all = np.zeros(aap.N)

        # advect those in the annulus
        c1n, c2n, c3n = AEP.get_Ns()
        # category 1 and 2
        c1_2n = c1n + c2n
        c1_2 = AEP.zone1_or_2
        uxh = ebdyc.interpolate_to_points(ux, aap.x, aap.y)
        uyh = ebdyc.interpolate_to_points(uy, aap.x, aap.y)
        vxh = ebdyc.interpolate_to_points(vx, aap.x, aap.y)
        vyh = ebdyc.interpolate_to_points(vy, aap.x, aap.y)
        uh = ebdyc.interpolate_to_points(u, aap.x, aap.y)
        vh = ebdyc.interpolate_to_points(v, aap.x, aap.y)
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
        xd_all[c1_2] = xd
        yd_all[c1_2] = yd
        # categroy 3... this is the tricky one
        if c3n > 0:
            for ind, ebdy in enumerate(ebdyc):
                c3l = AEP.zone3l[ind]
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
                urb = ebdy.interpolate_radial_to_boundary_normal_derivative(u[ind])
                vrb = ebdy.interpolate_radial_to_boundary_normal_derivative(v[ind])
                ub_interp   = interp(ub)
                vb_interp   = interp(vb)
                urb_interp  = interp(urb)
                vrb_interp  = interp(vrb)
                ubs_interp  = interp(d1_der(ub))
                vbs_interp  = interp(d1_der(vb))
                urbs_interp = interp(d1_der(urb))
                vrbs_interp = interp(d1_der(vrb))
                xo = aap.x[c3l]
                yo = aap.y[c3l]
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
                # take as guess inds our s, r
                s = AEP.zone3t[ind]
                r = AEP.zone3r[ind]
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
                xd_all[c3l] = xd
                yd_all[c3l] = yd
        # now interpolate to c
        if use_danger_zone:
            ch = ebdyc.interpolate_to_points(c - dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        else:
            ch = ebdyc.interpolate_to_points(c - dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True)
        # set the grid values
        c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:new_ebdyc.grid_pna_num]
        # set the radial values
        c_new.radial_value_list[0][:] = ch[new_ebdyc.grid_pna_num:].reshape(new_ebdyc[0].radial_shape)
        # overwrite under grid under annulus by radial grid
        _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
        # save this for a moment
        c_force = c_new.copy()

        # now solve diffusion equation
        solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=helmholtz_k0)
        c_new_update = solver(c_new/zeta0, tol=1e-12, verbose=False)
        A = s_singular0(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
        bvn = solver.get_boundary_normal_derivatives(c_new_update.radial_value_list)
        tau = np.linalg.solve(A, bvn)
        qfs = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP0,], Naive_SLP0, on_surface=True, form_b2c=False)
        sigma = qfs([tau,])
        out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k0)
        gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
        rslp = rslpl[0]
        c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
        c_new_update.grid_value[new_ebdyc.phys] += gslp
        c_new = c_new_update
        # overwrite under grid under annulus by radial grid
        _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)

        # compute lapc
        if timestepper == 'ab' and backcomputation:
            lapc = (c_new - c_force)/(nu*dt)

    else:  # the second order method

        st = time.time()

        bx = ebdy.bdy.x + 0.5*dt*(3*ub - ubo_new_parm)
        by = ebdy.bdy.y + 0.5*dt*(3*vb - vbo_new_parm)
        bx, by, new_t = arc_length_parameterize(bx, by, return_t=True)
        bu_interp = interp1d(0, 2*np.pi, bdy.dt, ub, p=True)
        bv_interp = interp1d(0, 2*np.pi, bdy.dt, vb, p=True)
        # old boundary velocity values have to be in the correct place
        ubo_new_parm = bu_interp(new_t)
        vbo_new_parm = bv_interp(new_t)
        ubo = ub.copy()
        vbo = vb.copy()

        myprint('          Boundary update: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        x_tracers.append(bx)
        y_tracers.append(by)

        # take gradients of the velocity fields
        ux, uy = ebdyc.gradient(u)
        vx, vy = ebdyc.gradient(v)

        myprint('          Velocity gradient: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # now generate a new ebdy based on the moved boundary
        new_bdy = GSB(x=bx, y=by)
        bh = new_bdy.dt*new_bdy.speed.min()
        # construct embedded boundary
        new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=coordinate_tol)
        new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
        umax = np.sqrt((u*u + v*v).data, where=np.logical_not(u.mask)).max()
        ddd = 2*umax*dt
        if use_danger_zone:
            new_ebdyc.register_grid(grid, danger_zone_distance=ddd)
        else:
            new_ebdyc.register_grid(grid)

        myprint('          Update ebdy: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # let's get the points that need to be interpolated to
        aap = new_ebdyc.pnar

        if use_danger_zone:
            AP_key  = ebdyc.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
            OAP_key = ebdyc_old.register_points(aap.x, aap.y, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
        else:
            AP_key  = ebdyc.register_points(aap.x, aap.y)
            OAP_key = ebdyc_old.register_points(aap.x, aap.y)

        myprint('          Register points to old ebdys: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # now we need to interpolate onto things
        AEP = ebdyc.registered_partitions[AP_key]
        OAEP = ebdyc_old.registered_partitions[OAP_key]

        # generate a holding ground for the new c
        c_new = EmbeddedFunction(new_ebdyc)
        c_new.zero()

        # get departure points
        xd_all = np.zeros(aap.N)
        yd_all = np.zeros(aap.N)
        xD_all = np.zeros(aap.N)
        yD_all = np.zeros(aap.N)

        # advect those in the annulus
        c1n, c2n, c3n = AEP.get_Ns()
        oc1n, oc2n, oc3n = OAEP.get_Ns()
        # category 1 and 2
        c1_2 = AEP.zone1_or_2
        oc1_2 = OAEP.zone1_or_2
        fc12 = np.logical_and(c1_2, oc1_2)
        fc12n = np.sum(fc12)

        myprint('          Separate categories: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # category 1 and 2
        uxh = ebdyc.interpolate_to_points(ux, aap.x, aap.y)
        uyh = ebdyc.interpolate_to_points(uy, aap.x, aap.y)
        vxh = ebdyc.interpolate_to_points(vx, aap.x, aap.y)
        vyh = ebdyc.interpolate_to_points(vy, aap.x, aap.y)
        uh =  ebdyc.interpolate_to_points(u, aap.x, aap.y)
        vh =  ebdyc.interpolate_to_points(v, aap.x, aap.y)

        uxoh = ebdyc_old.interpolate_to_points(uxo, aap.x, aap.y)
        uyoh = ebdyc_old.interpolate_to_points(uyo, aap.x, aap.y)
        vxoh = ebdyc_old.interpolate_to_points(vxo, aap.x, aap.y)
        vyoh = ebdyc_old.interpolate_to_points(vyo, aap.x, aap.y)
        uoh =  ebdyc_old.interpolate_to_points(uo,  aap.x, aap.y)
        voh =  ebdyc_old.interpolate_to_points(vo,  aap.x, aap.y)

        myprint('          Interpolate velocities: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        SLM = np.zeros([fc12n,] + [4,4], dtype=float)
        SLR = np.zeros([fc12n,] + [4,], dtype=float)

        # solve for departure points
        SLM[:,0,0] = uxh[fc12]
        SLM[:,0,1] = uyh[fc12]
        SLM[:,0,2] = 0.5/dt
        SLM[:,0,3] = 0.0
        SLM[:,1,0] = vxh[fc12]
        SLM[:,1,1] = vyh[fc12]
        SLM[:,1,2] = 0.0
        SLM[:,1,3] = 0.5/dt
        SLM[:,2,0] = 2.0/dt + 3*uxh[fc12]
        SLM[:,2,1] = 3*uyh[fc12]
        SLM[:,2,2] = -uxoh[fc12]
        SLM[:,2,3] = -uyoh[fc12]
        SLM[:,3,0] = 3*vxh[fc12]
        SLM[:,3,1] = 2.0/dt + 3*vyh[fc12]
        SLM[:,3,2] = -vxoh[fc12]
        SLM[:,3,3] = -vyoh[fc12]
        SLR[:,0] = uh[fc12]
        SLR[:,1] = vh[fc12]
        SLR[:,2] = 3*uh[fc12] - uoh[fc12]
        SLR[:,3] = 3*vh[fc12] - voh[fc12]
        OUT = np.linalg.solve(SLM, SLR)
        dx, dy, Dx, Dy = OUT[:,0], OUT[:,1], OUT[:,2], OUT[:,3]
        xd, yd = aap.x[fc12] - dx, aap.y[fc12] - dy
        xD, yD = aap.x[fc12] - Dx, aap.y[fc12] - Dy
        xd_all[fc12] = xd
        yd_all[fc12] = yd
        xD_all[fc12] = xD
        yD_all[fc12] = yD

        myprint('          Category 2 work: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # categroy 3... this is the tricky one
        fc3n = aap.N - fc12n
        if fc3n > 0:
            for ind, (ebdy, ebdy_old) in enumerate(zip(ebdyc, ebdyc_old)):
                c3l = AEP.zone3l[ind]
                oc3l = OAEP.zone3l[ind]
                fc3l = np.unique(np.concatenate([c3l, oc3l]))

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
                urrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(u.radial_value_list[0])
                vrrb = ebdy.interpolate_radial_to_boundary_normal_derivative2(v.radial_value_list[0])
                ub_interp   = interp(ub)
                vb_interp   = interp(vb)
                urb_interp  = interp(urb)
                vrb_interp  = interp(vrb)
                urrb_interp  = interp(urrb)
                vrrb_interp  = interp(vrrb)
                ubs_interp  = interp(d1_der(ub))
                vbs_interp  = interp(d1_der(vb))
                urbs_interp = interp(d1_der(urb))
                vrbs_interp = interp(d1_der(vrb))
                urrbs_interp = interp(d1_der(urrb))
                vrrbs_interp = interp(d1_der(vrrb))

                old_bx_interp  = interp(ebdy_old.bdy.x)
                old_by_interp  = interp(ebdy_old.bdy.y)
                old_bxs_interp = interp(d1_der(ebdy_old.bdy.x))
                old_bys_interp = interp(d1_der(ebdy_old.bdy.y))
                old_nx_interp  = interp(ebdy_old.bdy.normal_x)
                old_ny_interp  = interp(ebdy_old.bdy.normal_y)
                old_nxs_interp = interp(d1_der(ebdy_old.bdy.normal_x))
                old_nys_interp = interp(d1_der(ebdy_old.bdy.normal_y))
                old_ub = ebdy_old.interpolate_radial_to_boundary(uo.radial_value_list[0])
                old_vb = ebdy_old.interpolate_radial_to_boundary(vo.radial_value_list[0])
                old_urb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(uo.radial_value_list[0])
                old_vrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative(vo.radial_value_list[0])
                old_urrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(uo.radial_value_list[0])
                old_vrrb = ebdy_old.interpolate_radial_to_boundary_normal_derivative2(vo.radial_value_list[0])
                # i think the old parm is right, but should think about
                old_ub_interp   = interp(old_ub)
                old_vb_interp   = interp(old_vb)
                old_urb_interp  = interp(old_urb)
                old_vrb_interp  = interp(old_vrb)
                old_urrb_interp  = interp(old_urrb)
                old_vrrb_interp  = interp(old_vrrb)
                old_ubs_interp  = interp(d1_der(old_ub))
                old_vbs_interp  = interp(d1_der(old_vb))
                old_urbs_interp = interp(d1_der(old_urb))
                old_vrbs_interp = interp(d1_der(old_vrb))
                old_urrbs_interp = interp(d1_der(old_urrb))
                old_vrrbs_interp = interp(d1_der(old_vrrb))

                xx = aap.x[fc3l]
                yy = aap.y[fc3l]
                def objective(s, r, so, ro):
                    f = np.empty([s.size, 4])
                    f[:,0] = old_bx_interp(so) + ro*old_nx_interp(so) + 2*dt*ub_interp(s) + 2*dt*r*urb_interp(s) + dt*r**2*urrb_interp(s) - xx
                    f[:,1] = old_by_interp(so) + ro*old_ny_interp(so) + 2*dt*vb_interp(s) + 2*dt*r*vrb_interp(s) + dt*r**2*vrrb_interp(s) - yy
                    f[:,2] = bx_interp(s) + r*nx_interp(s) + 1.5*dt*ub_interp(s) + 1.5*dt*r*urb_interp(s) + 0.75*dt*r**2*urrb_interp(s) - 0.5*dt*old_ub_interp(so) - 0.5*dt*ro*old_urb_interp(so) - 0.25*dt*ro**2*old_urrb_interp(so) - xx
                    f[:,3] = by_interp(s) + r*ny_interp(s) + 1.5*dt*vb_interp(s) + 1.5*dt*r*vrb_interp(s) + 0.75*dt*r**2*vrrb_interp(s) - 0.5*dt*old_vb_interp(so) - 0.5*dt*ro*old_vrb_interp(so) - 0.25*dt*ro**2*old_vrrb_interp(so) - yy
                    return f
                def Jac(s, r, so, ro):
                    J = np.empty([s.size, 4, 4])
                    # derivative with respect to s
                    J[:,0,0] = 2*dt*ubs_interp(s) + 2*dt*r*urbs_interp(s) + dt*r**2*urrbs_interp(s)
                    J[:,1,0] = 2*dt*vbs_interp(s) + 2*dt*r*vrbs_interp(s) + dt*r**2*vrrbs_interp(s)
                    J[:,2,0] = bxs_interp(s) + r*nxs_interp(s) + 1.5*dt*ubs_interp(s) + 1.5*dt*r*urbs_interp(s) + 0.75*dt*r**2*urrbs_interp(s)
                    J[:,3,0] = bys_interp(s) + r*nys_interp(s) + 1.5*dt*vbs_interp(s) + 1.5*dt*r*vrbs_interp(s) + 0.75*dt*r**2*vrrbs_interp(s)
                    # derivative with respect to r
                    J[:,0,1] = 2*dt*urb_interp(s) + 2*dt*r*urrb_interp(s)
                    J[:,1,1] = 2*dt*vrb_interp(s) + 2*dt*r*vrrb_interp(s)
                    J[:,2,1] = nx_interp(s) + 1.5*dt*urb_interp(s) + 1.5*dt*r*urrb_interp(s)
                    J[:,3,1] = ny_interp(s) + 1.5*dt*vrb_interp(s) + 1.5*dt*r*vrrb_interp(s)
                    # derivative with respect to so
                    J[:,0,2] = old_bxs_interp(so) + ro*old_nxs_interp(so)
                    J[:,1,2] = old_bys_interp(so) + ro*old_nys_interp(so)
                    J[:,2,2] = -0.5*dt*old_ubs_interp(so) - 0.5*dt*ro*old_urbs_interp(so) - 0.25*dt*ro**2*old_urrbs_interp(so)
                    J[:,3,2] = -0.5*dt*old_vbs_interp(so) - 0.5*dt*ro*old_vrbs_interp(so) - 0.25*dt*ro**2*old_vrrbs_interp(so)
                    # derivative with respect to ro
                    J[:,0,3] = old_nx_interp(so)
                    J[:,1,3] = old_ny_interp(so)
                    J[:,2,3] = -0.5*dt*old_urb_interp(so) - 0.5*dt*ro*old_urrb_interp(so)
                    J[:,3,3] = -0.5*dt*old_vrb_interp(so) - 0.5*dt*ro*old_vrrb_interp(so)
                    return J
                # take as guess inds our s, r
                s =   AEP.full_t[fc3l]
                r =   AEP.full_r[fc3l]
                so = OAEP.full_t[fc3l]
                ro = OAEP.full_r[fc3l]
                # now solve for sd, rd
                res = objective(s, r, so, ro)
                mres1 = np.hypot(res[:,0], res[:,1]).max()
                mres2 = np.hypot(res[:,2], res[:,3]).max()
                mres = max(mres1, mres2)
                tol = 1e-12
                while mres > tol:
                    J = Jac(s, r, so, ro)
                    d = np.linalg.solve(J, res)
                    s  -= d[:,0]
                    r  -= d[:,1]
                    so -= d[:,2]
                    ro -= d[:,3]
                    res = objective(s, r, so, ro)
                    mres1 = np.hypot(res[:,0], res[:,1]).max()
                    mres2 = np.hypot(res[:,2], res[:,3]).max()
                    mres = max(mres1, mres2)
                r_fail_1 = r.max() > 0.0
                r_fail_2 = r.min() < -ebdy.radial_width
                ro_fail_1 = ro.max() > 0.0
                ro_fail_2 = ro.min() < -ebdy_old.radial_width
                r_fail = r_fail_1 or r_fail_2
                ro_fail = ro_fail_1 or ro_fail_2
                fail = r_fail or ro_fail
                fail_amount = 0.0
                if fail:
                    if r_fail_1:
                        fail_amount = max(fail_amount, r.max())
                        r[r > 0.0] = 0.0
                    if r_fail_2:
                        fail_amount = max(fail_amount, (-r-ebdy.radial_width).max())
                        r[r < -ebdy.radial_width] = -ebdy.radial_width
                    if ro_fail_1:
                        fail_amount = max(fail_amount, ro.max())
                        ro[ro > 0.0] = 0.0
                    if ro_fail_2:
                        fail_amount = max(fail_amount, (-ro-ebdy_old.radial_width).max())
                        ro[ro < -ebdy_old.radial_width] = -ebdy_old.radial_width
                    print('Failure! by: {:0.2e}'.format(fail_amount))

                # get the departure points
                xd = bx_interp(s) + nx_interp(s)*r
                yd = by_interp(s) + ny_interp(s)*r
                xD = old_bx_interp(so) + old_nx_interp(so)*ro
                yD = old_by_interp(so) + old_ny_interp(so)*ro
                xd_all[fc3l] = xd
                yd_all[fc3l] = yd
                xD_all[fc3l] = xD
                yD_all[fc3l] = yD

        myprint('          Category 3 work: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        if timestepper == 'ab':
            subst = time.time()

            # directly compute lapc instead of back-calculating it
            if not backcomputation:
                lapc = ebdyc.laplacian(c)
                myprint('               direct lapc: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()

            # now interpolate to c
            subst = time.time()
            if use_danger_zone:
                ch1 = ebdyc.interpolate_to_points(c + 0.5*dt*nu*lapc - 1.5*dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
            else:
                ch1 = ebdyc.interpolate_to_points(c + 0.5*dt*nu*lapc - 1.5*dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True)
            myprint('               interp1: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            if use_danger_zone:
                ch2 = ebdyc_old.interpolate_to_points(0.5*dt*scaled_nonlinearity(c_old), xD_all, yD_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
            else:
                ch2 = ebdyc_old.interpolate_to_points(0.5*dt*scaled_nonlinearity(c_old), xD_all, yD_all, fix_r=True)
            myprint('               interp2: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            ch = ch1 + ch2
            # set the grid values
            c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:new_ebdyc.grid_pna_num]
            myprint('               fill: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            # set the radial values
            c_new.radial_value_list[0][:] = ch[new_ebdyc.grid_pna_num:].reshape(new_ebdyc[0].radial_shape)
            # overwrite under grid under annulus by radial grid
            _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
            myprint('               overwrite: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            # save for the moment
            c_force = c_new.copy()
            myprint('               copy: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()

            myprint('          c advection: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

            subst = time.time()
            if not_first_time:
                solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=helmholtz_k, AS_list=AS_list)
                not_first_time = True
            else:
                solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=helmholtz_k)
                AS_list = solver.AS_list
            myprint('               make solver: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            c_new_update = solver(c_new/zeta, tol=1e-12, verbose=verbose)
            myprint('               solve: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            A = s_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
            myprint('               form A: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            bvn = solver.get_boundary_normal_derivatives(c_new_update.radial_value_list)
            myprint('               get bvn: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            tau = np.linalg.solve(A, bvn)
            myprint('               solve for tau: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            qfs = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [Singular_SLP,], Naive_SLP, on_surface=True, form_b2c=False)
            myprint('               get qfs evaluator: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            sigma = qfs([tau,])
            myprint('               get qfs sigma: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=helmholtz_k)
            myprint('               evaluate layer: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
            rslp = rslpl[0]
            c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
            c_new_update.grid_value[new_ebdyc.phys] += gslp
            c_new = c_new_update
            # overwrite under grid under annulus by radial grid
            _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
            myprint('               finish: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()

            myprint('          c diffusion: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

            # compute lapc
            if backcomputation:
                lapc = 2*(c_new - c_force)/(nu*dt)
                myprint('          backcomputation of lap c: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        elif timestepper == 'bdf':

            subst = time.time()
            if use_danger_zone:
                ch1 = ebdyc.interpolate_to_points((4/3)*c - (4/3)*dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
            else:
                ch1 = ebdyc.interpolate_to_points((4/3)*c - (4/3)*dt*scaled_nonlinearity(c), xd_all, yd_all, fix_r=True)
            myprint('               interp1: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            if use_danger_zone:
                ch2 = ebdyc_old.interpolate_to_points((1/3)*c_old - (2/3)*dt*scaled_nonlinearity(c_old), xD_all, yD_all, fix_r=True, dzl=new_ebdyc.danger_zone_list, gil=new_ebdyc.guess_ind_list)
            else:
                ch2 = ebdyc_old.interpolate_to_points((1/3)*c_old - (2/3)*dt*scaled_nonlinearity(c_old), xD_all, yD_all, fix_r=True)
            myprint('               interp2: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            ch = ch1 - ch2
            # set the grid values
            c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:new_ebdyc.grid_pna_num]
            myprint('               fill: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            # set the radial values
            c_new.radial_value_list[0][:] = ch[new_ebdyc.grid_pna_num:].reshape(new_ebdyc[0].radial_shape)
            # overwrite under grid under annulus by radial grid
            _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
            myprint('               overwrite: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()

            myprint('          c advection: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

            subst = time.time()
            if not_first_time:
                solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=bdf_helmholtz_k, AS_list=AS_list)
                not_first_time = True
            else:
                solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=bdf_helmholtz_k)
                AS_list = solver.AS_list
            myprint('               make solver: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            c_new_update = solver(c_new/bdf_zeta, tol=1e-12, verbose=verbose)
            myprint('               solve: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            A = bdf_s_singular(new_ebdyc.ebdys[0].bdy) - half_eye(new_ebdyc.ebdys[0].bdy)
            myprint('               form A: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            bvn = solver.get_boundary_normal_derivatives(c_new_update.radial_value_list)
            myprint('               get bvn: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            tau = np.linalg.solve(A, bvn)
            myprint('               solve for tau: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            qfs = QFS_Evaluator(new_ebdyc.ebdys[0].bdy_qfs, True, [bdf_Singular_SLP,], bdf_Naive_SLP, on_surface=True, form_b2c=False)
            myprint('               get qfs evaluator: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            sigma = qfs([tau,])
            myprint('               get qfs sigma: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            out = MH_Layer_Apply(new_ebdyc.bdy_inward_sources, new_ebdyc.grid_and_radial_pts, charge=sigma, k=bdf_helmholtz_k)
            myprint('               evaluate layer: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
            gslp, rslpl = new_ebdyc.divide_grid_and_radial(out)
            rslp = rslpl[0]
            c_new_update.radial_value_list[0] += rslp.reshape(M, nb)
            c_new_update.grid_value[new_ebdyc.phys] += gslp
            c_new = c_new_update
            # overwrite under grid under annulus by radial grid
            _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
            myprint('               finish: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()

            myprint('          c diffusion: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        else:
            raise Exception('timestepper should be bdf or ab')


    ebdy_old = ebdy
    ebdyc_old = ebdyc
    ebdy = new_ebdy
    ebdyc = new_ebdyc
    c_old = c
    c = c_new
    uxo = ux
    uyo = uy
    vxo = vx
    vyo = vy
    uo = u
    vo = v

    ts.append(t)

    t += dt
    print('   t = {:0.3f}'.format(t), 'of', max_time)

def plotto(u):
    fig, ax = plt.subplots()
    clf = u.plot(ax)
    fig.colorbar(clf)

################################################################################
# Evaluate

fig, ax = plt.subplots()
clf = c.plot(ax)
plt.colorbar(clf)

try:
    fig, ax = plt.subplots()
    clf = c_save.plot(ax)
    plt.colorbar(clf)

    c_diff = c_save - c
    c_adiff = np.abs(c_diff)
    fig, ax = plt.subplots()
    clf = c_diff.plot(ax)
    plt.colorbar(clf)
    print('Difference is: {:0.2e}'.format(c_adiff.max()))
    c_save = c.copy()
except:
    pass
    c_save = c.copy()

if False:
    ##### nufft; spectral; pad=0; fsuf=4; tmax=1.1
    dts                  = [0.05,   0.025,    0.0125,   0.00625,  0.003125]
    # first order (solution stays smooth for all of them, even at small timesteps)
    diffs_fbe_200_12_2_2 = [np.nan, 1.82e-02, 9.23e-03, 4.48e-03, 8.43e-03]
    diffs_fbe_300_24_2_2 = [np.nan, 1.82e-02, 9.24e-03, 4.66e-03, 2.34e-03]
    # ab, back computation (solution fails to stay smooth!)
    diffs_abb_200_12_2_2 = [np.nan, 3.28e-03, 1.33e-03, 1.94e-02, 1.02e+06]
    diffs_abb_300_24_2_2 = [np.nan, 3.29e-03, 8.74e-04, 2.40e-04, 3.83e+01]
    # ab, direct computation (solution stays smooth for all of them, even at small timesteps)
    diffs_abd_200_12_2_2 = [np.nan, 3.27e-03, 1.03e-03, 1.17e-02, 1.50e-01]
    diffs_abd_300_24_2_2 = [np.nan, 3.29e-03, 8.74e-04, 2.37e-04, 2.31e-02]
    # bdf (solution stays smooth for all of them, even at small timesteps)
    diffs_bdf_200_12_2_2 = [np.nan, 4.43e-03, 1.06e-03, 3.17e-03, 3.61e-02]
    # diffs_bdf_200_16_2_2 = [np.nan, 4.41e-03, 1.02e-03, 2.92e-04, 1.80e-03]
    diffs_bdf_300_24_2_2 = [np.nan, 4.42e-03, 1.02e-03, 2.55e-04, 6.31e-05, 1.85e-05]

    ##### polyi; fourth; pad=0; fsuf=4; tmax=1.1
    diffs_bdf_200_12_2_2 = [np.nan, 8.56e-03, 4.37e-03, 1.83e-03, 3.37e-02, 2.85e-01]
    diffs_bdf_300_24_2_2 = [np.nan, 4.38e-03, 1.06e-03, 2.94e-04, 7.22e-05, ]

    ##### polyi; fourth; pad=0; fsuf=4; tmax=6.0 (nice and smooth!)
    diffs_bdf_200_12_2_2 = [np.nan, 1.83e-04, 4.19e-05, 1.58e-04, 'SMOOTH!']
