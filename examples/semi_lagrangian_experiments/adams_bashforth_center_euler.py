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

max_time          = 1.3        # time to run to
reaction_value    = 0.5        # value for reaction term
const_factor      = 0.5        # constant advection velocity
diff_factor       = 0.7        # non-constant advection velocity
nu                = 0.1        # diffusion coefficient
nb                = 200        # number of points in boundary
grid_upsampling   = 2          # upsampling of grid relative to boundary
radial_upsampling = 2          # how much to upsample cheb grid by (relative to background)
M                 = 16         # number of chebyshev modes
pad_zone          = 0          # padding in radial grid (typically set to 0 for spectral methods)
dt                = 0.1/2      # timestep
solver_type       = 'spectral' # 'spectral' or 'fourth'
qfs_fsuf          = 4          # forced upsampling for QFS
coordinate_scheme = 'nufft'    # interpolation scheme for coordinate finding ('polyi' or 'nufft')
coordinate_tol    = 1e-12       # probablys should be set to be compatible with 
xmin              = -1.0       # domain definition
xmax              =  2.0       # domain definition
ymin              = -1.5       # domain definition
ymax              =  1.5       # domain definition
printjustment     =  40        # how much to adjust for printing
verbose           = False      # provide verbose output for radial solve
timing_marks      = True

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
ngx = int(nb/2)*grid_upsampling
ngy = int(nb/2)*grid_upsampling

# stuff for the modified helmholtz equation (for first timestep of Backward Euler)
zeta0         = nu*dt
helmholtz_k0  = np.sqrt(1.0/zeta0)
d_singular0   = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k0)
s_singular0   = lambda src: -(d_singular0(src)/src.weights).T*src.weights
Singular_SLP0 = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k0)
Naive_SLP0    = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k0, ifcharge=True)
# stuff for the modified helmholtz equation (for all other timesteps)
zeta        = 0.5*nu*dt
helmholtz_k = np.sqrt(1.0/zeta)
d_singular  = lambda src: src.Modified_Helmholtz_DLP_Self_Form(k=helmholtz_k)
s_singular  = lambda src: -(d_singular(src)/src.weights).T*src.weights
Singular_SLP  = lambda src, _: src.Modified_Helmholtz_SLP_Self_Form(k=helmholtz_k)
Naive_SLP     = lambda src, trg: MH_Layer_Form(src, trg, k=helmholtz_k, ifcharge=True)
# stuff for the modified helmholtz equation (for either)
half_eye      = lambda src: np.eye(src.N)*0.5

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
ebdy = EmbeddedBoundary(bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=1e-8)
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
    if t == 0:

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
        ux, uy = ebdyc.gradient2(u)
        vx, vy = ebdyc.gradient2(v)

        # now generate a new ebdy based on the moved boundary
        new_bdy = GSB(x=bx, y=by)
        bh = new_bdy.dt*new_bdy.speed.min()
        # construct embedded boundary
        new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=1e-8)
        new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
        new_ebdyc.register_grid(grid)

        # let's get the points that need to be interpolated to
        gp = new_ebdyc.grid_pna
        ap = new_ebdyc.radial_pts

        aax = np.concatenate([gp.x, ap.x])
        aay = np.concatenate([gp.y, ap.y])
        aap = PointSet(x=aax, y=aay)
        AP_key  = ebdy.register_points(aap.x, aap.y)

        # now we need to interpolate onto things
        AEP = ebdy.registered_partitions[AP_key]

        # generate a holding ground for the new c
        c_new = EmbeddedFunction(new_ebdyc)
        c_new.zero()

        # advect those in the annulus
        c1,  c2,  c3  = AEP.get_categories()
        c1n, c2n, c3n = AEP.get_category_Ns()
        c12 = np.logical_or(c1, c2)
        # category 1 and 2 (use Eulerian scheme)
        cx, cy = ebdyc.gradient2(c)
        ecnew = c - dt*(u*cx + v*cy) - dt*scaled_nonlinearity(c)
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
            # take as guess inds our s, r
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
        # now interpolate to c
        ch3 = ebdy.interpolate_to_points(c - dt*scaled_nonlinearity(c), xd, yd)
        ch12 = ebdy.interpolate_to_points(ecnew, aap.x[c12], aap.y[c12])
        # get the full thing
        ch = np.empty(aap.N)
        ch[c12] = ch12
        ch[c3] = ch3
        # set the grid values
        c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
        # set the radial values
        c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
        # overwrite under grid under annulus by radial grid
        _ = new_ebdyc.interpolate_radial_to_grid(c_new.radial_value_list, c_new.grid_value)
        # save this for a moment
        c_force = c_new.copy()

        # now solve diffusion equation
        solver = ModifiedHelmholtzSolver(new_ebdyc, solver_type=solver_type, k=helmholtz_k0)
        c_new_update = solver(c_new/zeta0, tol=1e-12, verbose=True)
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
        ux, uy = ebdyc.gradient2(u)
        vx, vy = ebdyc.gradient2(v)

        myprint('          Velocity gradient: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # now generate a new ebdy based on the moved boundary
        new_bdy = GSB(x=bx, y=by)
        bh = new_bdy.dt*new_bdy.speed.min()
        # construct embedded boundary
        new_ebdy = EmbeddedBoundary(new_bdy, True, M, bh/radial_upsampling, pad_zone=pad_zone, heaviside=MOL.step, qfs_fsuf=qfs_fsuf, coordinate_scheme=coordinate_scheme, coordinate_tolerance=1e-8)
        new_ebdyc = EmbeddedBoundaryCollection([new_ebdy,])
        umax = np.sqrt((u*u + v*v).data, where=np.logical_not(u.mask)).max()
        ddd = 2*umax*dt
        new_ebdyc.register_grid(grid, danger_zone_distance=ddd)

        myprint('          Update ebdy: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # let's get the points that need to be interpolated to
        gp = new_ebdyc.grid_pna
        ap = new_ebdyc.radial_pts

        aax = np.concatenate([gp.x, ap.x])
        aay = np.concatenate([gp.y, ap.y])
        aap = PointSet(x=aax, y=aay)

        AP_key  = ebdy.register_points(aap.x, aap.y, danger_zone=new_ebdy.in_danger_zone, gi=new_ebdy.guess_inds)
        OAP_key = ebdy_old.register_points(aap.x, aap.y, danger_zone=new_ebdy.in_danger_zone, gi=new_ebdy.guess_inds)

        myprint('          Register points to old ebdys: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # now we need to interpolate onto things
        AEP = ebdy.registered_partitions[AP_key]
        OAEP = ebdy_old.registered_partitions[OAP_key]

        # generate a holding ground for the new c
        c_new = EmbeddedFunction(new_ebdyc)
        c_new.zero()

        # advect those in the annulus
        c1,  c2,  c3  = AEP.get_categories()
        oc1, oc2, oc3 = OAEP.get_categories()
        fc3 = np.logical_or(c3, oc3)
        fc12 = np.logical_not(fc3)
        fc3n = np.sum(fc3)
        fc12n = np.sum(fc12)

        myprint('          Interpolate velocities: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        cx, cy = ebdyc.gradient2(c)
        ecnew1 = c - 1.5*dt*( u*cx+v*cy + scaled_nonlinearity(c) )
        ecnew2 = 0.5*dt*( uo*cx_old+vo*cy_old + scaled_nonlinearity(c_old) )

        myprint('          Category 2 work: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # categroy 3... this is the tricky one
        if fc3n > 0:
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

            xx = aap.x[fc3]
            yy = aap.y[fc3]
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
            s =   AEP.full_t[fc3]
            r =   AEP.full_r[fc3]
            so = OAEP.full_t[fc3]
            ro = OAEP.full_r[fc3]
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
                # print('Failure! by: {:0.2e}'.format(fail_amount))

            # get the departure points
            xd = bx_interp(s) + nx_interp(s)*r
            yd = by_interp(s) + ny_interp(s)*r
            xD = old_bx_interp(so) + old_nx_interp(so)*ro
            yD = old_by_interp(so) + old_ny_interp(so)*ro

        myprint('          Category 3 work: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

        # now interpolate to c
        subst = time.time()
        ch1 = ebdy.interpolate_to_points(c + 0.5*dt*nu*lapc - 1.5*dt*scaled_nonlinearity(c), xd, yd, True, new_ebdy.in_danger_zone[fc3], new_ebdy.guess_inds[fc3])
        myprint('               interp1: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
        ch2 = ebdy_old.interpolate_to_points(0.5*dt*scaled_nonlinearity(c_old), xD, yD, True, new_ebdy.in_danger_zone[fc3], new_ebdy.guess_inds[fc3])
        myprint('               interp2: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
        ch3 = ch1 + ch2
        che1 = ebdy.interpolate_to_points(ecnew1 + 0.5*dt*nu*lapc, aap.x[fc12], aap.y[fc12], True, new_ebdy.in_danger_zone[fc12], new_ebdy.guess_inds[fc12])
        che2 = ebdy_old.interpolate_to_points(ecnew2, aap.x[fc12], aap.y[fc12], True, new_ebdy.in_danger_zone[fc12], new_ebdy.guess_inds[fc12])
        ch = np.empty(aap.N)
        ch[fc3] = ch3
        ch[fc12] = (che1+che2)
        # set the grid values
        c_new.grid_value[new_ebdyc.phys_not_in_annulus] = ch[:gp.N]
        myprint('               fill: '.ljust(printjustment), '{:0.1f}'.format((time.time()-subst)*100)); subst=time.time()
        # set the radial values
        c_new.radial_value_list[0][:] = ch[gp.N:].reshape(ebdy.radial_shape)
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
        lapc = 2*(c_new - c_force)/(nu*dt)

        myprint('          backcomputation of lap c: '.ljust(printjustment), '{:0.1f}'.format((time.time()-st)*100)); st=time.time()

    ebdy_old = ebdy
    ebdyc_old = ebdyc
    ebdy = new_ebdy
    ebdyc = new_ebdyc
    c_old = c
    cx_old = cx
    cy_old = cy
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
    for x_tracer, y_tracer in zip(x_tracers, y_tracers):
        ax.plot(x_tracer, y_tracer, alpha=0.5)

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
    # 400 / 32 / spectral / polyi / 1e-8 / to t=0.3
    [1.09e-02, 2.69e-03, 7.32e-04, 2.35e-04, 5.82e-04]

