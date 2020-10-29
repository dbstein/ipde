import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from personal_utilities.arc_length_reparametrization import arc_length_parameterize
import pybie2d
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
star = pybie2d.misc.curve_descriptions.star
from fast_interp import interp1d

"""
Test semi-lagrangian solve...
"""

# max time
max_time = 0.2
# timesteps to test at
dts = [0.04/2**i for i in range(6)]
# number of grid points
nb = 200

# initial boundary
bdy = GSB(c=star(nb, x=0.0, y=0.0, a=0.1, f=3))
bdy = GSB(*arc_length_parameterize(bdy.x, bdy.y))
initial_bx = bdy.x
initial_by = bdy.y

# generate a velocity field
kk = 2*np.pi/3
u_function = lambda x, y, t:  np.sin(kk*x)*np.cos(kk*y)*(1+np.cos(2*np.pi*t))
v_function = lambda x, y, t: -np.cos(kk*x)*np.sin(kk*y)*(1+np.cos(2*np.pi*t))

################################################################################
# Test simple advection of boundary without reparametrization
print('')
print('Without reparametrization')

# run simulations
final_bxs = []
final_bys = []
for dt in dts:
	bx = initial_bx
	by = initial_by
	t = 0.0
	while t < max_time-1e-10:
		bu = u_function(bx, by, t)
		bv = v_function(bx, by, t)
		if t == 0:
			bx = bx + dt*bu
			by = by + dt*bv
		else:
			bx = bx + 0.5*dt*(3*bu-buo)
			by = by + 0.5*dt*(3*bv-bvo)
		t += dt
		buo = bu
		bvo = bv
	final_bxs.append(bx)
	final_bys.append(by)
# analyze the error
diffs = []
for first, second in zip(final_bxs[:-1], final_bxs[1:]):
	diffs.append(np.abs(first-second).max())
diffs = np.array(diffs)
rats = diffs[:-1]/diffs[1:]
print('Mean ratio: {:0.1f}'.format(np.mean(rats)))

################################################################################
# Test dumb advection of boundary with reparametrization
print('')
print('With dumb reparametrization')

# run simulations
final_bxs = []
final_bys = []
for dt in dts:
	bx = initial_bx
	by = initial_by
	t = 0.0
	while t < max_time-1e-10:
		bu = u_function(bx, by, t)
		bv = v_function(bx, by, t)
		if t == 0:
			bx = bx + dt*bu
			by = by + dt*bv
		else:
			bx = bx + 0.5*dt*(3*bu-buo)
			by = by + 0.5*dt*(3*bv-bvo)
		bx, by = arc_length_parameterize(bx, by)
		t += dt
		buo = bu
		bvo = bv
	final_bxs.append(bx)
	final_bys.append(by)
# analyze the error
diffs = []
for first, second in zip(final_bxs[:-1], final_bxs[1:]):
	diffs.append(np.abs(first-second).max())
diffs = np.array(diffs)
rats = diffs[:-1]/diffs[1:]
print('Mean ratio: {:0.1f}'.format(np.mean(rats)))

################################################################################
# Test smart advection of boundary with reparametrization
print('')
print('With smart reparametrization')

# run simulations
final_bxs = []
final_bys = []
for dt in dts:
	bx = initial_bx
	by = initial_by
	t = 0.0
	while t < max_time-1e-10:
		bu = u_function(bx, by, t)
		bv = v_function(bx, by, t)
		if t == 0:
			bx = bx + dt*bu
			by = by + dt*bv
		else:
			bx = bx + 0.5*dt*(3*bu-buo)
			by = by + 0.5*dt*(3*bv-bvo)
		bx, by, new_t = arc_length_parameterize(bx, by, return_t=True)
		t += dt
		bu_interp = interp1d(0, 2*np.pi, bdy.dt, bu, p=True)
		bv_interp = interp1d(0, 2*np.pi, bdy.dt, bv, p=True)
		buo = bu_interp(new_t)
		bvo = bv_interp(new_t)
	final_bxs.append(bx)
	final_bys.append(by)
# analyze the error
diffs = []
for first, second in zip(final_bxs[:-1], final_bxs[1:]):
	diffs.append(np.abs(first-second).max())
diffs = np.array(diffs)
rats = diffs[:-1]/diffs[1:]
print('Mean ratio: {:0.1f}'.format(np.mean(rats)))

################################################################################
# Test smart advection of boundary with reparametrization (BDF Type for consistency...)
print('')
print('With smart reparametrization, BDF STYLE')

# run simulations
final_bxs = []
final_bys = []
for dt in dts:
	bx = initial_bx
	by = initial_by
	t = 0.0
	while t < max_time-1e-10:
		bu = u_function(bx, by, t)
		bv = v_function(bx, by, t)
		_bx, _by = bx, by
		if t == 0:
			bx = bx + dt*bu
			by = by + dt*bv
		else:
			bx = 4*bx/3.0 - bxo/3.0 + dt*(4*bu-2*buo)/3.0
			by = 4*by/3.0 - byo/3.0 + dt*(4*bv-2*bvo)/3.0
		bx, by, new_t = arc_length_parameterize(bx, by, return_t=True)
		t += dt
		bu_interp = interp1d(0, 2*np.pi, bdy.dt, bu, p=True)
		bv_interp = interp1d(0, 2*np.pi, bdy.dt, bv, p=True)
		buo = bu_interp(new_t)
		bvo = bv_interp(new_t)
		bx_interp = interp1d(0, 2*np.pi, bdy.dt, _bx, p=True)
		by_interp = interp1d(0, 2*np.pi, bdy.dt, _by, p=True)
		bxo = bx_interp(new_t)
		byo = by_interp(new_t)
	final_bxs.append(bx)
	final_bys.append(by)
# analyze the error
diffs = []
for first, second in zip(final_bxs[:-1], final_bxs[1:]):
	diffs.append(np.abs(first-second).max())
diffs = np.array(diffs)
rats = diffs[:-1]/diffs[1:]
print('Mean ratio: {:0.1f}'.format(np.mean(rats)))




