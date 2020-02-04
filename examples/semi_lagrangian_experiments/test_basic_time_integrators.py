import numpy as np

"""
we will test solving the ODE df/dt = f + t*np.cos(f); f(0) = 1
our reference solution will be done using 4th order RK with a small timestep
"""

max_time = 1.0
dts = 0.1 / 2**np.arange(0, 7)

def timestep(y0, max_time, dt, timestepper):
	t = 0.0
	dt = max_time/int(np.ceil(max_time/dt))
	y = y0
	while t < max_time-1e-10:
		y, t = timestepper(y, t, dt)
	return y
def analyze(errors, name):
	errors = np.array(errors)
	ratios = errors[:-1] / errors[1:]
	mean_ratio = np.mean(ratios)
	convergence_rate = np.log2(mean_ratio)
	print(('Convergence rate for ' + name + ' is: ').ljust(45), '{:0.2f}'.format(convergence_rate) + ';', '   Final Error: {:0.2e}'.format(errors[-1]))
def full_f(t, y):
	return y+t*np.cos(y)
def explicit_f(t, y):
	return t*np.cos(y)
def test(timestepper, name):
	errors = []
	for ind, dt in enumerate(dts):
		y = timestep(1.0, 1.0, dt, timestepper)
		errors.append(np.abs(y-y_reference))
	analyze(errors, name)

################################################################################
# Get the reference solution

def rk4(y, t, dt):
	"""
	Take one timestep of RK4
	y (float):     current value of y
	t  (float):    current time
	dt (float):    timestep
	"""
	f = full_f
	k1 = dt*f( t,      y      )
	k2 = dt*f( t+dt/2, y+k1/2 )
	k3 = dt*f( t+dt/2, y+k2/2 )
	k4 = dt*f( t+dt,   y+k3   )
	return y + (k1 + 2*k2 + 2*k3 + k4)/6, t + dt

y_reference = timestep(1.0, 1.0, 1e-3, rk4)

""" RK4 Diagnostics
dt  = [1e-1, 1e-2,   1e-3,    1e-4     1e-5    ]
dif = [----, 1.1e-6, 1.2e-10, 4.4e-14, 2.5e-13 ]
we thus take dt = 1e-3, and consider the solution good to ~13 digits
"""

print('')

################################################################################
# Convergence study for Forward Euler

def forward_euler(y, t, dt):
	return y + dt*full_f(t, y), t + dt
test(forward_euler, 'Forward Euler')

################################################################################
# Convergence study for RK2 (midpoint method)

def rk2_midpoint(y, t, dt):
	f = full_f
	k1 = dt*f(t, y)
	return y + dt*f(t + dt/2, y + k1/2), t + dt
test(rk2_midpoint, 'RK 2, Midpoint')

################################################################################
# Convergence study for RK2 (Ralston)

def rk2_ralston(y, t, dt):
	f = full_f
	k1 = dt*f(t,          y          )
	k2 = dt*f(t + 2*dt/3, y + 2*k1/3 )
	return y + (k1 + 3*k2)/4, t + dt
test(rk2_ralston, 'RK 2, Ralston')

################################################################################
# Convergence study for RK3 (Kutta)

def rk3_kutta(y, t, dt):
	f = full_f
	k1 = dt*f(t,        y             )
	k2 = dt*f(t + dt/2, y + k1/2      )
	k3 = dt*f(t + dt,   y - k1 + 2*k2 )
	return y + (k1 + 4*k2 + k3)/6, t + dt
test(rk3_kutta, 'RK 3, Kutta')

################################################################################
# Convergence study for RK3 (Heun)

def rk3_heun(y, t, dt):
	f = full_f
	k1 = dt*f(t,          y          )
	k2 = dt*f(t +   dt/3, y +   k1/3 )
	k3 = dt*f(t + 2*dt/3, y + 2*k2/3 )
	return y + (k1 + 3*k3)/4, t + dt
test(rk3_heun, 'RK 3, Heun')

################################################################################
# Convergence study for RK3 (Ralston)

def rk3_ralston(y, t, dt):
	f = full_f
	k1 = dt*f(t,          y          )
	k2 = dt*f(t +   dt/2, y +   k1/2 )
	k3 = dt*f(t + 3*dt/4, y + 3*k2/4 )
	return y + (2*k1 + 3*k2 + 4*k3)/9, t + dt
test(rk3_ralston, 'RK 3, Ralston')

################################################################################
# Convergence study for RK3 (SSPRK)

def rk3_ssprk(y, t, dt):
	f = full_f
	k1 = dt*f(t,        y               )
	k2 = dt*f(t + dt,   y + k1          )
	k3 = dt*f(t + dt/2, y + k1/4 + k2/4 )
	return y + (k1 + k2 + 4*k3)/6, t + dt
test(rk3_ssprk, 'RK 3, SSPRK')

################################################################################
# Convergence study for RK4

test(rk4, 'RK 4, Classical')

################################################################################
# Convergence study for RK-4 3/8th Rule

def rk4_38(y, t, dt):
	f = full_f
	k1 = dt*f( t,          y                  )
	k2 = dt*f( t +   dt/3, y + k1/3           )
	k3 = dt*f( t + 2*dt/3, y - k1/3 + k2      )
	k4 = dt*f( t +   dt,   y + k1   - k2 + k3 )
	return y + (k1 + 3*k2 + 3*k3 + k4)/8, t + dt
test(rk4_38, 'RK 4, 3/8th')

################################################################################
# Convergence study for Embedded Scheme

def bogacki_shampine_internal(y, t, dt):
	f = full_f
	k1 = dt*f(t,          y                          )
	k2 = dt*f(t + dt/2,   y +   k1/2                 )
	k3 = dt*f(t + 3*dt/4, y + 3*k2/4                 )
	k4 = dt*f(t + dt,     y + 2*k1/9 + k2/3 + 4*k3/9 )
	return k1, k2, k3, k4
def bogacki_shampine_3(y, t, dt):
	k1, k2, k3, k4 = bogacki_shampine_internal(y, t, dt)
	return y + (2*k1 + 3*k2 + 4*k3)/9, t + dt
def bogacki_shampine_2(y, t, dt):
	k1, k2, k3, k4 = bogacki_shampine_internal(y, t, dt)
	return y + (7*k1 + 6*k2 + 8*k3 + 3*k4)/24, t + dt

test(bogacki_shampine_2, 'Bogacki-Shampine 2')
test(bogacki_shampine_3, 'Bogacki-Shampine 3')

################################################################################
# Convergence study for IMEX-RK L-Stable ARS(2,2,2)

def ars222(y, t, dt):
	g = 1 - np.sqrt(2)/2
	d = 1 - 1/(2*g)
	ef = explicit_f
	e1 = ef(t, y)
	i1 = 0.0
	k1 = y + dt*g1*e1
	l1 = k1/(1-dt*g)
	


	k1 = 
	p1 = y + dt*ef(t, y)
	y1 = p1/(1-g)
	p2 = 


	k2 = dt*f( t +   dt/3, y + k1/3           )
	k3 = dt*f( t + 2*dt/3, y - k1/3 + k2      )
	k4 = dt*f( t +   dt,   y + k1   - k2 + k3 )
	return y + (k1 + 3*k2 + 3*k3 + k4)/8, t + dt
test(rk4_38, 'RK 4, 3/8th')




