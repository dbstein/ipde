import numpy as np

dts = 0.1 / 2**np.arange(10)
res = np.empty_like(dts)

max_time = 1.0
initial_c = 1.0
reaction_value = 0.5
nonlinearity = lambda x: x*np.cos(x)
scaled_nonlinearity = lambda x: reaction_value*nonlinearity(x)
implicit_term  = 1.0  # value for implicit term to be solved for

for dti, dt in enumerate(dts):

	t = 0
	c = initial_c
	while t < max_time-1e-10:
		if t == 0:
			c_star = c - dt*scaled_nonlinearity(c)
			c_new = c_star / (1+implicit_term*dt)
		else:
			c_star = (1/3)*(4*c-c_old) - (2/3)*dt*(2*scaled_nonlinearity(c) - scaled_nonlinearity(c_old)) 
			c_new = c_star / (1+implicit_term*dt*(2/3))
		c_old = c
		c = c_new
		t += dt

	res[dti] = c

err = res[1:] - res[:-1]
rat = err[:-1]/err[1:]

print(rat)
