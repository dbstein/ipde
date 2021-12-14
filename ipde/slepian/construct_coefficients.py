"""
This file to be used to construct Chebyshev coefficients for
Heaviside and Bump functions

Perhaps redo this to compute step functions directly?  Without
intermediary representation in SlepianMollifier; but probably doesn't matter.

Maybe should also just rerun with more points in SlepianMollifier calls...
"""

import numpy as np
from ipde.slepian.function_generator_bump_step import SlepianMollifier

def fit_chebyshev(f, order):
	x, w = np.polynomial.chebyshev.chebgauss(order)
	c = np.polynomial.chebyshev.chebfit(x, f(x), order-1)
	return c
def fit_chebyshev_even(f, order):
	x, w = np.polynomial.chebyshev.chebgauss(order)
	c = np.polynomial.chebyshev.chebfit(x, f(x), order-1)
	return c[::2]
def fit_and_check_chebyshev(f, order):
	c = fit_chebyshev_even(f, order)
	return np.abs(c[-1]).max() > 1e-14

# let's generate data for different values of slepian_r
with open('heaviside_coefficients_raw.py', 'a') as the_file:
	for slepian_r in range(1, 201):
		print(slepian_r)
		S = SlepianMollifier(slepian_r)
		bump_order = 8
		while fit_and_check_chebyshev(S.bump, bump_order):
			bump_order += 2
		step_order = 8
		while fit_and_check_chebyshev(lambda x: (S.step(x)-0.5)/x, step_order):
			step_order += 2
		bump_c_even = fit_chebyshev_even(S.bump, bump_order)
		step_c_even = fit_chebyshev_even(lambda x: (S.step(x)-0.5)/x, step_order)
		bump_c = fit_chebyshev(S.bump, bump_order)
		step_c = fit_chebyshev(S.step, step_order)
		bump_str = 'bump_c_' + str(slepian_r) + ' = np.array(' + np.array2string(bump_c, separator=', ', precision=16) + ')'
		step_str = 'step_c_' + str(slepian_r) + ' = np.array(' + np.array2string(step_c, separator=', ', precision=16) + ')'
		bump_even_str = 'bump_c_even_' + str(slepian_r) + ' = np.array(' + np.array2string(bump_c_even, separator=', ', precision=16) + ')'
		step_even_str = 'step_c_even_' + str(slepian_r) + ' = np.array(' + np.array2string(step_c_even, separator=', ', precision=16) + ')'
		the_file.write(bump_str + '\n')
		the_file.write(bump_even_str + '\n')
		the_file.write(step_str + '\n')
		the_file.write(step_even_str + '\n')

