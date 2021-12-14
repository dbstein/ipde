import numpy as np
import numba

from ipde.slepian.chebeval import numba_chebeval, numba_chebeval_even, numba_step_eval
from ipde.slepian.heaviside_coefficients import bump_list, step_list
from ipde.slepian.heaviside_coefficients import bump_even_list, step_even_list

slepian_r = 100

bump_c = bump_list[slepian_r]
step_c = step_list[slepian_r]
bump_c_even = bump_even_list[slepian_r]
step_c_even = step_even_list[slepian_r]

n = 1000*1000*10
x = 2*np.random.rand(n)-1

o1 = numba_chebeval(x, bump_c)
o2 = numba_chebeval_even(x, bump_c_even)
print('\nError in even evaluation is: {:0.2e}'.format(np.abs(o1-o2).max()))

print('Timing for general evaluation: ')
%timeit numba_chebeval(x, bump_c)
print('Timing for even evaluation: ')
%timeit numba_chebeval_even(x, bump_c_even)

o1 = numba_chebeval(x, step_c)
o2 = numba_step_eval(x, step_c_even)
print('\nError in step evaluation is: {:0.2e}'.format(np.abs(o1-o2).max()))

print('Timing for general evaluation: ')
%timeit numba_chebeval(x, step_c)
print('Timing for even evaluation: ')
%timeit numba_step_eval(x, step_c_even)
