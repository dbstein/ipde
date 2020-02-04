import numpy as np

def f(x):
	return np.exp(np.sin(x))

M = 5
Ntest = 100
hs = 1.0 / 2**np.arange(50)

snodes = np.linspace(-1, 1, M)
MAT = np.empty([M,M], dtype=float)
for i in range(M):
	MAT[:,i] = np.power(snodes,i)
IMAT = np.linalg.inv(MAT)

print('')
print('Order is:           ', M)
print('Condition number is: {:0.1f}'.format(np.linalg.cond(MAT)))
print('Errors: ')

x = np.random.rand(Ntest)

for h in hs:

	nodes = snodes * h
	data = f(nodes)
	coefs = IMAT.dot(data)

	truth = f(x*h)
	pred = coefs[0]
	for i in range(1,M):
		pred += coefs[i]*x**i
	err = truth - pred
	max_err = np.abs(err).max()

	print('h: {:0.1e}'.format(h), 'Error: {:0.1e}'.format(max_err))

