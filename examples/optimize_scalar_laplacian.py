import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier
from ipde.annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 2000
ng = int(nb/2)
M = 50
helmholtz_k = 10.0
pad_zone = 2
interior = False
slepian_r = 20

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.1, f=5))
# construct a grid
grid = Grid([-1.5, 1.5], ng, [-1.5, 1.5], ng, x_endpoints=[True, False], y_endpoints=[True, False])
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, interior, M, grid.xh*0.75, pad_zone, MOL.step)
# register the grid
print('\nRegistering the grid')
ebdy.register_grid(grid)

################################################################################
# Extract radial information from ebdy and construct annular solver

def mfft(f):
    M = f.shape[0]
    N = f.shape[1]
    NS = N - 1
    N2 = int(N/2)
    fh = np.fft.fft(f)
    temp = np.empty((M, NS), dtype=complex)
    temp[:,:N2] = fh[:,:N2]
    temp[:,N2:] = fh[:,N2+1:]
    return temp
def mifft(fh):
    M = fh.shape[0]
    NS = fh.shape[1]
    N = NS + 1
    N2 = int(N/2)
    temp = np.empty((M, N), dtype=complex)
    temp[:,:N2]   = fh[:,:N2]
    temp[:,N2]    = 0.0
    temp[:,N2+1:] = fh[:,N2:]
    return np.fft.ifft(temp)
def fourier_multiply(fh, m):
    return mfft(m*mifft(fh))

def mrfft(f):
    M = f.shape[0]
    N = f.shape[1]
    N2 = int(N/2)
    fh = np.fft.rfft(f)
    temp = np.empty((M, N2), dtype=complex)
    temp[:,:N2] = fh[:,:N2]
    return temp
def mrifft(fh):
    M = fh.shape[0]
    NS = fh.shape[1]
    N = 2*NS
    temp = np.empty((M, NS+1), dtype=complex)
    temp[:,:NS]   = fh[:,:NS]
    temp[:,NS]    = 0.0
    return np.fft.irfft(temp)
def rfourier_multiply(fh, m):
    w = mrifft(fh)
    np.multiply(w, m, out=w)
    return mrfft(w)

def scalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    iks = AAG.iks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    uh_t = R01.dot(uh*iks)
    uh_tt = R12.dot(fourier_multiply(uh_t, ipsi1)*iks)
    uh_rr = D12.dot(fourier_multiply(D01.dot(uh), psi1))
    luh = fourier_multiply(uh_rr+uh_tt, ipsi2)
    return luh

def optim_scalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    riks = AAG.riks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    w0sh = uh.shape
    w1sh = uh.shape[0]-1, uh.shape[1]
    w2sh = uh.shape[0]-2, uh.shape[1]
    w0 = np.empty(w0sh, dtype=complex)
    w1 = np.empty(w1sh, dtype=complex)
    w2a = np.empty(w2sh, dtype=complex)
    w2b = np.empty(w2sh, dtype=complex)
    np.multiply(uh, riks, out=w0)
    np.dot(R01, w0, out=w1) 
    w1b = rfourier_multiply(w1, ipsi1)
    np.multiply(w1b, riks, out=w1)
    np.dot(R12, w1, out=w2a)
    np.dot(D01, uh, out=w1)
    w1b = rfourier_multiply(w1, psi1)
    np.dot(D12, w1b, out=w2b)
    np.add(w2a, w2b, out=w2a)
    luh = rfourier_multiply(w2a, ipsi2)
    return luh

# get the forces and BCs for the problem
x, y = ebdy.radial_x, ebdy.radial_y
k = 2*np.pi/3
solution_func = lambda x, y: np.exp(np.sin(k*x))*np.sin(k*y)
force_func = lambda x, y: helmholtz_k**2*solution_func(x, y)-k**2*np.exp(np.sin(k*x))*np.sin(k*y)*(np.cos(k*x)**2-np.sin(k*x)-1.0)
force = force_func(x, y)
AAG = ApproximateAnnularGeometry(nb, M, ebdy.radial_width, ebdy.approximate_radius)
sp =  ebdy.bdy.speed     if interior else ebdy.interface.speed
cur = ebdy.bdy.curvature if interior else ebdy.interface.curvature
RAG = RealAnnularGeometry(sp, cur, AAG)
CO = AAG.CO
AAG.riks = AAG.iks[:int(nb/2)]

fh = mfft(force)
rfh = mrfft(force)

import line_profiler
%load_ext line_profiler
%lprun -f scalar_laplacian scalar_laplacian(CO, AAG, RAG, fh)
%lprun -f optim_scalar_laplacian optim_scalar_laplacian(CO, AAG, RAG, rfh)

o1 = mifft(scalar_laplacian(CO, AAG, RAG, fh))
o2 = mrifft(optim_scalar_laplacian(CO, AAG, RAG, rfh))
print(np.allclose(o1, o2))

# %lprun -f mfft scalar_laplacian(CO, AAG, RAG, fh2)
# %lprun -f optim_scalar_laplacian optim_scalar_laplacian(CO, AAG, RAG, fh2)


