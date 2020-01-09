import numpy as np

def fd_x_4(f, h, periodic_fix=False):
    fx = np.zeros_like(f)
    iah = 1.0/(12*h)
    fx[2:-2] = -(f[4:] - 8*f[3:-1] + 8*f[1:-3] - f[:-4])*iah
    if periodic_fix:
        fx[ 0] = -(f[2] - 8*f[ 1] + 8*f[-1] - f[-2])*iah
        fx[ 1] = -(f[3] - 8*f[ 2] + 8*f[ 0] - f[-1])*iah
        fx[-1] = -(f[1] - 8*f[ 0] + 8*f[-2] - f[-3])*iah
        fx[-2] = -(f[0] - 8*f[-1] + 8*f[-3] - f[-4])*iah
    return fx

def fd_y_4(f, h, periodic_fix=False):
    fy = np.zeros_like(f)
    iah = 1.0/(12*h)
    fy[:,2:-2] = -(f[:,4:] - 8*f[:,3:-1] + 8*f[:,1:-3] - f[:,:-4])*iah
    if periodic_fix:
        fy[:, 0] = -(f[:,2] - 8*f[:, 1] + 8*f[:,-1] - f[:,-2])*iah
        fy[:, 1] = -(f[:,3] - 8*f[:, 2] + 8*f[:, 0] - f[:,-1])*iah
        fy[:,-1] = -(f[:,1] - 8*f[:, 0] + 8*f[:,-2] - f[:,-3])*iah
        fy[:,-2] = -(f[:,0] - 8*f[:,-1] + 8*f[:,-3] - f[:,-4])*iah
    return fy

def fourier(f, ik):
    fh = np.fft.fft2(f)
    fh *= ik
    return np.fft.ifft2(fh).real
