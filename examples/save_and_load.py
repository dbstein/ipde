import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import pybie2d
from ipde.embedded_boundary import EmbeddedBoundary, LoadEmbeddedBoundary
from ipde.ebdy_collection import EmbeddedBoundaryCollection, LoadEmbeddedBoundaryCollection
from ipde.embedded_function import EmbeddedFunction, LoadEmbeddedFunction
from ipde.heavisides import SlepianMollifier
star = pybie2d.misc.curve_descriptions.star
squish = pybie2d.misc.curve_descriptions.squished_circle
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
Grid = pybie2d.grid.Grid

nb = 300
M = 12
slepian_r = 2*M

# get heaviside function
MOL = SlepianMollifier(slepian_r)
# construct boundary
bdy = GSB(c=star(nb, a=0.2, f=5))
bh = bdy.dt*bdy.speed.min()
# construct embedded boundary
ebdy = EmbeddedBoundary(bdy, True, M, bh*1, heaviside=MOL.step)
ebdys = [ebdy,]
ebdyc = EmbeddedBoundaryCollection([ebdy,])
# register the grid
print('\nRegistering the grid')
# ebdyc.register_grid(grid)
ebdyc.generate_grid(bh, bh)

# construct an embedded function
f = EmbeddedFunction(ebdyc)
f.define_via_function(lambda x, y: np.sin(x)*np.exp(y))

# try saving ebdy
ebdy_dict = ebdy.save()
ebdy2 = LoadEmbeddedBoundary(ebdy_dict)

# try saving ebdyc
ebdyc_dict = ebdyc.save()
ebdyc2 = LoadEmbeddedBoundaryCollection(ebdyc_dict)

# try saving f
f_full_dict = f.full_save()
f_dict = f.save()
f2, _ = LoadEmbeddedFunction(f_dict, ebdyc2)
f3, ebdyc3 = LoadEmbeddedFunction(f_full_dict)

# now lets try saving these to file.
import pickle
pickle.dump( [ebdy_dict, ebdyc_dict, f_full_dict, f_dict], open( "save.p", "wb" ) )



