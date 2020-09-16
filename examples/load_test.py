import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
from ipde.embedded_boundary import LoadEmbeddedBoundary
from ipde.ebdy_collection import LoadEmbeddedBoundaryCollection
from ipde.embedded_function import LoadEmbeddedFunction

# read in data
[ebdy_dict, ebdyc_dict, f_full_dict, f_dict] = pickle.load( open( "save.p", "rb" ) )

# load ebdy
ebdy = LoadEmbeddedBoundary(ebdy_dict)
# load ebdyc
ebdyc = LoadEmbeddedBoundaryCollection(ebdyc_dict)
# set ebdy to ebdyc[0]
ebdy = ebdyc[0]
# load f from full dict
f1, ebdyc1 = LoadEmbeddedFunction(f_full_dict)
# load f from not full dict
f2, _ = LoadEmbeddedFunction(f_dict, ebdyc)

# take the gradient of f
fx, fy = ebdyc.gradient(f2)

# plot fx, fy
fig, [ax0, ax1] = plt.subplots(1,2)
fx.plot(ax0, cmap=mpl.cm.jet)
fy.plot(ax1, cmap=mpl.cm.jet)
