import numpy as np
from .internals.stationary import StationaryHelper
from .internals.moving import MovingHelper

class Advector(object):
    """
    General class for semi-lagrangian advection
    """
    def __init__(self, ebdyc, typelist=['stationary',]*len(ebdyc)):
        """
        Type can either be 'spectral' or 'fourth'
        """
        self.ebdyc = ebdyc
        self.typelist = typelist
        self.helpers = []
        grid = ebdyc.grid
        for atype, ebdy in zip(typelist, ebdyc.ebdys):
            if atype is 'stationary':
                helper = StationaryHelper(ebdy, grid)
            else:
                helper = MovingHelper(ebdy, grid)
            self.helpers.append(helper)
    def set_velocity(self, u, v, urs, vrs):
        pass
    def advect_function(self, f, frs):
        pass
