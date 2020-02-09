import numpy as np

class AdvectionHelper(object):
    def __init__(self, ebdy_new, ebdy, u, v, ux=None, uy=None, vx=None, vy=None,
        ebdy_old=None, uo=None, vo=None, uxo=None, uyo=None, vxo=None, vyo=None):
        """
        Helper class for semi-Lagrangian advection
        """
        self.ebdy_new = ebdy_new
        self.ebdy     = ebdy
        self.ebdy_old = ebdy_old
        self.order = 1 if self.ebdy_old is None else 2
        self.u = u
        self.v = v
        if self.ux is None:
            self.ux, self.uy = ebdy.gradient2(self.u, self.v)
