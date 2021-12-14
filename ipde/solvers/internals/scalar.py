import numpy as np
from ...annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry

class ScalarHelper(object):
    """
    General class for scalar solvers
    """
    def __init__(self, ebdy, helper=None, grid_backend='pybie2d'):
        self.ebdy = ebdy
        self.grid_backend = grid_backend
        self.interior = self.ebdy.interior
        if helper is None:
            self.AAG = ApproximateAnnularGeometry(self.ebdy.bdy.N, self.ebdy.M,
                self.ebdy.radial_width, self.ebdy.approximate_radius)
            self._define_annular_solver()
        else:
            self.annular_solver = helper.annular_solver
        self._set_boundary_estimators()
        self._get_RAG()
        self._get_qfs()
        self._define_layer_apply()
    def _get_geometry(self):
        raise NotImplementedError
    def _define_annular_solver(self):
        raise NotImplementedError
    def _get_qfs(self):
        raise NotImplementedError
    def _define_layer_apply(self):
        raise NotImplementedError
    def _get_RAG(self):
        bb = self.ebdy.bdy if self.interior else self.ebdy.interface
        sp = bb.speed
        cur = bb.curvature
        self.RAG = RealAnnularGeometry(sp, cur, self.annular_solver.AAG)
    def _set_boundary_estimators(self):
        CO = self.annular_solver.AAG.CO
        if self.interior:
            self._bv_estimator = CO.obc_dirichlet[0]
            self._bn_estimator = CO.obc_neumann[0]
            self._iv_estimator = CO.ibc_dirichlet[0]
            self._in_estimator = CO.ibc_neumann[0]
        else:
            self._bv_estimator = CO.ibc_dirichlet[0]
            self._bn_estimator = CO.ibc_neumann[0]
            self._iv_estimator = CO.obc_dirichlet[0]
            self._in_estimator = CO.obc_neumann[0]
    def get_boundary_values(self, ur):
        """
        Return ur evaluated on the boundary
        """
        return self._bv_estimator.dot(ur)
    def get_boundary_normal_derivatives(self, ur):
        """
        Return du/dn evaluated on the boundary
        """
        return self._bn_estimator.dot(ur)
    def get_interface_values(self, ur):
        """
        Return ur evaluated on the interface
        """
        return self._iv_estimator.dot(ur)
    def get_interface_normal_derivatives(self, ur):
        """
        Return du/dn evaluated on the interior
        """
        return self._in_estimator.dot(ur)
    def __call__(self, fr, bv, bx, by, **kwargs):
        """
        kwargs are to be passed onto the annular solver
        """
        ebdy = self.ebdy
        ucn = bx*ebdy.interface.normal_x + by*ebdy.interface.normal_y
        # compute the radial solution
        zer = np.zeros_like(bv)
        ur = self.annular_solver.solve(self.RAG, fr, zer, zer, **kwargs)
        self.iterations_last_call = self.annular_solver.iterations_last_call
        # evaluate the normal derivative of the radial solution
        urn = self.get_interface_normal_derivatives(ur)
        # get the single layer to smooth the interface
        slp = urn - ucn
        dlp = bv
        if not self.interior:
            slp *= -1.0
            dlp *= -1.0
        # get effective layer potentials for this
        sigma_g = self.interface_qfs_g([slp, dlp])
        sigma_r = self.interface_qfs_r([slp, dlp])
        # save ur, sigma_r, sigma_g in an internal state
        self.ur = ur
        self.sigma_r = sigma_r
        self.sigma_g = sigma_g
        # we now report back to master
        return sigma_g
    def correct(self, ub):
        # evaluate the effect of all sigma_g except ours, as well as the effect
        # of our own sigma_r, onto the radial points

        # First, we get the u on our interface associate with our sigma_g
        src = self.interface_qfs_g.source
        w = self.Layer_Apply(src, self.ebdy.interface, self.sigma_g)
        # now subtract this from the given ub
        ub = ub - w

        # okay, now we want a density on our interface_radial_source that gives us this
        sigma_r_adj = self.interface_qfs_r.u2s(ub)

        # add this to the previous sigma_r
        sigma_r_tot = sigma_r_adj + self.sigma_r

        # evaluate this onto radial points and add to ur
        src = self.interface_qfs_r.source
        rslp = self.Layer_Apply(src, self.ebdy.radial_targ, sigma_r_tot)
        self.ur += rslp.reshape(self.ur.shape)

        return self.ur
