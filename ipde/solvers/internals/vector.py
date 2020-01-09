import numpy as np
from ...annular.annular import ApproximateAnnularGeometry, RealAnnularGeometry

def v2f(x):
    return x.reshape(2, x.size//2)

class VectorHelper(object):
    """
    General class for vector helpers
    """
    def __init__(self, ebdy, annular_solver=None, **kwargs):
        self.ebdy = ebdy
        self.interior = self.ebdy.interior
        self._extract_extra_kwargs(**kwargs)
        if annular_solver is None:
            self.AAG = ApproximateAnnularGeometry(self.ebdy.bdy.N, self.ebdy.M,
                self.ebdy.radial_width, self.ebdy.approximate_radius)
            self._define_annular_solver()
        else:
            self.annular_solver = annular_solver
        self._set_boundary_estimators()
        self._get_RAG()
        self._get_qfs()
        self._define_layer_apply()
    def _extract_extra_kwargs(self, **kwargs):
        pass
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
        CO = self.AAG.CO
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
    def get_boundary_values(self, fr):
        """
        Return fr evaluated on the boundary
        """
        return self._bv_estimator.dot(fr)
    def get_interface_values(self, fr):
        """
        Return fr evaluated on the interface
        """
        return self._iv_estimator.dot(fr)
    def get_boundary_traction_uvp(self, u, v, p):
        """
        Return T(U), in x, y coords, on bdy
        given radial u, v, p
        """
        return self._get_traction_uvp(u, v, p, self._bv_estimator)
    def get_boundary_traction_rtp(self, Ur, Ut, p):
        """
        Return T(U), in r, t coords, on bdy
        given radial Ur, Ut, p
        """
        return self._get_traction_rtp(Ur, Ut, p, self._bv_estimator)
    def get_interface_traction_uvp(self, u, v, p):
        """
        Return T(U), in x, y coords, on bdy
        given radial u, v, p
        """
        return self._get_traction_uvp(u, v, p, self._iv_estimator)
    def get_interface_traction_rtp(self, Ur, Ut, p):
        """
        Return T(U), in r, t coords, on interface
        given radial Ur, Ut, p
        """
        return self._get_traction_rtp(Ur, Ut, p, self._iv_estimator)
    def _get_traction_uvp(self, u, v, p, estimator):
        Ur, Ut = self.ebdy.convert_uv_to_rt(u, v)
        Tr, Tt = self._get_traction_rtp(Ur, Ut, p, estimator)
        return self.ebdy.convert_rt_to_uv(Tr, Tt)
    def _get_traction_rtp(self, Ur, Ut, p, estimator):
        ebdy = self.ebdy
        Urr = ebdy._radial_grid_r_derivative(Ur)
        Urt = ebdy._radial_grid_tau_derivative(Ur)
        Utr = ebdy.radial_speed*ebdy._radial_grid_r_derivative(Ut*ebdy.inverse_radial_speed)
        Urr_b = estimator.dot(Urr)
        Urt_b = estimator.dot(Urt)
        Utr_b = estimator.dot(Utr)
        p_b   = estimator.dot(p)
        Tr = 2*Urr_b - p_b
        Tt = Utr_b + Urt_b
        return Tr, Tt
    def __call__(self, fur, fvr, bu, bv, btxx, btxy, btyy, **kwargs):
        ebdy = self.ebdy
        # get traction due to grid solution
        btx = btxx*ebdy.interface.normal_x + btxy*ebdy.interface.normal_y
        bty = btxy*ebdy.interface.normal_x + btyy*ebdy.interface.normal_y
        # compute the radial solution (using homogeneous 0 BCs)
        fr, ft = ebdy.convert_uv_to_rt(fur, fvr)
        zer = np.zeros(ebdy.bdy.N)
        rr, tr, pr = self.annular_solver.solve(self.RAG, fr, ft, zer, zer, zer, zer, **kwargs)
        # put this into u, v
        ur, vr = ebdy.convert_rt_to_uv(rr, tr)
        # get the traction due to the radial solution
        rtx, rty = self.get_interface_traction_uvp(ur, vr, pr)
        # get the traction jump for the slp
        taus = np.concatenate([ rtx-btx, rty-bty ])
        taud = np.concatenate([ bu,      bv      ])
        # adjust for interior
        if not self.interior:
            taus *= -1.0
            taud *= -1.0
        # get effective layer potentials
        sigma_g = v2f(self.interface_qfs_g([taus, taud]))
        sigma_r = v2f(self.interface_qfs_r([taus, taud]))
        # save ur, sigma_r, sigma_g in an internal state
        self.ur = ur
        self.vr = vr
        self.pr = pr
        self.sigma_r = sigma_r
        self.sigma_g = sigma_g
        # we now report back to master
        return sigma_g
    def correct(self, ub, vb):
        # evaluate the effect of all sigma_g except ours, as well as the effect
        # of our own sigma_r, onto the radial points

        # First, we get the u on our interface associate with our sigma_g
        w = self.Layer_Apply(self.ebdy.interface_grid_source, self.ebdy.interface, self.sigma_g)
        # now subtract this from the given ub
        ub = ub - w[0]
        vb = vb - w[1]
        Ub = np.concatenate([ub, vb])

        # okay, now we want a density on our interface_radial_source that gives us this
        sigma_r_adj = v2f(self.interface_qfs_r.u2s(Ub))

        # add this to the previous sigma_r
        sigma_r_tot = sigma_r_adj + self.sigma_r

        # evaluate this onto radial points and add to ur
        rslp = self.Layer_Apply(self.ebdy.interface_radial_source, self.ebdy.radial_targ, sigma_r_tot)
        self.ur += rslp[0].reshape(self.ur.shape)
        self.vr += rslp[1].reshape(self.ur.shape)

        return self.ur, self.vr
