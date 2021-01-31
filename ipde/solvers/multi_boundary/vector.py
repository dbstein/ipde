import numpy as np
from ...derivatives import fourier, fd_x_4, fd_y_4
# from ...ebdy_collection import EmbeddedFunction, BoundaryFunction
from ...ebdy_collection import BoundaryFunction
from ...embedded_function import EmbeddedFunction

class VectorSolver(object):
    def __init__(self, ebdyc, solver_type='spectral', AS_list=None, **kwargs):
        self.ebdyc = ebdyc
        self.solver_type = solver_type
        if AS_list is None: AS_list = [None,]*self.ebdyc.N
        self.AS_list = AS_list
        self._extract_extra_kwargs(**kwargs)
        self.helpers = []
        for ebdy, AS in zip(self.ebdyc.ebdys, AS_list):
            self.helpers.append(self._get_helper(ebdy, AS))
        self.AS_list = [helper.annular_solver for helper in self.helpers]
        # compute necessary spectral operators
        self.grid = self.ebdyc.grid
        self.kx, self.ky = self.ebdyc.kx, self.ebdyc.ky
        self.ikx, self.iky = self.ebdyc.ikx, self.ebdyc.iky
        self.lap = -self.kx*self.kx - self.ky*self.ky
        self._get_specific_operators()
        # set derivative method
        self._set_derivative_method()
        # set interpolation order
        self.interpolation_order = 3 if self.solver_type == 'fourth' else np.Inf
        # extract the grid smoothed step function
        self.grid_step = self.ebdyc.grid_step
        # define the Layer_Apply function
        self._define_layer_apply()
    def _extract_extra_kwargs(self, **kwargs):
        pass
    def _get_helper(self, ebdy, AS):
        raise NotImplementedError
    def _grid_solve(self, fc):
        raise NotImplementedError
    def _get_specific_operators(self):
        pass
    def _set_derivative_method(self):
        if self.solver_type == 'spectral':
            self.dx = lambda x: fourier(x, self.ikx)
            self.dy = lambda x: fourier(x, self.iky)
        else:
            self.dx = lambda x: fd_x_4(x, self.ebdyc.grid.xh)
            self.dy = lambda x: fd_y_4(x, self.ebdyc.grid.yh)
    def get_boundary_values(self, frs):
        return np.concatenate([helper.get_boundary_values(fr) for fr, helper in zip(frs, self.helpers)])
    def get_boundary_tractions(self, urs, vrs, prs):
        return np.concatenate([helper.get_boundary_traction_uvp(ur, vr, pr) for ur, vr, pr, helper in zip(urs, vrs, prs, self.helpers)])
    def __call__(self, fu, fv, **kwargs):
        # separate the components
        fuc = fu.get_smoothed_grid_value()
        fvc = fv.get_smoothed_grid_value()
        fur_list = fu.get_radial_value_list()
        fvr_list = fv.get_radial_value_list()
        uc, vc, pc = self._grid_solve(fuc, fvc)
        # interpolate the solution to the interface
        bus = self.ebdyc.interpolate_grid_to_interface(uc, order=self.interpolation_order, cutoff=False)
        bvs = self.ebdyc.interpolate_grid_to_interface(vc, order=self.interpolation_order, cutoff=False)
        # get the grid solution's derivatives
        ucx, ucy = self.dx(uc), self.dy(uc)
        vcx, vcy = self.dx(vc), self.dy(vc)
        # compute the grid solutions stress
        tcxx = 2*ucx - pc
        tcxy = ucy + vcx
        tcyy = 2*vcy - pc
        # interpolate these to the interface
        btxxs = self.ebdyc.interpolate_grid_to_interface(tcxx, order=self.interpolation_order, cutoff=False)
        btxys = self.ebdyc.interpolate_grid_to_interface(tcxy, order=self.interpolation_order, cutoff=False)
        btyys = self.ebdyc.interpolate_grid_to_interface(tcyy, order=self.interpolation_order, cutoff=False)
        # convert these from lists to vectors
        bul, bvl = self.ebdyc.v2l(bus), self.ebdyc.v2l(bvs)
        btxxl, btxyl, btyyl = self.ebdyc.v2l(btxxs), self.ebdyc.v2l(btxys), self.ebdyc.v2l(btyys)
        # compute the needed layer potentials
        sigmag_list = []
        for helper, fur, fvr, bu, bv, btxx, btxy, btyy in zip(self.helpers, fur_list, fvr_list, bul, bvl, btxxl, btxyl, btyyl):
            sigmag_list.append(helper(fur, fvr, bu, bv, btxx, btxy, btyy, **kwargs))
        # we now need to evaluate this onto the grid / interface points
        sigmag = np.column_stack(sigmag_list)
        out = self.Layer_Apply(self.ebdyc.grid_source, self.ebdyc.grid_pnai, sigmag)
        # need to divide this apart
        gu, bus = self.ebdyc.divide_pnai(out[0])
        gv, bvs = self.ebdyc.divide_pnai(out[1])
        gp, bps = self.ebdyc.divide_pnai(out[2])
        # we can now add gu directly to uc
        uc[self.ebdyc.phys_not_in_annulus] += gu
        vc[self.ebdyc.phys_not_in_annulus] += gv
        pc[self.ebdyc.phys_not_in_annulus] += gp
        # we have to send the bslps back to the indivdual ebdys to deal with them
        single_ebdy = len(self.ebdyc) == 1
        urs, vrs, prs = zip(*[helper.correct(bu, bv, bp, single_ebdy) for helper, bu, bv, bp in zip(self.helpers, bus, bvs, bps)])
        # interpolate urs onto uc
        _ = self.ebdyc.interpolate_radial_to_grid1(urs, uc)
        _ = self.ebdyc.interpolate_radial_to_grid1(vrs, vc)
        _ = self.ebdyc.interpolate_radial_to_grid1(prs, pc)
        uc *= self.ebdyc.phys
        vc *= self.ebdyc.phys
        pc *= self.ebdyc.phys
        u = EmbeddedFunction(self.ebdyc)
        v = EmbeddedFunction(self.ebdyc)
        p = EmbeddedFunction(self.ebdyc)
        u.load_data(uc, urs)
        v.load_data(vc, vrs)
        p.load_data(pc, prs)
        return u, v, p
    def _define_layer_apply(self):
        self.Layer_Apply = self.helpers[0].Layer_Apply

