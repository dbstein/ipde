import numpy as np
from ...derivatives import fourier, fd_x_4, fd_y_4
from ...ebdy_collection import BoundaryFunction
from ...embedded_function import EmbeddedFunction

class ScalarSolver(object):
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
    def get_boundary_values(self, urs):
        use_ef = type(urs) == EmbeddedFunction
        # if use_ef:
        #     _, _, urs = urs.get_components()
        bv_list = [helper.get_boundary_values(ur) for ur, helper in zip(urs, self.helpers)]
        if use_ef:
            bv = BoundaryFunction(self.ebdyc)
            bv.load_data(bv_list)
            return bv
        else:
            return np.concatenate(bv_list)
    def get_boundary_normal_derivatives(self, urs):
        return np.concatenate([helper.get_boundary_normal_derivatives(ur) for ur, helper in zip(urs, self.helpers)])
    def __call__(self, f, fr_list=None, **kwargs):
        """
        If fr_list is None, then f should be of type EmbeddedFunction
        (the fr_list option is to be deprecated...)

        Returns either EmbeddedFunction or u/ur_list
        """
        use_ef = type(f) == EmbeddedFunction
        if fr_list is None and not use_ef:
            raise Exception('If fr_list is not provided, f must be of type EmbeddedFunction')
        if use_ef:
            _, fc, fr_list = f.get_components()
            # fc = f.get_smoothed_grid_value()
            # fr_list = f.radial_value_list
        else:
            fc = f*self.grid_step
        # get the grid-based solution
        uc = self._grid_solve(fc)
        # interpolate the solution to the interface
        bvs = self.ebdyc.interpolate_grid_to_interface(uc, order=self.interpolation_order, cutoff=False)
        # get the grid solution's derivatives and interpolate to interface
        ucx, ucy = self.dx(uc), self.dy(uc)
        bxs = self.ebdyc.interpolate_grid_to_interface(ucx, order=self.interpolation_order, cutoff=False)
        bys = self.ebdyc.interpolate_grid_to_interface(ucy, order=self.interpolation_order, cutoff=False)
        # convert these from lists to vectors
        bvl, bxl, byl = self.ebdyc.v2l(bvs), self.ebdyc.v2l(bxs), self.ebdyc.v2l(bys)
        # compute the needed layer potentials
        sigmag_list = []
        for helper, fr, bv, bx, by in zip(self.helpers, fr_list, bvl, bxl, byl):
            sigmag_list.append(helper(fr, bv, bx, by, **kwargs))
        self.iteration_counts = [helper.iterations_last_call for helper in self.helpers]
        # we now need to evaluate this onto the grid / interface points
        sigmag = np.concatenate(sigmag_list)
        out = self.Layer_Apply(self.ebdyc.grid_source, self.ebdyc.grid_pnai, sigmag)
        # need to divide this apart
        gu, bus = self.ebdyc.divide_pnai(out)
        # we can now add gu directly to uc
        uc[self.ebdyc.phys_not_in_annulus] += gu
        # we have to send the bslps back to the indivdual ebdys to deal with them
        urs = [helper.correct(bu) for helper, bu in zip(self.helpers, bus)]
        # interpolate urs onto uc
        _ = self.ebdyc.interpolate_radial_to_grid1(urs, uc)
        uc *= self.ebdyc.phys
        if use_ef:
            ue = EmbeddedFunction(self.ebdyc)
            ue.load_data(uc, urs)
            return ue
        else:
            return uc, urs
    def _define_layer_apply(self):
        self.Layer_Apply = self.helpers[0].Layer_Apply

