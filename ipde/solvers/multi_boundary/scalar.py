import numpy as np
from ipde.derivatives import fourier, fd_x_4, fd_y_4
from ipde.embedded_function import EmbeddedFunction, BoundaryFunction
from pybie2d.boundaries.collection import BoundaryCollection
from near_finder.nufft_interp2d import periodic_interp2d

class ScalarSolver(object):
    def __init__(self, ebdyc, solver_type, helpers, grid_backend):
        self.ebdyc = ebdyc
        self.solver_type = solver_type
        self.grid_backend = grid_backend
        if helpers is None: helpers = [None,]*self.ebdyc.N
        self.helpers = []
        for ebdy, helper in zip(self.ebdyc, helpers):
            self.helpers.append(self._get_helper(ebdy, helper))
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
        # collect grid sources
        self._collect_grid_sources()
        # get grid evaluator
        self._define_grid_evaluator()
    def _collect_grid_sources(self):
        self.grid_sources = BoundaryCollection()
        for helper in self.helpers:
            self.grid_sources.add(helper.interface_qfs_g.source,
                                            'i' if helper.interior else 'e')
        self.grid_sources.amass_information()
    def _get_helper(self, ebdy, helper):
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
    def get_boundary_values(self, u):
        bv_list = [helper.get_boundary_values(ur) for ur, helper in zip(u, self.helpers)]
        bv = BoundaryFunction(self.ebdyc)
        bv.load_data(bv_list)
        return bv
    def get_boundary_normal_derivatives(self, u):
        bv_list = [helper.get_boundary_normal_derivatives(ur) for ur, helper in zip(u, self.helpers)]
        bv = BoundaryFunction(self.ebdyc)
        bv.load_data(bv_list)
        return bv
    def evaluate_to_grid_pnai(self, sigmag):
        if self.split_grid_evaluation:
            grid_out = self.Grid_Evaluator(sigmag)
            grid_pna = grid_out[self.ebdyc.phys_not_in_annulus]
            interface_out = self.Layer_Apply(self.grid_sources, self.ebdyc.all_iv, sigmag)
            out = np.concatenate([grid_pna, interface_out])
        else:
            out = self.Grid_Evaluator(sigmag)
        return out
    def __call__(self, f, **kwargs):
        """
        f must be of type EmbeddedFunction
        """
        _, fc, fr_list = f.get_components()
        # get the grid-based solution
        uch, uc = self._grid_solve(fc)
        self.uc_save = uc.copy() # only for plotting!  can be turned off for speed
        # get interpolate method
        if self.interpolation_order == np.Inf:
            # get derivatives in Fourier space
            ucxh, ucyh = self.ikx*uch, self.iky*uch
            uch_stack = np.stack([uch, ucxh, ucyh])
            interpolater = periodic_interp2d(fh=uch_stack, eps=1e-14, upsampfac=1.25, spread_kerevalmeth=0)
            all_bvs = interpolater(self.ebdyc.interfaces_x_transf, self.ebdyc.interfaces_y_transf)
            bvs = all_bvs[0].real
            bxs = all_bvs[1].real
            bys = all_bvs[2].real
        else:
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
        # out = self.Layer_Apply(self.grid_sources, self.ebdyc.grid_pnai, sigmag)
        out = self.evaluate_to_grid_pnai(sigmag)
        # need to divide this apart
        gu, bus = self.ebdyc.divide_pnai(out)
        # we can now add gu directly to uc
        uc[self.ebdyc.phys_not_in_annulus] += gu
        # we have to send the bslps back to the indivdual ebdys to deal with them
        urs = [helper.correct(bu) for helper, bu in zip(self.helpers, bus)]
        # interpolate urs onto uc
        _ = self.ebdyc.interpolate_radial_to_grid1(urs, uc)
        uc *= self.ebdyc.phys
        ue = EmbeddedFunction(self.ebdyc)
        ue.load_data(uc, urs)
        return ue
    def _define_layer_apply(self):
        self.Layer_Apply = self.helpers[0].Layer_Apply

