from .modified_helmholtz import AnnularModifiedHelmholtzSolver

class AnnularPoissonSolver(AnnularModifiedHelmholtzSolver):
    """
    Spectrally accurate Poisson solver on annular domain

    Solves Lu = f in the annulus described by the Annular Geometry AG
    Subject to the Robin boundary condition:
    ia*u(ri) + ib*u_r(ri) = ig (boundary condition at the inner radius)
    oa*u(ro) + ob*u_r(ro) = og (boundary condition at the outer radius)

    On instantionation, a preconditioner is formed with ia, ib, ua, ub
        defining the boundary conditions
    These can be changed at solvetime, but preconditioning may not work so well
    """
    def __init__(self, AAG, ia=1.0, ib=0.0, oa=1.0, ob=0.0):
        super().__init__(AAG, 0.0, ia=ia, ib=ib, oa=oa, ob=ob)
    def solve(self, RAG, f, ig, og, ia=None, ib=None, oa=None, ob=None,
                                                    verbose=False, **kwargs):
        return super().solve(RAG, -f, ig=ig, og=og, ia=ia, ib=ib, oa=oa, ob=ob, \
                                                    verbose=verbose, **kwargs)
