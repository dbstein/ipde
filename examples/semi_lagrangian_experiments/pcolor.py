# test pcolor
%pylab
import pybie2d
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
from ipde.embedded_boundary import EmbeddedBoundary
from ipde.heavisides import SlepianMollifier

xv = np.linspace(0, 1, 4, endpoint=False)
yv = np.linspace(0, 1, 4, endpoint=False)
x, y = np.meshgrid(xv, yv, indexing='ij')
f = np.random.rand(4,4)

def periodic_pcolor(xv, yv, f, **kwargs):
	xv = np.concatenate([xv, (xv[-1]+xv[1]-xv[0],)])
	yv = np.concatenate([yv, (yv[-1]+yv[1]-yv[0],)])
	plt.pcolor(xv, yv, f, **kwargs)

periodic_pcolor(xv, yv, f)

bdy = GSB(c=star(100, x=0.0, y=0.0, a=0.1, f=3))
MOL = SlepianMollifier(10)
ebdy = EmbeddedBoundary(bdy, True, 4, bdy.max_h, 0, MOL.step)

rx = radial_rx
def fourier_cheybshev_pcolor(rv, tv, f):


