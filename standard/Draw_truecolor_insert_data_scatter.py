
from scipy.interpolate.interpnd import _ndim_coords_from_arrays,NDInterpolatorBase,LinearNDInterpolator,CloughTocher2DInterpolator
import numpy as np
from scipy.spatial import cKDTree
import time
class NearestNDInterpolator(NDInterpolatorBase):


    def __init__(self, x, y, rescale=False, tree_options=None):
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
                                    need_contiguous=False,
                                    need_values=False)
        print('The input points size is %s!'%(len(x)))
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)

        self.values = np.asarray(y)

    def __call__(self, *args):
        print('The output points size is %s!' % (len(args[0][0])*len(args[0][1])))
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        dist, i = self.tree.query(xi)
        return self.values[i]

def calc_time(func):

    def wrapper(*args,**kwargs):
        s_time = time.time()
        res=func(*args,**kwargs)
        print('Interpolate used time %s s'%(time.time()-s_time))
        return res
    return wrapper

@calc_time
def insert_data2d_scatter(points, values, xi, method='nearest', fill_value=np.nan,
                          rescale=False):

    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from scipy.interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        if method == 'nearest':
            fill_value = 'extrapolate'
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value,
                                  rescale=rescale)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value,
                                        rescale=rescale)
        return ip(xi)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))