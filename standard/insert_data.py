import numpy as np
from numpy import (array, transpose, atleast_1d, atleast_2d,
                   ravel, asarray, )
from scipy.interpolate import fitpack,dfitpack


class insert_data2d(object):

    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=None):
        x = ravel(x)
        y = ravel(y)
        z = asarray(z)

        rectangular_grid = (z.size == len(x) * len(y))
        if rectangular_grid:
            if z.ndim == 2:
                if z.shape != (len(y), len(x)):
                    raise ValueError("When on a regular grid with x.size = m "
                                     "and y.size = n, if z.ndim == 2, then z "
                                     "must have shape (n, m)")
            if not np.all(x[1:] >= x[:-1]):
                j = np.argsort(x)
                x = x[j]
                z = z[:, j]
            if not np.all(y[1:] >= y[:-1]):
                j = np.argsort(y)
                y = y[j]
                z = z[j, :]
            z = ravel(z.T)
        else:
            z = ravel(z)
            if len(x) != len(y):
                raise ValueError(
                    "x and y must have equal lengths for non rectangular grid")
            if len(z) != len(x):
                raise ValueError(
                    "Invalid length for input z for non rectangular grid")

        try:
            kx = ky = {'linear': 1,
                       'cubic': 3,
                       'quintic': 5}[kind]
        except KeyError:
            raise ValueError("Unsupported interpolation type.")

        if not rectangular_grid:
            # TODO: surfit is really not meant for interpolation
            self.tck = fitpack.bisplrep(x, y, z, kx=kx, ky=ky, s=0.0)
        else:
            nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(
                x, y, z, None, None, None, None,
                kx=kx, ky=ky, s=0.0)
            self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)],
                        kx, ky)

        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.x, self.y, self.z = [array(a, copy=copy) for a in (x, y, z)]

        self.x_min, self.x_max = np.amin(x), np.amax(x)
        self.y_min, self.y_max = np.amin(y), np.amax(y)

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):

        x = atleast_1d(x)
        y = atleast_1d(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y should both be 1-D arrays")

        if not assume_sorted:
            x = np.sort(x)
            y = np.sort(y)

        if self.bounds_error or self.fill_value is not None:
            out_of_bounds_x = (x < self.x_min) | (x > self.x_max)
            out_of_bounds_y = (y < self.y_min) | (y > self.y_max)

            any_out_of_bounds_x = np.any(out_of_bounds_x)
            any_out_of_bounds_y = np.any(out_of_bounds_y)

        if self.bounds_error and (any_out_of_bounds_x or any_out_of_bounds_y):
            raise ValueError("Values out of range; x must be in %r, y in %r"
                             % ((self.x_min, self.x_max),
                                (self.y_min, self.y_max)))

        z = fitpack.bisplev(x, y, self.tck, dx, dy)
        z = atleast_2d(z)
        z = transpose(z)

        if self.fill_value is not None:
            if any_out_of_bounds_x:
                z[:, out_of_bounds_x] = self.fill_value
            if any_out_of_bounds_y:
                z[out_of_bounds_y, :] = self.fill_value

        if len(z) == 1:
            z = z[0]
        return array(z)