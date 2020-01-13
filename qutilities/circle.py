""" circle.py

author: daniel parker

this file defines the Circle datatype and complex plane circle fitting methods
"""

import warnings

import numpy as np
import pandas as pd

from scipy.linalg import eig

import matplotlib.patches as patches

from fitkit import *

class Circle(object):
    def __init__(self, z, r):
        self.z = z
        self.r = r

    @property
    def x(self):
        return np.real(self.z)

    @property
    def y(self):
        return np.imag(self.z)

    def rotate(self, theta):
        self.z *= np.exp(1j*theta)
        return self

    def scale(self, scaling_factor):
        self.z *= scaling_factor
        self.r *= np.abs(scaling_factor)
        return self

    def add_to(self, axes):
        axes.add_patch(patches.Circle((self.x, self.y), radius = self.r, fill = False))
        axes.scatter(self.x, self.y, marker = '.', color = 'k')
        axes.relim()
        axes.autoscale_view()

def circle_fit(s21_complex, attempts = 5):
    """ fit a circle to a resonance on the complex plane

    Args:
        s21_complex:    A Signal1D formatted resonance, with phase and magnitude
                        components represented by complex numbers
        attempts:       The number of times to attempt to fit the data before
                        raising an exception. Sometimes the solution is a circle
                        with a very big radius that intersects the data at a
                        single point. This is a mathematically valid solution but
                        not a useful one. A single point is removed at random from
                        the data until a fit with a radius < the estimated diameter
                        from the range of real values is achieved. Default value
                        is 5. To attempt once set attempts to 1. Do not set to 0.

    Returns:
        circle:         A Circle with fitted radius and centre (represented by
                        a complex attribute z)
        error:          The sum of the squares error in the magnitude deviation
                        from the fitted circle. This is intended for use in
                        optimisations that process the resonance to make it more
                        circular
    """
    x, y = np.real(s21_complex.values), np.imag(s21_complex.values)
    xp = x - np.mean(x)
    yp = y - np.mean(y)

    z = xp**2 + yp**2

    # increases the abs values of the complex data so that the moments don't
    # look small compared to n = x.size. also has the benefit of simplifying
    # the matricies
    scale = z.sum()/x.size

    xp /= np.sqrt(scale)
    yp /= np.sqrt(scale)
    zp = xp**2 + yp**2

    M = np.array(
        [
            [zp@zp,  xp@zp, yp@zp, x.size],
            [zp@xp,  xp@xp, yp@xp, 0     ],
            [zp@yp,  xp@yp, yp@yp, 0     ],
            [x.size, 0    , 0    , x.size]
        ])

    P = x.size*np.array(
        [
            [4, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])

    # find eigenvector associated with smallest non-negative eigenvalue
    vals, vects = eig(M, b = P)
    idxs, = np.where(vals > 0)
    A, B, C, D = vects[:,idxs][:,vals[idxs].argmin()]

    xc, yc = -np.sqrt(scale) * B/(2*A), -np.sqrt(scale) * C/(2*A)
    xc += np.mean(x)
    yc += np.mean(y) # undo the initial transformations

    r = np.sqrt(scale) * np.sqrt(B**2 + C**2 - 4*A*D)/(2*np.abs(A))
    err = np.sum(np.abs(np.abs(s21_complex.values - (xc + 1j*yc)) - r))

    # randomly remove a sample if the radius of the fitted circle makes no sense
    # and try to fit the data again
    if r > 100*np.ptp(x):
        if attempts > 0:
            k = np.random.choice(len(s21_complex))
            x, y = s21_complex.x, s21_complex.values
            x = np.append(x[:k], x[k+1:])
            y = np.append(y[:k], y[k+1:])
            return circle_fit(Signal1D(y, xraw = x), attempts = attempts - 1)
        warnings.warn("Failed to fit non-big circle to data")

    return Circle(xc + 1j*yc, r), err
