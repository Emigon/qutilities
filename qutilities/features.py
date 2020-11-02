""" features.py

author: daniel parker

tools for extracting features from resonances
"""

import numpy as np
import sympy as sp

from fitkit import *

def global_gain_and_phase(b_G = (-40, 0, 40),
                          b_theta = (-np.pi, 0, np.pi)):
    """ returns a Parametric1D model of a global gain and phase component

    Parameters:
        G:      the global gain in dB
        theta:  the global phase in degrees

    Args:   The parameter bounds as required by Parametric1D
    """
    G, theta = sp.symbols('G theta')
    cartesian = 10**(G/10) * sp.exp(1j * theta)
    return Parametric1D(cartesian, {'G': b_G, 'theta': b_theta}, call_type=pd.Series)

def z_at_f_infty(s21, circle, clockwise = False):
    """ locates the point z on the given circle corresponding to s21(f = infinity)

    Args:
        s21:    Signal1D representation of the resonance data. Data is assumed to
                be circular prior to estimation (i.e. remove line delay first)
        circle: The circle fitted to the data
    """
    theta = ((np.angle(s21.values[-1] - circle.z) - \
              np.angle(s21.values[ 0] - circle.z))/2 % np.pi)

    if clockwise:
        z = circle.z + (s21.values[-1] - circle.z)*np.exp(-1j*(theta + np.pi))
    else:
        z = circle.z + (s21.values[0] - circle.z)*np.exp(1j*theta)

    # adjust the point so that it sits on the fitted circle
    return circle.z + circle.r*(z - circle.z)/np.abs(z - circle.z)

def fwhm(sparam):
    """ estimate the full-width half-min of the resonance sparam (Signal1D) """
    mag = np.abs(sparam)
    half_max = mag.min() + .5*np.ptp(mag.values)
    fwhm = np.ptp(mag[mag <= half_max].index)
    return fwhm
