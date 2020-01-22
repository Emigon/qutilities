""" notch.py

author: daniel parker

models and tools for fitting the resonance parameters of a notch-type resonator
"""

import warnings

import numpy as np
import sympy as sp

from .circle import *
from qutilities import *

from fitkit import *
from fitkit.decimate import *

def ideal_notch(b_Qi = (3, 4.5, 6),
                b_Qc = (3, 4.5, 6),
                b_phi = (-60*ureg('degree'), 0*ureg('degree'), 60*ureg('degree')),
                b_fr = (1*ureg('GHz'), 5*ureg('GHz'), 11*ureg('GHz'))):
    """ returns a Parametric1D model for an ideal notch resonator

    Params:
        Qi:     The log10 of the internal quality factor
        Qc:     The log10 of the modulus of the complex coupling quality factor
        phi:    The argument of the complex coupling quality factor
        fr:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Ql, Qi, Qc, phi, fr, f = sp.symbols('Ql Qi Qc phi fr f')
    s21 = 1 - Ql/(10**Qc) * sp.exp(-1j*phi)/(1 + 2j*Ql*(f/fr - 1))
    expr = s21.subs(Ql, 1/(1/(10**Qi) + 1/((10**Qc)*sp.cos(phi))))

    params = {'Qi': b_Qi, 'Qc': b_Qc, 'phi': b_phi, 'fr': b_fr}

    return Parametric1D(expr, params)

def rm_global_gain_and_phase(s21):
    """ scale and rotate s21 such that s21(f = infty) sits a 1 + 0j

    Args:
        s21:    Signal1D representation of the resonance data. Assumed to already
                be circular

    Returns:
        s21:    The repositioned input
        pm_env: The Parametric1D model for the global gain and phase components
    """
    pm_env = global_gain_and_phase()

    z = z_at_f_infty(s21, circle_fit(s21)[0]) # from .features

    pm_env.v['G'] = 10*np.log10(np.abs(z))
    pm_env.v['theta'] = np.angle(z)*ureg('radian')

    return s21 / z, pm_env

def round_into_range(val, lo, hi):
    """ round val into the range (lo, hi) """
    return np.min([np.max([val, lo]), hi])

def fit_notch(s21, N = 500):
    """ fit the resonance parameters for a notch resonator to the resonance s21

    Args:
        s21:    the complex resonance represented as a fitkit.Signal1D type
        N:      the number of sample to decimate the signal to when fitting the
                loaded quality factor. see code for details

    Returns:
        notch:  The Parametric1D model of the fitted resonance
        stds:   The standard deviations of the fitted quality factors Qc and Qi
    """
    notch = ideal_notch(b_fr = (np.min(s21.x), np.mean(s21.x), np.max(s21.x)))
    circle, eps = circle_fit(s21)

    # estimate the tilt angle based on the circle centre
    notch.v['phi'] = -np.angle(1 - circle.z)*ureg('radian')

    # estimate the resonance frequency to be diametrically opposite 1 + 0j
    sfr = -(1 + 0j - circle.z) + circle.z
    notch.v['fr'] = (s21 - sfr).abs().idxmin()

    # construct a simplified model using measured parameters above
    Ql, fr, f = sp.symbols('Ql fr f')
    expr = 1 - 2*circle.r*np.exp(-1j*notch.v['phi'].magnitude) \
               / (1 + 2j*Ql*(f*ureg('Hz')/notch.v['fr'] - 1))

    Qrough = notch.v['fr']/fwhm(s21) # the fwhm estimates a good starting point
    simple_pm = Parametric1D(expr, {'Ql': (.8*Qrough, Qrough, 1.2*Qrough)})

    # convert to dimensionless as units have been cancelled in above expr. sympy
    # and pint don't play nice!
    in_hz = Signal1D(s21.values, xraw = s21.x.to('Hz').magnitude)

    # decimate the data to reduce the influence of off resonance points on the fit
    simple_pm.fit(decimate_by_derivative(in_hz, N))

    # calculate individual quality factors from the fitted loaded quality factor
    Qc = simple_pm.v['Ql']/(2*circle.r)
    notch.v['Qc'] = round_into_range(np.log10(Qc), notch.v._l['Qc'], notch.v._u['Qc'])

    Qi = 1/(1/simple_pm.v['Ql'] - 1/(Qc*np.cos(notch.v['phi'])))
    if Qi < 0:
        warnings.warn('Attempted to set negative Qi')
    notch.v['Qi'] = round_into_range(np.log10(np.abs(Qi)), notch.v._l['Qi'], notch.v._u['Qi'])

    # estimate the standard deviations associated with the estimated Q factors
    stds = {}
    stds['sigma_x'] = 2*np.std(np.abs(s21.values - circle.z))
    stds['sigma_Qi'] = _sigma_Qi(2*circle.r, simple_pm.v['Ql'], stds['sigma_x'])
    stds['sigma_Qc'] = _sigma_Qc(2*circle.r, simple_pm.v['Ql'], stds['sigma_x'])

    return notch, stds

# for uncertainty calculations (see docs/uncertainty.*)
def _sigma_Qc(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if r < 0:
        return np.nan

    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qc for r = %.2f' % r)

    return Ql * sigma_x / x**2

def _mu_Qc(x, Ql):
    return Ql/x

def _mu_Qi(x, Ql):
    return 1/(1/Ql - 1/_mu_Qc(x, Ql))

def _sigma_Qi(x, Ql, sigma_x):
    r = 1/(1/x - 1)
    if r < 0:
        return np.nan

    if np.log10(r) < -2 or np.log10(r) > 2:
        warnings.warn('Likely underestimated std on Qi for r = %.2f' % r)

    return _mu_Qi(x, Ql)**2 / _mu_Qc(x, Ql)**2 * _sigma_Qc(x, Ql, sigma_x)
