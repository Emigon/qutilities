""" reflection.py

author: daniel parker

models and tools for fitting the resonance parameters of a reflection-type resonator
"""

import warnings

import numpy as np
import sympy as sp

from .circle import *
from  qutilities import *

from fitkit import *
from fitkit.decimate import *

def ideal_reflection(b_Qi = (3, 4.5, 6),
                     b_Qc = (3, 4.5, 6),
                     b_fr = (1e9, 5e9, 11e9)):
    """ returns a Parametric1D model for an ideal notch resonator

    Params:
        Qi:     The log10 of the internal quality factor
        Qc:     The log10 of the modulus of the complex coupling quality factor
        fr:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Qi, Qc, Kc, Ki, fr, f = sp.symbols('Qi Qc Kc Ki fr f')

    s11 = ((Kc - Ki) + 2j*(f - fr))/((Kc + Ki) - 2j*(f - fr))
    expr = s11.subs(Kc, fr/(10**Qc)).subs(Ki, fr/(10**Qi))

    params = {'Qi': b_Qi, 'Qc': b_Qc, 'fr': b_fr}

    return Parametric1D(expr, params)

def rm_global_gain_and_phase(s11):
    """ scale and rotate s11 such that s21(f = infty) sits a 1 + 0j

    Args:
        s11:    Signal1D representation of the resonance data. Assumed to already
                be circular

    Returns:
        s11:    The repositioned input
        pm_env: The Parametric1D model for the global gain and phase components
    """
    pm_env = global_gain_and_phase()

    z = np.exp(-1j*np.pi)*z_at_f_infty(s11, circle_fit(s11)[0], clockwise=True)

    pm_env.v['G'] = 10*np.log10(np.abs(z))
    pm_env.v['theta'] = np.angle(z)*ureg('radian')

    return s11 / z, pm_env

def fit_reflection(s11):
    """ fit the resonance parameters for a notch resonator to the resonance s11

    Args:
        s11:    The complex resonance represented as a fitkit.Signal1D type

    Returns:
        model:  The Parametric1D model of the fitted resonance
    """
    pm = ideal_reflection(b_fr=(np.min(s11.x), np.mean(s11.x), np.max(s11.x)))
    circle, _ = circle_fit(s11)

    pm.v['fr'] = (s11).abs().idxmin()

    Ql = pm.v['fr']/fwhm(s11) # the fwhm estimates a good starting point
    Qc = Ql*(1 - circle.r)
    if Qc < 0:
        warnings.warn('attempted to set negative Qc')
    pm.v.set('Qi', np.log10(circle.r*Ql), clip=True)
    pm.v.set('Qc', np.log10(np.abs(Qc)), clip=True)

    # Nelder-Mead polish the parameters
    pm.fit(s11)

    return pm
