""" transmission.py

author: daniel parker

models and tools for fitting the resonance parameters of a transmission-type resonator
"""

import warnings

import numpy as np
import sympy as sp

from .circle import *
from qutilities import *

from fitkit import *

def ideal_transmission(b_Qi = (3, 4.5, 6),
                       b_Q1 = (3, 4.5, 6),
                       b_Q2 = (3, 4.5, 6),
                       b_fr = (1e9, 5e9, 11e9)):
    """ returns a Parametric1D model for an ideal tranmssion style resonator in s21

    Params:
        Qi:     The log10 of the internal quality factor
        Q1:     The log10 of the complex coupling quality factor from port 1
        Q2:     The log10 of the complex coupling quality factor from port 2
        fr:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Qi, Q1, Q2, Ki, K1, K2, fr, f = sp.symbols('Qi, Q1, Q2, Ki, K1, K2, fr, f')

    s21 = 2*sp.sqrt(K1*K2)/(K1 + K2 + Ki - 4j*np.pi*(f - fr))
    s11 = (K1 - K2 - Ki + 4j*np.pi*(f - fr))/(K1 + K2 + Ki - 4j*np.pi*(f - fr))
    expr_s21 = s21.subs(Ki, 2*np.pi*fr/(10**Qi)).subs(K1, 2*np.pi*fr/(10**Q1)).subs(K2, 2*np.pi*fr/(10**Q2))
    expr_s11 = s11.subs(Ki, 2*np.pi*fr/(10**Qi)).subs(K1, 2*np.pi*fr/(10**Q1)).subs(K2, 2*np.pi*fr/(10**Q2))
    expr_s22 = s11.subs(Ki, 2*np.pi*fr/(10**Qi)).subs(K1, 2*np.pi*fr/(10**Q2)).subs(K2, 2*np.pi*fr/(10**Q1))

    params = {'Qi': b_Qi, 'Q1': b_Q1, 'Q2': b_Q2, 'fr': b_fr}

    return Parametric1D(expr_s11, params),\
           Parametric1D(expr_s22, params),\
           Parametric1D(expr_s21, params)

def rm_global_gain_and_phase(s11, s22, s21, flip_phase=True):
    """ scale and rotate all sparameters such that s11(f = infty) sits a -1 + 0j

    Args:
        s21:    Signal1D representation of the resonance data. Assumed to already
                be circular

    Returns:
        s21:    The repositioned input
        pm_env: The Parametric1D model for the global gain and phase components
    """

    # NOTE: I don't understand why I need to flip the phase yet but I'm making it
    # optional until I do.

    z = z_at_f_infty(s11, circle_fit(s11)[0])

    correction = np.exp(1j*(np.pi - np.angle(z)))/np.abs(z)
    result = [correction*s11, correction*s22, correction*s21]

    if flip_phase:
        for i, r in enumerate(result):
            result[i] = Signal1D(r.real().values - 1j*r.imag().values, xraw=r.x)

    return result[0], result[1], result[2]

def fit_transmission(s11, s22, s21):
    """ fit the resonance parameters for a notch resonator to the resonance s11

    Args:
        s11:    The complex resonance represented as a fitkit.Signal1D type. All
                scattering parameters are required to fit transmission style
                resonances.
        s22:    S22 scattering parameter. Same type as s11.
        s21:    S21 scattering parameter. Same type as s21.

    Returns:
        model:  The Parametric1D model of the fitted resonance
    """
    b_fr = (np.min(s21.x), np.mean(s21.x), np.max(s21.x))
    pm_s11, pm_s22, pm_s21 = ideal_transmission(b_fr=b_fr)

    for pm in [pm_s11, pm_s22, pm_s21]:
        pm.v['fr'] = (s21).abs().idxmax()

    # fit the parameters
    opts = {'n': 3, 'iters': 50, 'sampling_method': 'sobol'}
    pm_s21.freeze('fr')
    optresult = pm_s21.fit(s21, method='shgo', opts=opts)
    pm_s21.unfreeze('fr')

    for k in pm_s21.v:
        pm_s11.v[k] = pm_s21.v[k]
        pm_s22.v[k] = pm_s21.v[k]

    return pm_s11, pm_s22, pm_s21
