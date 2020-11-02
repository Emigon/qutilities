""" reflection.py

author: daniel parker

models and tools for fitting the resonance parameters of a reflection-type resonator
"""

import warnings

import numpy as np
import pandas as pd
import sympy as sp

from .circle import *
from  qutilities import *

from fitkit import Parametric1D

def ideal_reflection(b_Qi = (2, 4, 6),
                     b_Qc = (2, 4, 6),
                     b_theta = (-np.pi, 0, np.pi),
                     b_phi = (-np.pi, 0, np.pi),
                     b_fc = (1e9, 5e9, 11e9)):
    """ returns a Parametric1D model for an ideal notch resonator

    Params:
        Qi:     The internal quality factor
        Qc:     The modulus of the complex coupling quality factor
        phi:    The argument of the complex coupling quality factor
        theta:  A phase correction factor
        fc:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Qi, Qc, theta, Kc, Ki, phi, fc, f = sp.symbols('Qi Qc theta Kc Ki phi fc f')

    s11 = ((Kc - Ki) - 2j*(f - fc))/((Kc + Ki) + 2j*(f - fc))
    expr = s11.subs(Kc, fc/((10**Qc)*sp.exp(1j*phi))).subs(Ki, fc/(10**Qi))
    expr *= sp.exp(1j*theta)

    params = {'Qi': b_Qi, 'Qc': b_Qc, 'theta': b_theta, 'fc': b_fc, 'phi': b_phi}

    return Parametric1D(expr, params, call_type=pd.Series)

def fit_reflection(s11, k=50):
    """ fit the resonance parameters for a notch resonator to the resonance s11

    Args:
        s11:    The complex resonance represented as an pd.Series type
        k:      The number of samples taken from both ends of the spectrum, to use
                to apply a global phase offset to the data

    Returns:
        model:  The Parametric1D model of the fitted resonance
    """
    freq = s11.index
    pm = ideal_reflection(b_fc=(np.min(freq), np.mean(freq), np.max(freq)))
    circle, _ = circle_fit(s11)

    pm['fc'] = s11.apply(np.abs).idxmin()
    pm['phi'] = np.angle(1 - circle.z) # use the circle tilt to get phi

    # when f = fc, |s11| = |(Kc - Ki)/(Kc + Ki)| = |1 - 2 Ki/(Kc + Ki)| ~ 1 - 2*r
    # ==> Ki ~ r*Kl
    # ==> Qi ~ Ql/r
    # ==> Re(Qc) ~ 1/(1/Ql - 1/Qi) = Qi/(1/r - 1)

    Ql = pm['fc']/fwhm(s11) # the fwhm estimates a good starting point
    Qi = Ql/circle.r
    Qc = Qi/(1/circle.r - 1) / np.cos(pm['phi'])
    if Qc < 0:
        warnings.warn('attempted to set negative Qc')
    pm.set('Qi', np.log10(Qi), clip=True)
    pm.set('Qc', np.log10(np.abs(Qc)), clip=True)

    def metric(y1, y2):
        return (((np.real(y1.values) - np.real(y2.values))**2).sum() + \
                ((np.imag(y1.values) - np.imag(y2.values))**2).sum())

    pm.freeze(['Qi', 'Qc', 'fc', 'phi'])
    pm.fit(pd.concat([s11[:k], s11[-k:]]), metric=metric)

    s11_narrow = s11.loc[pm['fc'] - 2*fwhm(s11):pm['fc'] + 2*fwhm(s11)][::2]

    pm.freeze('theta')
    pm.unfreeze(['Qi', 'Qc', 'fc', 'phi'])
    opt_result = pm.fit(s11_narrow, metric=metric)
    pm.unfreeze('phi')

    return pm
