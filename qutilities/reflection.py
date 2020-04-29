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

def ideal_reflection(b_Qi = (3, 4.5, 6),
                     b_Qc = (3, 4.5, 6),
                     b_theta = (-np.pi, 0, np.pi),
                     b_phi = (-np.pi, 0, np.pi),
                     b_fr = (1e9, 5e9, 11e9)):
    """ returns a Parametric1D model for an ideal notch resonator

    Params:
        Qi:     The log10 of the internal quality factor
        Qc:     The log10 of the modulus of the complex coupling quality factor
        theta:  The argument of the complex coupling quality factor
        fr:     The resonance frequency

    Args:   Parameter bounds as required by Parametric1D
    """
    Qi, Qc, theta, Kc, Ki, phi, fr, f = sp.symbols('Qi Qc theta Kc Ki phi fr f')

    s11 = ((Kc - Ki) + 2j*(f - fr))/((Kc + Ki) - 2j*(f - fr))
    expr = s11.subs(Kc, fr/((10**Qc)*sp.exp(1j*theta))).subs(Ki, fr/(10**Qi))
    expr *= sp.exp(1j*phi)

    params = {'Qi': b_Qi, 'Qc': b_Qc, 'theta': b_theta, 'phi': b_phi, 'fr': b_fr}

    return Parametric1D(expr, params, call_type=pd.Series)

def fit_reflection(s11, k=50):
    """ fit the resonance parameters for a notch resonator to the resonance s11

    Args:
        s11:    The complex resonance represented as a fitkit.Signal1D type
        k:      The number of samples taken from both ends of the spectrum, to use
                to apply a global phase offset to the data

    Returns:
        model:  The Parametric1D model of the fitted resonance
    """
    pm = ideal_reflection(b_fr=(np.min(s11.index), np.mean(s11.index), np.max(s11.index)))
    circle, _ = circle_fit(s11)

    pm.v['fr'] = (s11).abs().idxmin()
    pm.v['theta'] = np.angle(1 - circle.z) # use the circle tilt to get theta

    # when f = fr, |s11| = |(Kc - Ki)/(Kc + Ki)| = |1 - 2 Ki/(Kc + Ki)| ~ 1 - 2*r
    # ==> Ki ~ r*Kl
    # ==> Qi ~ Ql/r
    # ==> Re(Qc) ~ 1/(1/Ql - 1/Qi) = Qi/(1/r - 1)
    
    Ql = pm.v['fr']/fwhm(s11) # the fwhm estimates a good starting point
    Qi = Ql/circle.r
    Qc = Qi/(1/circle.r - 1) / np.cos(pm.v['theta'])
    if Qc < 0:
        warnings.warn('attempted to set negative Qc')
    pm.v.set('Qi', np.log10(Qi), clip=True)
    pm.v.set('Qc', np.log10(np.abs(Qc)), clip=True)

    def metric(y1, y2):
        return ((np.real(y1) - np.real(y2))**2).sum() + \
               ((np.imag(y1) - np.imag(y2))**2).sum()

    pm.freeze(['Qi', 'Qc', 'fr', 'theta'])
    pm.fit(pd.concat([s11.iloc[:k], s11.iloc[-k:]]), metric=metric)

    s11_narrow = s11.loc[pm.v['fr'] - 2*fwhm(s11):pm.v['fr'] + 2*fwhm(s11)].iloc[::2]

    pm.freeze('phi')
    pm.unfreeze(['Qi', 'Qc', 'fr', 'theta'])
    pm.fit(s11_narrow, metric=metric)
    pm.unfreeze('phi')

    return pm
