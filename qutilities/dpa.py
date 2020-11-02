""" dpa.py

author: daniel parker

models and tools for fitting the resonance parameters degenerate parametric amplifier
"""

import warnings

import numpy as np
import pandas as pd
import sympy as sp

from fitkit import Parametric1D

def ideal_dpa_reflection(b_Qi=(2, 3, 6),
                         b_Qc=(2, 3, 6),
                         b_fc=(1e9, 7e9, 10e9),
                         b_fp=(10e9, 15e9, 20e9),
                         b_lamda=(1e4, 1e7, 1e10),
                         b_phi=(-np.pi, 0, np.pi),
                         b_theta=(-np.pi, 0, np.pi)):
    """ returns a Parametric1D model for an ideal DPA in reflection

    Params:
        Qi:     The log10 of the internal quality factor
        Qc:     The log10 of the modulus of the complex coupling quality factor
        fc:     The cavity frequency (in Hz)
        fp:     The pump frequency (in Hz)
        lamda:  The DPA non-linearity (in Hz)
        phi:    The argument of the coupling quality factor (in rads)
        theta:  The phase offset factor (in rads)

    Args:   Parameter bounds as required by Parametric1D
    """
    Ki, Qi, Kc, Qc, fp, fc, f, delta, lamda, phi, theta, varphi = \
        sp.symbols('Ki Qi Kc Qc fp fc f delta lamda phi theta varphi', real=True)

    s11 = ( Kc*(Kc + Ki)/2 + 1j*Kc*(delta + f) ) /\
          ( delta**2 + ((Kc + Ki)/2 + 1j*f)**2 - lamda**2 ) - 1

    expr = s11.subs(f, f - fp/2)\
              .subs(Kc, fc/(10**(Qc)*sp.exp(1j*phi)))\
              .subs(Ki, fc/(10**Qi))\
              .subs(delta, fc - fp/2)
    expr *= sp.exp(-1j*theta)

    params = {
        'Qi':     b_Qi,
        'Qc':     b_Qc,
        'fp':     b_fp,
        'fc':     b_fc,
        'lamda':  b_lamda,
        'phi':    b_phi,
        'theta':  b_theta,
    }

    return Parametric1D(expr, params, call_type=pd.Series)
