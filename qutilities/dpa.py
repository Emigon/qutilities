""" dpa.py

author: daniel parker

models and tools for fitting the resonance parameters degenerate parametric amplifier
"""

import warnings

import numpy as np
import xarray as xr
import sympy as sp

from  qutilities import *

from fitkit import *
from fitkit.decimate import *

def ideal_dpa_reflection(b_Qi = (10, 1000, 1000000),
                         b_Qc = (10, 1000, 1000000),
                         b_fc = (1e9, 7e9, 10e9),
                         b_fp = (10e9, 15e9, 20e9),
                         b_lamda = (1e4, 1e7, 1e10),
                         b_phi = (-np.pi, 0, np.pi),
                         b_theta = (-np.pi, 0, np.pi)):
    """ returns a Parametric1D model for an ideal DPA in reflection

    Params:
        Qi:     The log10 of the internal Q
        Qc:     The log10 of the external Q
        fc:     The cavity frequency (in Hz)
        fp:     The pump frequency (in Hz)
        lamda:  The log10 of the DPA non-linearity (in Hz)
        phi:    The argument of the DPA non-linearity

    Args:   Parameter bounds as required by Parametric1D
    """
    Ki, Qi, Kc, Qc, fp, fc, f, delta, lamda, phi, theta, varphi, fudge = \
        sp.symbols('Ki Qi Kc Qc fp fc f delta lamda phi theta varphi fudge', real=True)

    # TODO : [real(Kc) + Ki] OR [Kc + Ki]? read Khalil more closely
    s11 = ( Kc*(Kc + Ki)/2 + 1j*Kc*(delta + f + lamda) ) /\
          ( delta**2 + ((Kc + Ki)/2 + 1j*f)**2 - lamda**2 ) - 1

    expr = s11.subs(f, f - fp/2)\
              .subs(Kc, fc/(Qc*sp.exp(1j*phi)))\
              .subs(Ki, fc/Qi)\
              .subs(delta, fc - fp/2)\
              .subs(lamda, lamda)
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

    return Parametric1D(expr, params, call_type=xr.DataArray)
