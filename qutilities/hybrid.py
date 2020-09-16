""" hybrid.py

author: daniel parker

models and tools for fitting hybridised resonances
"""

import warnings

import numpy as np
import xarray as xr
import sympy as sp

from  qutilities import *

from fitkit import *

def hybrid_reflection(b_fa = (1e9, 8e9, 40e9),
                      b_fd = (-40e9, 15e9, 40e9),
                      b_df = (-100e6, 0, 100e6),
                      b_Ki = (2, 3, 8),
                      b_Kc = (2, 3, 8),
                      b_theta = (-np.pi, 0, np.pi),
                      b_gamma = (2, 3, 8),
                      b_g = (2, 3, 8)):
    """ reflection model for a hybridized optomechanical resonance """
    params = {
        'Kc':    b_Kc,     # the log10 of the modulus for the complex coupling rate
        'theta': b_theta,  # the argument for the complex coupling rate
        'Ki':    b_Ki,     # the log10 of the internal loss rate of the cavity
        'gamma': b_gamma,  # the log10 of the total loss rate of coupled mode
        'g':     b_g,      # the log10 of the coupling rate between the cavity and coupled mode
        'f_a':   b_fa,     # the frequency of the cavity (in Hz)
        'df':    b_df,     # the frequency offset of the coupled mode (in Hz)
    }

    K, Kc, theta, Ki, gamma, g = sp.symbols('K, Kc theta Ki gamma g')
    f, f_d, f_b, f_a, df = sp.symbols('f f_d f_b f_a df')

    s11 = -1 - 1j*Kc*sp.exp(-1j*theta)*(1j*gamma/2 + 2*np.pi*(f - f_d - f_b))/ \
          (10**(2*g) - (1j*K/2 + 2*np.pi*(f - f_a))*(1j*gamma/2 + 2*np.pi*(f - f_d - f_b)))
    expr = s11.subs(K, Ki + Kc*sp.exp(-1j*theta)).subs(f_b, f_a - f_d + df)

    # convert rates into log units
    expr = expr.subs(Kc, 10**Kc).subs(Ki, 10**Ki).subs(gamma, 10**gamma)

    return Parametric1D(expr, params, call_type=xr.DataArray)
