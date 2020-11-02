""" processing methods to remove line delay from a resonance

author: daniel parker
"""

import numpy as np
import sympy as sp

from fitkit import *
from fitkit.decimate import *

from .circle import *
from qutilities import *

def pm_line_delay(b_tau = (0, 0, 25/3e8)):
    """ returns a Parametric1D model for the line delay

    Params:
        tau:    The line delay

    Args:   Parameter bounds as required by Parametric1D
    """
    tau, f = sp.symbols('tau f')
    return Parametric1D(sp.exp(-2j*np.pi*tau*f), {'tau': b_tau}, call_type=pd.Series)

def rm_line_delay(sparam, k=10, N=201):
    """ remove the line delay from S-parameter

    Args:
        sparam: A pandas Series representation of the S-parameter data. Using
                a span of approximately 10*fwhm of the resonance is recommended
        k:      The number of samples from the beggining of the phase response
                used to estimate the initial gradient before optimiser polishing
        N:      The number of points to decimate S-parameter by when fitting tau
                using the non-linear optimiser

    Returns:
        sparam: The input S-parameter data with the line delay removed from the
                phase response
        model:  The Parametric1D model for the line delay
    """

    # unwrap the phase and fit linear model to obtain a starting point for tau
    phi = np.unwrap(np.angle(sparam.values))

    p = np.poly1d(np.polyfit(sparam.index[:k], phi[:k], 1))
    tau_0 = p.c[0]/(2*np.pi) # we expect this to be negative

    rough = sparam*pd.Series(np.exp(-2j*np.pi*tau_0*sparam.index), index=sparam.index)

    # construct the model and a circle fitting based error function
    pm_neg = pm_line_delay(b_tau = (-np.abs(tau_0), 0, np.abs(tau_0)))

    def errf(v, self, sigma, _):
        try:
            pm_neg.v['tau'] = v[0]*pm_neg.v['tau']
        except:
            pass # return the same result as the previously set parameter!
        return circle_fit(sigma*pm_neg(sigma.index))[-1]

    # subsample to speed up circle fitting
    subsample = decimate_by_derivative(rough, N)
    shgo_opts = {'n': 100, 'iters': 1, 'sampling_method': 'sobol'}
    pm_neg.fit(subsample, method = 'shgo', errf = errf, opts = shgo_opts)

    tau_f = -(pm_neg.v['tau'] + tau_0)
    bounds = (tau_f - .5*np.abs(tau_f), tau_f, tau_f + .5*np.abs(tau_f))
    ld_model = pm_line_delay(b_tau = bounds)
    return sparam*pd.Series(np.exp(2j*np.pi*tau_f*sparam.index), index=sparam.index),\
           ld_model
