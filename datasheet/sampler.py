""" sampler.py

author: daniel parker
"""

import numpy as np
from copy import deepcopy
from tqdm import tqdm

from fitkit.datasheet import *

def notch_sampler(pm1d, N, x, **callkwargs):
    """ sample notch model for N uniformly sampled points in the parameter space

    Args:
        pm1d:       The Parametric1D model of the notch resonance (may include
                    environment terms). This function assumes that pm1d has
                    parameters 'fr', 'Qi', 'Qc' and 'phi'
        N:          The number of samples to draw
        x:          Dummy argument to maintain the same function signature as
                    fitkit.datasheet.sample. Set to None. This is ignored.
        callkwargs: Keyword arguments to pass to Parametric1D.__call__

    Yields:
        Signal1D:   The samples of the sampled resonance with centre 'fr' and span
                    10*fwhm
        mdata:      A copy of the parameters that were sampled
    """
    v_init = deepcopy(pm1d.v)
    for _ in tqdm(range(N)):
        for p in pm1d.v:
            pm1d.v[p] = pint_safe_uniform(pm1d.v._l[p], pm1d.v._u[p])

        fr = pm1d.v['fr']
        phi = np.cos(pm1d.v['phi'])
        Ql = 1/(1/(10**pm1d.v['Qi']) + 1/((10**pm1d.v['Qc'])*np.cos(phi)))
        span = 20 * (fr/Ql) # the estimated fwhm

        samples = pm1d(np.linspace(fr - span/2, fr + span/2, 801), **callkwargs)
        yield samples, deepcopy(pm1d.v)
