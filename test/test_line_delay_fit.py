import pytest

import numpy as np
import matplotlib.pyplot as plt

from fitkit import *
from fitkit.datasheet import *

from qutilities import *

@pytest.fixture
def resonators_w_line_delays():
    fc = 5e9
    nch = ideal_notch(b_fr = (5e9, 5e9, 5e9)) # force resonance frequency to be fc
    env = pm_line_delay()

    resonances, taus = [], []
    x = pint_safe_linspace(4.95e9, 5.05e9, 1000)
    np.random.seed(40)
    for (s21, m1), (ld, m2) in zip(sample(nch, 5, x), sample(env, 5, x)):
        # sample each resonance with centre frequency fc and span 20*fwhm
        fwhm_samples = s21.samples_below(s21.min() + np.ptp(s21.values)/2)
        fwhm = np.ptp(fwhm_samples.x)
        f = pint_safe_linspace(fc - 10*fwhm, fc + 10*fwhm, 1000)
        model = nch(f).values*env(f).values

        # inject some magnitude only noise. XXX: build this into fitkit
        mags = np.abs(model) + np.random.normal(scale = 2e-3, size = len(f))
        resonances += [Signal1D(mags*np.exp(1j*np.angle(model)), xraw = f)]
        taus += [env.v['tau']]

    return zip(taus, resonances)

@pytest.mark.plot
def test_ld_fit_plot(resonators_w_line_delays):
    for tau, res in resonators_w_line_delays:
        transformed, pm1d = rm_line_delay(res)
        assert pm1d.v['tau'] == pytest.approx(tau, rel = 5e-2)
        axes = plot_s21(transformed)
        axes = plot_s21(res, axes = axes)
        circle_fit(transformed)[0].add_to(axes[-1])
        plt.show()
