#!/usr/bin/python

from qutilities import *
from qutitlites.notch import *
from fitkit.datasheet import *
from sampler import *

def fitter(sig1d):
    fit, stds = fit_notch(sig1d)
    return fit.v, stds

metric = percentage_error_metric_creator(fitter)

notch_model = ideal_notch()

np.random.seed(42)
dataset, _ = snr_sweep(notch_model, None, metric, [15, 20, 25, 30], 100, notch_sampler)

fig, axes = snr_boxplot(dataset)
fig.savefig('figures/notch_fitting.png', dpi = 300)
plt.show()
