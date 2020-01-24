#!/usr/bin/python

from qutilities import *
from qutilities.notch import *
from fitkit.datasheet import *
from sampler import *

def fitter(sig1d):
    processed, model = rm_line_delay(sig1d)
    return model.v, processed

metric = percentage_error_metric_creator(fitter)

model = ideal_notch()*pm_line_delay()*global_gain_and_phase()

np.random.seed(42)
dataset, _ = snr_sweep(model, None, metric, [15, 20, 25, 30], 100, notch_sampler)

fig, axes = snr_boxplot(dataset)
fig.savefig('figures/line_delay.png', dpi = 300)
plt.show()
