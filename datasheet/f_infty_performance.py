#!/usr/bin/python

from qutilities import *
from qutilities.notch import *
from fitkit.datasheet import *
from sampler import *

def fitter(sig1d):
    z = z_at_f_infty(sig1d, circle_fit(sig1d)[0])
    return {'G': 10*np.log10(np.abs(z)), 'theta': np.angle(z)*ureg('radian')}, z

metric = percentage_error_metric_creator(fitter)

model = ideal_notch()*global_gain_and_phase()

np.random.seed(42)
dataset, _ = snr_sweep(model, None, metric, [15, 20, 25, 30], 100, notch_sampler)

fig, axes = snr_boxplot(dataset)
fig.savefig('figures/global_phase_and_gain.png', dpi = 300)
plt.show()
