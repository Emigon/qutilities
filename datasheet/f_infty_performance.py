#!/usr/bin/python

from qutilities import *
from fitkit.datasheet import *
from sampler import *

np.random.seed(42)

# the models
notch_model = ideal_notch()
env_model = global_gain_and_phase()
full_model = notch_model*env_model

def perc_metric(sig1d, mdata):
    z = z_at_f_infty(sig1d, circle_fit(sig1d)[0])
    in_dB = 10*np.log10(np.abs(z))
    perc_G = 100*np.abs(in_dB - mdata['G'])/mdata['G']
    perc_theta = 100*np.abs(np.rad2deg(np.angle(z)) - mdata['theta'])/mdata['theta']
    # ^ saves us rerunning the snr_sweep. we can swap the perc_theta and metric
    # rows in the returned table
    return perc_G, {'perc_theta': perc_theta, 'z': z, 'parameters': mdata}

N = 100 # the number of signals to sample per snr
snrs = [15, 20, 25, 30] # the snrs to test

dataset = []
for snr in snrs:
    sampler = notch_sampler(full_model, N, snr = snr)
    table = apply_metric_to(sampler, perc_metric)
    table['snr'] = N*[snr]
    dataset.append(table)

dataset = pd.concat(dataset, ignore_index = True)

fig, (ax1, ax2) = plt.subplots(nrows = 2, sharex = True, figsize = (8, 5))

plt.sca(ax1)
snr_boxplot(dataset)
plt.sca(ax2)
dataset['metric'] = dataset['perc_theta']
snr_boxplot(dataset)

ax1.set_ylabel('% error in $G$')
ax2.set_ylabel('% error in $\\theta$')

fig.tight_layout()
fig.savefig('figures/global_phase_and_gain.png', dpi = 300)
plt.show()
