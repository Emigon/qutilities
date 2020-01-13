#!/usr/bin/python

from qutilities import *
from fitkit.datasheet import *
from sampler import *

np.random.seed(42)

# the models
notch_model = ideal_notch()
env_model = pm_line_delay()*global_gain_and_phase()
full_model = notch_model*env_model

def perc_metric(sig1d, mdata):
    processed, ld_model = rm_line_delay(sig1d)
    perc = 100*np.abs(ld_model.v['tau'] - mdata['tau'])/mdata['tau']
    table = pd.Series({'fitted': sig1d, 'ld_removed': processed, 'parameters': mdata})
    return perc, table

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
snr_boxplot(dataset, showfliers = False)

ax1.set_ylim(1e-4)
ax2.set_ylim(1e-4)

ax1.set_yscale('log')
ax2.set_yscale('log')

# ^ normalisation of eigenvector is handled by sqrt. other factors are cancelled
ax1.set_ylabel('% error in $\\tau$')
ax2.set_ylabel('% error in $\\tau$')

ax1.set_title('With outliers')
ax2.set_title('95th percentile')

fig.tight_layout()
fig.savefig('figures/line_delay.png', dpi = 300)
plt.show()
