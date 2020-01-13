#!/usr/bin/python

from qutilities import *
from fitkit.datasheet import *
from sampler import *

np.random.seed(42)

notch_model = ideal_notch()

def perc_metric(sig1d, mdata):
    fit, fit_mdata = fit_notch(sig1d)

    for p in fit.v:
        fit_mdata[f"perc_{p}"] = 100*np.abs((fit.v[p] - mdata[p])/mdata[p])
        if hasattr(fit.v[p], 'units'):
            fit_mdata[f"perc_{p}"] = fit_mdata[f"perc_{p}"].to_reduced_units().magnitude
    fit_mdata['parameters'] = mdata
    fit_mdata['test_case'] = sig1d

    return fit_mdata['perc_fr'], fit_mdata

N = 100 # the number of signals to sample per snr
snrs = [15, 20, 25, 30] # the snrs to test

dataset = []
for snr in snrs:
    sampler = notch_sampler(notch_model, N, snr = snr)
    table = apply_metric_to(sampler, perc_metric)
    table['snr'] = N*[snr]
    dataset.append(table)

dataset = pd.concat(dataset, ignore_index = True)

fig, axes = plt.subplots(nrows = len(notch_model.v), sharex = True, figsize = (8, 10))

for p, ax in zip(notch_model.v, axes):
    dataset['metric'] = dataset[f'perc_{p}']
    plt.sca(ax)
    snr_boxplot(dataset)

    ax.set_ylabel(f'% error in ${p}$')

fig.tight_layout()
fig.savefig('figures/notch_fitting.png', dpi = 300)
plt.show()
