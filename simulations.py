import sys

import numpy as np
import numba

import eitest

if len(sys.argv) != 5:
    print(f'USAGE: {sys.argv[0]} eager|lazy instant|causal tt|ks|mmd none|bonferroni|sidak|holm|hochberg|simes')
    sys.exit(1)

# command line parameters
sample_method = sys.argv[1]
instantaneous = (sys.argv[2] == 'instant')
twosamp_test = sys.argv[3]
multi_test = sys.argv[4]

# other global parameters
default_T = 8192
n_pairs = 100
alpha = 0.05
lag_cutoff = 32

@numba.njit
def event_series_bernoulli(series_length, event_count):
    '''Generate an iid Bernoulli distributed event series.

    series_length: length of the event series
    event_count: number of events'''

    event_series = np.zeros(series_length)
    event_series[np.random.choice(np.arange(0, series_length), event_count, replace=False)] = 1
    return event_series

@numba.njit
def time_series_mean_impact(event_series, order, signal_to_noise):
    '''Generate a time series with impacts in mean as described in the paper.

    The impact weights are sampled iid from N(0, signal_to_noise),
    and additional noise is sampled iid from N(0,1). The detection problem will
    be harder than in time_series_meanconst_impact for small orders, as for small
    orders we have a low probability to sample at least one impact weight with a
    high magnitude. On the other hand, since the impact is different at every lag,
    we can detect the impacts even if the order is larger than the max_lag value
    used in the test.

    event_series: input of shape (T,) with event occurrences
    order: order of the event impacts
    signal_to_noise: signal to noise ratio of the event impacts'''

    series_length = len(event_series)
    weights = np.random.randn(order)*np.sqrt(signal_to_noise)
    time_series = np.random.randn(series_length)
    for t in range(series_length):
        if event_series[t] == 1:
            time_series[t+1:t+order+1] += weights[:order-max(0, (t+order+1)-series_length)]
    return time_series

@numba.njit
def time_series_meanconst_impact(event_series, order, const):
    '''Generate a time series with impacts in mean by adding a constant.
    Better for comparing performance across different impact orders, since the
    magnitude of the impact will always be the same.

    event_series: input of shape (T,) with event occurrences
    order: order of the event impacts
    const: constant for mean shift'''

    series_length = len(event_series)
    time_series = np.random.randn(series_length)
    for t in range(series_length):
        if event_series[t] == 1:
            time_series[t+1:t+order+1] += const
    return time_series

@numba.njit
def time_series_var_impact(event_series, order, variance):
    '''Generate a time series with impacts in variance as described in the paper.

    event_series: input of shape (T,) with event occurrences
    order: order of the event impacts
    variance: variance under event impacts'''

    series_length = len(event_series)
    time_series = np.random.randn(series_length)
    for t in range(series_length):
        if event_series[t] == 1:
            for tt in range(t+1, min(series_length, t+order+1)):
                time_series[tt] = np.random.randn()*np.sqrt(variance)
    return time_series

@numba.njit
def time_series_tail_impact(event_series, order, dof):
    '''Generate a time series with impacts in tails as described in the paper.

    event_series: input of shape (T,) with event occurrences
    order: delay of the event impacts
    dof: degrees of freedom of the t distribution'''

    series_length = len(event_series)
    time_series = np.random.randn(series_length)*np.sqrt(dof/(dof-2))
    for t in range(series_length):
        if event_series[t] == 1:
            for tt in range(t+1, min(series_length, t+order+1)):
                time_series[tt] = np.random.standard_t(dof)
    return time_series

def test_simul_pairs(impact_model, param_T, param_N, param_q, param_r,
               n_pairs, lag_cutoff, instantaneous, sample_method,
               twosamp_test, multi_test, alpha):
    true_positive = 0.
    false_positive = 0.
    for _ in range(n_pairs):
        es = event_series_bernoulli(param_T, param_N)
        if impact_model == 'mean':
            ts = time_series_mean_impact(es, param_q, param_r)
        elif impact_model == 'meanconst':
            ts = time_series_meanconst_impact(es, param_q, param_r)
        elif impact_model == 'var':
            ts = time_series_var_impact(es, param_q, param_r)
        elif impact_model == 'tail':
            ts = time_series_tail_impact(es, param_q, param_r)
        else:
            raise ValueError('impact_model must be "mean", "meanconst", "var" or "tail"')

        # coupled pair
        samples = eitest.obtain_samples(es, ts, lag_cutoff=lag_cutoff,
                                        method=sample_method,
                                        instantaneous=instantaneous,
                                        sort=(twosamp_test == 'ks')) # samples need to be sorted for K-S test
        tstats, pvals = eitest.pairwise_twosample_tests(samples, twosamp_test, min_pts=2)
        pvals_adj = eitest.multitest(np.sort(pvals[~np.isnan(pvals)]), multi_test)
        true_positive += (pvals_adj.min() < alpha)

        # uncoupled pair
        samples = eitest.obtain_samples(np.random.permutation(es), ts, lag_cutoff=lag_cutoff,
                                        method=sample_method,
                                        instantaneous=instantaneous,
                                        sort=(twosamp_test == 'ks'))
        tstats, pvals = eitest.pairwise_twosample_tests(samples, twosamp_test, min_pts=2)
        pvals_adj = eitest.multitest(np.sort(pvals[~np.isnan(pvals)]), multi_test)
        false_positive += (pvals_adj.min() < alpha)

    return true_positive/n_pairs, false_positive/n_pairs


# ## Mean impact model

default_N = 64
default_r = 1.
default_q = 4

# ### ... by number of events

vals = [4, 8, 16, 32, 64, 128, 256]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='mean', param_T=default_T,
                                        param_N=val, param_q=default_q, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# mean impact model (T={default_T}, q={default_q}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# N\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by impact order

vals = [1, 2, 4, 8, 16, 32]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='mean', param_T=default_T,
                                        param_N=default_N, param_q=val, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# mean impact model (T={default_T}, N={default_N}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# q\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by signal-to-noise ratio

vals = [1./32, 1./16, 1./8, 1./4, 1./2, 1., 2., 4.]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='mean', param_T=default_T,
                                        param_N=default_N, param_q=default_q, param_r=val,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# mean impact model (T={default_T}, N={default_N}, q={default_q}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# r\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ## Meanconst impact model

default_N = 64
default_r = 0.5
default_q = 4

# ### ... by number of events

vals = [4, 8, 16, 32, 64, 128, 256]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='meanconst', param_T=default_T,
                                        param_N=val, param_q=default_q, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# meanconst impact model (T={default_T}, q={default_q}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# N\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by impact order

vals = [1, 2, 4, 8, 16, 32]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='meanconst', param_T=default_T,
                                        param_N=default_N, param_q=val, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# meanconst impact model (T={default_T}, N={default_N}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# q\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by mean value

vals = [0.125, 0.25, 0.5, 1, 2]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='meanconst', param_T=default_T,
                                        param_N=default_N, param_q=default_q, param_r=val,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# meanconst impact model (T={default_T}, N={default_N}, q={default_q}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# r\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ## Variance impact model
# In the paper, we show results with the variance impact model parametrized by the **variance increase**. Here we directly modulate the variance.

default_N = 64
default_r = 8.
default_q = 4

# ### ... by number of events

vals = [4, 8, 16, 32, 64, 128, 256]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='var', param_T=default_T,
                                        param_N=val, param_q=default_q, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# var impact model (T={default_T}, q={default_q}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# N\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by impact order

vals = [1, 2, 4, 8, 16, 32]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='var', param_T=default_T,
                                        param_N=default_N, param_q=val, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# var impact model (T={default_T}, N={default_N}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# q\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by variance

vals = [2., 4., 8., 16., 32.]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='var', param_T=default_T,
                                        param_N=default_N, param_q=default_q, param_r=val,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# var impact model (T={default_T}, N={default_N}, q={default_q}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# r\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ## Tail impact model

default_N = 512
default_r = 3.
default_q = 4

# ### ... by number of events

vals = [64, 128, 256, 512, 1024]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='tail', param_T=default_T,
                                        param_N=val, param_q=default_q, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# tail impact model (T={default_T}, q={default_q}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# N\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by impact order

vals = [1, 2, 4, 8, 16, 32]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='tail', param_T=default_T,
                                        param_N=default_N, param_q=val, param_r=default_r,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# tail impact model (T={default_T}, N={default_N}, r={default_r}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# q\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()

# ### ... by degrees of freedom

vals = [2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]

tprs = np.empty(len(vals))
fprs = np.empty(len(vals))
for i, val in enumerate(vals):
    tprs[i], fprs[i] = test_simul_pairs(impact_model='tail', param_T=default_T,
                                        param_N=default_N, param_q=default_q, param_r=val,
                                        n_pairs=n_pairs, sample_method=sample_method,
                                        lag_cutoff=lag_cutoff, instantaneous=instantaneous,
                                        twosamp_test=twosamp_test, multi_test=multi_test, alpha=alpha)

print(f'# tail impact model (T={default_T}, N={default_N}, q={default_q}, n_pairs={n_pairs}, cutoff={lag_cutoff}, instantaneous={instantaneous}, alpha={alpha}, {sample_method}-{twosamp_test}-{multi_test})')
print(f'# r\ttpr\tfpr')
for i, (tpr, fpr) in enumerate(zip(tprs, fprs)):
    print(f'{vals[i]}\t{tpr}\t{fpr}')
print()


