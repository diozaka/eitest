import numpy as np
import scipy.stats as ss
from numba import njit

@njit
def distance_transform(event_series):
    '''Compute the distance to the most recent event at every time step.

    We consider all values != 0 as event occurrences.'''
    T = len(event_series)
    distance_to_event = np.empty(T)
    distance_to_event[event_series != 0] = 0
    distance_to_event[event_series == 0] = np.inf
    for i in range(1, T):
        distance_to_event[i] = min(distance_to_event[i-1]+1, distance_to_event[i])
    return distance_to_event

@njit
def obtain_samples(event_series, time_series, lag_cutoff=0):
    '''Compute the samples T_k for all lags k as mentioned in the paper.

    Setting lag_cutoff to zero means that all lags are considered, for values larger
    than zero only lags up to and including the specified value are considered.'''
    dt = distance_transform(event_series)
    min_lag = 1
    max_lag = max(dt[~np.isinf(dt)])
    if lag_cutoff > 0:
        max_lag = min(max_lag, lag_cutoff)
    max_lag = int(max_lag)

    sample = dict() # dict comprehension not supported by Numba, do it manually
    for lag in range(min_lag, max_lag+1):
        sample[lag] = np.sort(time_series[dt == lag])
    return sample

@njit
def _ks_twosamp_stat(data1, data2, min_pts):
    '''Numba helper to compute the test statistic value for the Kolmogorov-Smirnov two-sample test.

    Adapted from scipy.stats.ks_2samp.'''
    n1 = len(data1)
    n2 = len(data2)
    if (n1 < min_pts) or (n2 < min_pts):
        return np.nan, np.nan
    data_all = np.concatenate((data1, data2))
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0*n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0*n2)
    d = np.absolute(cdf1 - cdf2).max()
    en = np.sqrt(n1 * n2 / (n1 + n2))
    return d, en

@njit
def _ks_twosamp_stat_pairwise(sample, min_pts):
    '''Numba helper to compute all pairwise Kolmogorov-Smirnov two-sample tests.'''
    lags = list(sorted(sample.keys()))
    ds  = np.empty(len(lags)*(len(lags)-1)//2)
    ens = np.empty(len(lags)*(len(lags)-1)//2)
    k = 0
    for i in range(len(lags)):
        data1 = sample[lags[i]]
        for j in range(i+1, len(lags)):
            data2 = sample[lags[j]]
            d, en = _ks_twosamp_stat(data1, data2, min_pts)
            ds[k]  = d
            ens[k] = en
            k += 1
    return ds, ens

def pairwise_twosample_tests(sample, test, min_pts=2):
    '''For each pair of lags i<j, test whether the distributions at lags i and j are identical.

    sample: dict with numeric lag value as key and numpy.array as value, as returned
            by obtain_samples(). Keys do not have to be consecutive.
    test: either 'ks' or 'mmd'.
    min_pts: minimum number of data points required in a sample to tested (default: 2);
             test result is np.nan for skipped tests.'''

    if test == 'ks':
        ds, ens = _ks_twosamp_stat_pairwise(sample, min_pts)
        prob = ss.distributions.kstwobign.sf(ens * ds)
        return ds, prob
    else:
        raise NotImplementedError("only ks implemented so far")

def multitest(sorted_pvals, adjust):
    adj_pvals = np.empty_like(sorted_pvals)
    rejected = np.zeros_like(sorted_pvals, dtype=bool)
    m = len(sorted_pvals)

    if adjust == 'none':
        adj_pvals[:] = sorted_pvals
    elif adjust == 'bonferroni':
        adj_pvals[:] = sorted_pvals*m
    elif adjust == 'sidak':
        adj_pvals[:] = 1-np.power(1-sorted_pvals,m)
    elif adjust == 'holm':
        adj_pvals[:] = sorted_pvals*np.linspace(1,m,m)[::-1]
        adj_pvals[:] = np.maximum.accumulate(adj_pvals)
    elif adjust == 'hochberg':
        adj_pvals[:] = sorted_pvals*np.linspace(1,m,m)[::-1]
        adj_pvals[:] = np.minimum.accumulate(adj_pvals[::-1])[::-1]
    elif adjust == 'simes':
        adj_pvals[:] = sorted_pvals*m/np.linspace(1,m,m)
        adj_pvals[:] = np.minimum.accumulate(adj_pvals[::-1])[::-1]
    else:
        raise ValueError('no valid adjustment method specified, try: none, bonferroni, sidak, holm, hochberg, simes')

    return adj_pvals
