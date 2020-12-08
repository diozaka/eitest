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
def obtain_samples(event_series, time_series, lag_cutoff=0, instantaneous=True, sort=True):
    '''Compute the samples T_k for all lags k as mentioned in the paper.

    Setting lag_cutoff to zero means that all lags are considered, for values larger
    than zero only lags up to and including the specified value are considered.
    If instantaneous is True, the sample at lag k=0 is included.
    If sort is True, all samples are sorted ascendingly.'''

    dt = distance_transform(event_series)
    min_lag = 0 if instantaneous else 1
    max_lag = max(dt[~np.isinf(dt)])
    if lag_cutoff > 0:
        max_lag = min(max_lag, lag_cutoff)
    max_lag = int(max_lag)

    sample = dict() # dict comprehension not supported by Numba, do it manually
    for lag in range(min_lag, max_lag+1):
        samp = time_series[dt == lag]
        if sort:
            samp = np.sort(samp)
        sample[lag] = samp
    return sample

def plot_samples(samples, ax, max_lag=-1):
    lags = np.sort([l for l in samples.keys() if (max_lag < 0) or (l <= max_lag)])
    ax.boxplot([samples[l] for l in lags], positions=lags)

@njit
def _ks_twosamp_stat(data1, data2, min_pts):
    '''Compute the test statistic value for the Kolmogorov-Smirnov two-sample test.

    Adapted from scipy.stats.ks_2samp. data1 and data2 must be sorted.'''
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
    '''Compute all pairwise Kolmogorov-Smirnov two-sample tests.'''
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

@njit
def _mmd_rbf_dot(s1, s2, deg):
    '''Compute the radial basis function inner product.

    Adapted for Python from the Matlab MMD implementation by Arthur Gretton:
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    s1, s2: inputs of shape (N, D)
    deg: kernel size'''
    G = np.square(s1).sum(axis=1)
    H = np.square(s2).sum(axis=1)
    Q = np.repeat(G, s2.shape[0]).reshape(-1, s2.shape[0])
    R = np.repeat(H, s1.shape[0]).reshape(s1.shape[0],-1).transpose()
    H = Q + R - 2*np.dot(s1, s2.transpose())
    H = np.exp(-H/2/deg**2)
    return H

@njit
def _mmd_midpoint_heuristic(s1, s2):
    '''Compute the rbf kernel size from the midpoint heuristic.

    Adapted for Python from the Matlab MMD implementation by Arthur Gretton:
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    s1, s2: inputs of shape (N, D)
    returns the heuristic value or np.nan, if the heuristic could not be computed'''
    Z = np.concatenate((s1, s2))
    N = Z.shape[0]
    if N > 100:
        Zmed = Z[:100,:]
        N = 100
    else:
        Zmed = Z
    G = np.sum(Zmed*Zmed, axis=1)
    Q = np.repeat(G, N).reshape(N,-1)
    R = np.repeat(G, N).reshape(N,-1).transpose()
    dists = Q + R - 2*np.dot(Zmed, Zmed.transpose())
    dists = (dists - np.tril(dists)).reshape(-1)
    if np.any(dists > 0):
        deg = np.sqrt(0.5*np.median(dists[dists > 0]))
    else:
        deg = np.nan
    return deg

@njit
def _mmd_twosamp_stat(s1, s2, deg):
    '''Compute the MMD two-sample test statistic and parameters for the Gamma approximation.

    Adapted for Python from the Matlab MMD implementation by Arthur Gretton:
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    s1, s2: inputs of shape (N, D)'''
    m = s1.shape[0]

    K  = _mmd_rbf_dot(s1, s1, deg)
    L  = _mmd_rbf_dot(s2, s2, deg)
    KL = _mmd_rbf_dot(s1, s2, deg)

    tstat  = 1./m**2 * np.sum(K + L - KL - KL.transpose())
    tstat = tstat * m

    meanMMD = 2./m * (1 - 1./m*np.sum(np.diag(KL)))

    K  = K  - np.diag(np.diag(K))
    L  = L  - np.diag(np.diag(L))
    KL = KL - np.diag(np.diag(KL))

    varMMD = 2./m/(m-1) * 1./m/(m-1) * np.sum(np.square(K + L - KL - KL.transpose()))

    gamma_shape = meanMMD**2 / varMMD
    gamma_scale = varMMD*m / meanMMD

    return tstat, gamma_shape, gamma_scale

@njit
def _mmd_twosamp_stat_pairwise(sample, min_pts):
    '''Numba helper to compute all pairwise MMD two-sample tests.'''
    lags = list(sorted(sample.keys()))
    tstats = np.empty(len(lags)*(len(lags)-1)//2)
    g_shps = np.empty(len(lags)*(len(lags)-1)//2)
    g_scls = np.empty(len(lags)*(len(lags)-1)//2)
    k = 0
    for i in range(len(lags)):
        data1 = sample[lags[i]].reshape(len(sample[lags[i]]), -1) # cast shape T to Tx1, if necessary
        for j in range(i+1, len(lags)):
            data2 = sample[lags[j]].reshape(len(sample[lags[j]]), -1)

            # samples have to be cropped to the same length
            m = min(len(data1), len(data2))
            if m < min_pts:
                tstats[k], g_shps[k], g_scls[k] = np.nan, np.nan, np.nan
            else:
                data1c = data1[:m,:]
                data2c = data2[:m,:]
                deg = _mmd_midpoint_heuristic(data1c, data2c)
                tstats[k], g_shps[k], g_scls[k] = _mmd_twosamp_stat(data1c, data2c, deg)
            k += 1
    return tstats, g_shps, g_scls

def pairwise_twosample_tests(sample, test, min_pts=2):
    '''For each pair of lags i<j, test whether the distributions at lags i and j are identical.

    sample: dict with numeric lag value as key and numpy.array as value, as returned
            by obtain_samples(). Keys do not have to be consecutive.
    test: either 'ks' or 'mmd'.
    min_pts: minimum number of data points required in a sample to tested (default: 2);
             test result is np.nan for skipped tests.'''

    if test == 'ks':
        ds, ens = _ks_twosamp_stat_pairwise(sample, min_pts)
        # There are several alternatives to compute the p-values of the test statistic values.
        # SciPy 0.19.x (used for the experiments in the paper):
        probs = ss.distributions.kstwobign.sf((ens + 0.12 + 0.11 / ens) * ds)
        # SciPy 1.4.x (with mode='asymp'):
        # probs = ss.distributions.kstwobign.sf(ens * ds)
        # SciPy 1.4.x (with mode='exact'):
        # TODO
        return ds, probs
    elif test == 'mmd':
        tstats, g_shps, g_scls = _mmd_twosamp_stat_pairwise(sample, min_pts)
        probs = np.array([ss.distributions.gamma.sf(tstats[k], g_shps[k], scale=g_scls[k]) for k in range(len(tstats))])
        return tstats, probs
    else:
        raise NotImplementedError('test must be either ks or mmd')

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
