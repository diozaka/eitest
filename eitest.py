import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import numba

@numba.njit
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

@numba.njit
def obtain_samples(event_series, time_series, lag_cutoff=0, method='eager', instantaneous=True, sort=True):
    '''Compute the samples T_k for all lags k.

    With "eager" sampling, we sample from P(x_t | e_{t-k}=1, e_{t-k+1}=0, ..., e_t=0)
    as described in the paper. With "lazy" sampling, we sample from P(x_t | e_{t-k}=1).
    Lazy sampling results in larger samples, but leads to duplicate use of the same
    observation from the time series in multiple random samples.

    Setting lag_cutoff to zero means that all lags are considered, for values larger
    than zero only lags up to and including the specified value are considered.
    If instantaneous is True, the sample at lag k=0 is included.

    If sort is True, all samples are sorted ascendingly.'''
    sample = dict()
    series_length = len(event_series)

    if method == 'eager':
        # sample from P(x_t | e_{t-k}=1, e_{t-k+1}=0, ..., e_t=0)
        dt = distance_transform(event_series)
        for lag in range(0 if instantaneous else 1, series_length if lag_cutoff == 0 else lag_cutoff):
            idx = np.where(dt == lag)[0]
            if len(idx) < 2:
                break
            sample[lag] = time_series[idx].copy()
    elif method == 'lazy':
        # sample from P(x_t | e_{t-k}=1)
        event_idx = np.where(event_series == 1)[0]
        for lag in range(0 if instantaneous else 1, series_length if lag_cutoff == 0 else lag_cutoff):
            idx = event_idx + lag
            idx = idx[idx < series_length]
            if len(idx) < 2:
                break
            sample[lag] = time_series[idx].copy()

    if sort:
        for lag in sample:
            sample[lag].sort()

    return sample

def plot_samples(samples, ax=None, max_lag=-1):
    lags = np.sort([l for l in samples.keys() if (max_lag < 0) or (l <= max_lag)])
    if ax is None:
        ax = plt.gca()
    ax.boxplot([samples[l] for l in lags], positions=lags)

@numba.njit
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

@numba.njit
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


@numba.njit
def _mmd_rbf_dists(s1, s2):
    '''Compute the pairwise squared Euclidean distances D[i, j] = ||s1[i] - s2[j]||^2.

    s1, s2: inputs of shape (N, D)'''

    N, D = s1.shape
    s1_norm = np.sum(s1 ** 2, axis=-1)
    s2_norm = np.sum(s2 ** 2, axis=-1)
    dists = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            dists[i, j] = s1_norm[i] + s2_norm[j]
            for d in range(D):
                dists[i, j] -= 2 * s1[i, d] * s2[j, d]
    return dists


@numba.njit
def _mmd_rbf_dot(s1, s2, sigma):
    '''Compute the radial basis function inner product.

    s1, s2: inputs of shape (N, D)
    sigma: kernel bandwidth'''

    return np.exp(-1./(2. * sigma**2) * _mmd_rbf_dists(s1, s2))


@numba.njit
def _mmd_median_heuristic(s1, s2, crop=-1):
    '''Compute the rbf kernel bandwidth with the median heuristic.

    Garreau, D., Jitkrittum, W., & Kanagawa, M. (2017). Large sample analysis
    of the median heuristic. arXiv:1707.07269v3 [math.ST]

    s1, s2: inputs of shape (N, D)
    crop: if >= 0, maximum number of points to consider from each sample'''

    sample = np.concatenate((
                s1[:(None if crop < 0 else crop)],
                s2[:(None if crop < 0 else crop)]))
    dists = _mmd_rbf_dists(sample, sample)

    # extract unique dists from the lower triangular part
    dists_tril = np.tril(dists, k=-1).flatten()
    dists_tril = dists_tril[dists_tril > 0]

    sigma = np.sqrt(0.5 * np.median(dists_tril))
    return sigma


@numba.njit
def _mmd_twosamp_stat(s1, s2, sigma):
    '''Compute the biased MMD two-sample test statistic and Gamma approximation.

    The MMD test statistic is computed with rbf kernels of bandwidth sigma. The
    returned test statistic value is rescaled by the number of samples; the rescaled
    value follows a gamma distribution with the returned shape and scale parameters.

    Adapted for Python from the Matlab MMD implementation by Arthur Gretton:
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

        Gretton, A., Harchaoui, Z., Fukumizu, K., & Sriperumbudur, B. K. (2009). A Fast,
        Consistent Kernel Two-Sample Test. In: Neural Information Processing Systems.

        Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
        A kernel two-sample test. Journal of Machine Learning Research (JMLR), 13, 723–773.

    s1, s2: inputs of shape (N, D)
    sigma: bandwidth for the rbf kernel

    returns: (tstat, gamma_shape, gamma_scale)'''

    N = s1.shape[0]

    # compute Gram matrices
    gram_11 = _mmd_rbf_dot(s1, s1, sigma)
    gram_22 = _mmd_rbf_dot(s2, s2, sigma)
    gram_12 = _mmd_rbf_dot(s1, s2, sigma)

    # biased test statistic value: estimate population
    # expectations in MMD via empirical means
    tstat = 1./N**2 * (np.sum(gram_11) - 2*np.sum(gram_12) + np.sum(gram_22))

    # the Gamma approximation to the null distribution holds
    # for the rescaled test statistic value
    tstat = tstat * N

    # mean of the null distribution
    null_mean = 2./N * (1 - 1./N*np.sum(np.diag(gram_12)))

    # eliminate diagonal entries in the Gram matrices
    gram_11 -= np.diag(np.diag(gram_11))
    gram_22 -= np.diag(np.diag(gram_22))
    gram_12 -= np.diag(np.diag(gram_12))

    # variance of the null distribution
    null_var = 2./N/(N-1) * 1./N/(N-1) * np.sum(np.square(gram_11 - gram_12 - gram_12.T + gram_22))

    # obtain Gamma parameters from mean and variance
    gamma_shape = null_mean**2 / null_var
    gamma_scale = N*null_var / null_mean

    return tstat, gamma_shape, gamma_scale

@numba.njit
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
                deg = _mmd_median_heuristic(data1c, data2c, crop=50)
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
