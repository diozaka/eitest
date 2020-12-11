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
def obtain_samples(event_series, time_series, lag_cutoff=0,
        method='eager', instantaneous=True, sort=True):
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
        for lag in range(
                0 if instantaneous else 1,
                series_length if lag_cutoff <= 0 else (lag_cutoff + 1)):
            idx = np.where(dt == lag)[0]
            if len(idx) < 2:
                break
            sample[lag] = time_series[idx].copy()
    elif method == 'lazy':
        # sample from P(x_t | e_{t-k}=1)
        event_idx = np.where(event_series == 1)[0]
        for lag in range(
                0 if instantaneous else 1,
                series_length if lag_cutoff <= 0 else (lag_cutoff + 1)):
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

@numba.njit(parallel=True)
def _ks_twosamp_stat_pairwise(sample, min_pts):
    '''Compute all pairwise Kolmogorov-Smirnov two-sample tests.'''

    lags = list(sorted(sample.keys()))
    K = len(lags)
    ds = np.empty(K*(K-1)//2)
    ens = np.empty(K*(K-1)//2)

    # compute pairwise tests in parallel
    for k in numba.prange(K*(K-1)//2):
        i = k // K
        j = k % K
        if j <= i:
            i = K - i - 2
            j = K - j - 1

        data1 = sample[lags[i]]
        data2 = sample[lags[j]]
        d, en = _ks_twosamp_stat(data1, data2, min_pts)
        ds[k]  = d
        ens[k] = en

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
def _mmd_median_heuristic(sample):
    '''Compute the rbf kernel bandwidth with the median heuristic.

    Garreau, D., Jitkrittum, W., & Kanagawa, M. (2017). Large sample analysis
    of the median heuristic. arXiv:1707.07269v3 [math.ST]

    sample: input of shape (N, D)'''

    # extract all positive distances from the lower triangular part
    dists = _mmd_rbf_dists(sample, sample)
    dists_tril = np.tril(dists, k=-1).flatten()
    dists_tril = dists_tril[dists_tril > 0]

    dists_median = 0.
    if dists_tril.size == 0:
        # no distance is > 0; use very small value to avoid bandwidth = 0
        dists_median = 1e-5
    elif dists_tril.min() == dists_tril.max():
        # all distances are identical, numba fails to compile np.median in this case
        dists_median = dists_tril[0]
    else:
        dists_median = np.median(dists_tril)

    dists_median = max(1e-5, dists_median) # avoid numerical issues
    return np.sqrt(0.5 * dists_median)


@numba.njit
def _mmd_twosamp_stat(s1, s2, med_heu_crop=0):
    '''Compute the biased MMD two-sample test statistic and Gamma approximation in O(N^2).

    The MMD test statistic is computed using rbf kernels with bandwidth selected by
    the median heuristic. The returned test statistic value is rescaled by the number
    of samples; the rescaled value follows a gamma distribution with the returned
    shape and scale parameters.

        Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. J. (2006).
        A kernel method for the two-sample-problem. In: Neural Information Processing Systems.

        Gretton, A., Harchaoui, Z., Fukumizu, K., & Sriperumbudur, B. K. (2009). A Fast,
        Consistent Kernel Two-Sample Test. In: Neural Information Processing Systems.

        Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
        A kernel two-sample test. Journal of Machine Learning Research (JMLR), 13, 723–773.

    Adapted and optimized for NumPy/Numba from the Matlab MMD implementation by Arthur Gretton:
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    s1, s2: samples of shape (N, D)
    med_heu_crop: number of data points from every sample to consider
                  in the median heuristic for the rbf kernel bandwidth (0 = all)

    returns: (tstat, gamma_shape, gamma_scale)'''

    N = s1.shape[0]

    # estimate the bandwidth for the rbf kernel from subsamples of the data
    med_heu_sample = np.concatenate((
                s1[:(N if med_heu_crop <= 0 else med_heu_crop)],
                s2[:(N if med_heu_crop <= 0 else med_heu_crop)]))
    sigma = _mmd_median_heuristic(med_heu_sample)

    # compute Gram matrices with the rbf kernel
    gram_11 = _mmd_rbf_dot(s1, s1, sigma)
    gram_22 = _mmd_rbf_dot(s2, s2, sigma)
    gram_12 = _mmd_rbf_dot(s1, s2, sigma)

    # compute the biased estimate for MMD^2, where population expectations
    # are replaced by empirical means (Gretton et al. 2006, eq. 4):
    squared_MMD = 1./N**2 * (np.sum(gram_11) - 2*np.sum(gram_12) + np.sum(gram_22))

    # the rescaled MMD^2 value approximately follows a Gamma distribution
    # under the null hypothesis (Gretton et al. 2009, eq. 8):
    tstat = N * squared_MMD

    # expected value of the biased MMD estimate
    null_mean = 2./N * (1. - 1./N*np.sum(np.diag(gram_12)))
    null_mean = max(null_mean, 1e-7) # avoid numerical issues

    # eliminate diagonal entries in the Gram matrices
    gram_11 -= np.diag(np.diag(gram_11))
    gram_22 -= np.diag(np.diag(gram_22))
    gram_12 -= np.diag(np.diag(gram_12))

    # variance of the biased MMD estimate
    null_var = 2./N/(N-1) * 1./N/(N-1) * np.sum(np.square(gram_11 - gram_12 - gram_12.T + gram_22))
    null_var = max(null_var, 1e-7) # avoid numerical issues

    # obtain Gamma parameters from mean and variance
    gamma_shape = null_mean**2 / null_var
    gamma_scale = N*null_var / null_mean

    return tstat, gamma_shape, gamma_scale

@numba.njit(parallel=True)
def _mmd_twosamp_stat_pairwise(sample, min_pts):
    '''Numba helper to compute all pairwise MMD two-sample tests.

       sample: dict with samples at all lags
       min_pts: minimum required number of points in every sample'''

    lags = list(sorted(sample.keys()))
    K = len(lags)
    tstats = np.empty(K*(K-1)//2)
    g_shps = np.empty(K*(K-1)//2)
    g_scls = np.empty(K*(K-1)//2)

    # compute pairwise tests in parallel
    for k in numba.prange(K*(K-1)//2):
        i = k // K
        j = k % K
        if j <= i:
            i = K - i - 2
            j = K - j - 1

        data1 = sample[lags[i]].reshape(len(sample[lags[i]]), -1)
        data2 = sample[lags[j]].reshape(len(sample[lags[j]]), -1)

        # crop samples to the same length
        m = min(len(data1), len(data2))
        if m < min_pts:
            tstats[k], g_shps[k], g_scls[k] = np.nan, np.nan, np.nan
        else:
            data1c = data1[:m,:]
            data2c = data2[:m,:]
            tstats[k], g_shps[k], g_scls[k] = _mmd_twosamp_stat(data1c, data2c, med_heu_crop=50)

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
