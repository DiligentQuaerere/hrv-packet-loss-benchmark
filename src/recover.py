"""Gap-recovery methods. Unified signature: recover(peaks, gaps, fs, rng=None).

peaks: detected R-peak sample indices on the masked signal (sorted int array).
gaps:  list of (start_s, end_s) deleted blocks.
Returns recovered peak sample indices (sorted int array).

Placement (all methods): n = floor(gap_len / RR_est) peaks bridging the last
surviving peak before the gap (a) to the first surviving peak after it (b),
i.e. n+1 equal-spans for M1/M2. Bridging avoids phase-mismatch jumps at the
gap boundaries, which would otherwise inject broadband power into HF.
M1/M2: constant intervals (global / local RR estimate).
M3: n+1 draws ~ U(mean-SD, mean+SD) of intact RRs, proportionally rescaled to
sum exactly to the bridge span (b-a), placed cumulatively from a.

RR statistics use "intact" intervals only: consecutive detected-peak pairs
whose interval does not cross any gap.
"""
import numpy as np


def _gap_samples(gaps, fs):
    return [(int(round(s * fs)), int(round(e * fs))) for s, e in gaps]


def _intact_rr(peaks, gs, fs):
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks) / fs
    ok = np.ones(len(rr), dtype=bool)
    for g0, g1 in gs:
        ok &= ~((peaks[:-1] < g1) & (peaks[1:] > g0))
    return rr[ok]


def _bridge(peaks, g0, g1):
    before = peaks[peaks < g0]
    after = peaks[peaks >= g1]
    a = int(before[-1]) if len(before) else g0
    b = int(after[0]) if len(after) else g1
    return a, b, before, after


def recover_none(peaks, gaps, fs, rng=None):
    """Baseline: no recovery; each gap stays one long RR interval."""
    return np.asarray(peaks, dtype=np.int64)


def recover_mean_global(peaks, gaps, fs, rng=None):
    """Method 1: n = floor(gap_len / mean_intact_RR), constant bridging RR."""
    peaks = np.asarray(peaks, dtype=np.int64)
    gs = _gap_samples(gaps, fs)
    rr = _intact_rr(peaks, gs, fs)
    mean_rr = float(np.mean(rr)) if len(rr) else 0.8
    out = list(peaks)
    for g0, g1 in gs:
        a, b, _, _ = _bridge(peaks, g0, g1)
        n = int(np.floor(((g1 - g0) / fs) / mean_rr))
        for k in range(1, n + 1):
            out.append(int(round(a + (b - a) * k / (n + 1))))
    return np.array(sorted(set(out)), dtype=np.int64)


def recover_mean_local(peaks, gaps, fs, rng=None):
    """Method 2: same placement, RR estimate = mean of the intact RR
    immediately left and immediately right of the gap (fallback: global)."""
    peaks = np.asarray(peaks, dtype=np.int64)
    gs = _gap_samples(gaps, fs)
    rr = _intact_rr(peaks, gs, fs)
    fallback = float(np.mean(rr)) if len(rr) else 0.8
    out = list(peaks)
    for g0, g1 in gs:
        a, b, before, after = _bridge(peaks, g0, g1)
        vals = []
        if len(before) >= 2:
            vals.append((before[-1] - before[-2]) / fs)
        if len(after) >= 2:
            vals.append((after[1] - after[0]) / fs)
        local_rr = float(np.mean(vals)) if vals else fallback
        n = int(np.floor(((g1 - g0) / fs) / local_rr))
        for k in range(1, n + 1):
            out.append(int(round(a + (b - a) * k / (n + 1))))
    return np.array(sorted(set(out)), dtype=np.int64)


def recover_random_sd(peaks, gaps, fs, rng=None):
    """Method 3: n = floor(gap_len / mean_intact_RR); n+1 RR draws from
    U(mean-SD, mean+SD), rescaled to bridge span, cumulative from a."""
    if rng is None:
        rng = np.random.default_rng(0)
    peaks = np.asarray(peaks, dtype=np.int64)
    gs = _gap_samples(gaps, fs)
    rr = _intact_rr(peaks, gs, fs)
    m = float(np.mean(rr)) if len(rr) else 0.8
    s = float(np.std(rr, ddof=1)) if len(rr) >= 2 else 0.05
    lo, hi = max(0.2, m - s), m + s
    out = list(peaks)
    for g0, g1 in gs:
        a, b, _, _ = _bridge(peaks, g0, g1)
        n = int(np.floor(((g1 - g0) / fs) / m))
        if n < 1 or b <= a:
            continue
        draws = rng.uniform(lo, hi, n + 1)
        draws *= ((b - a) / fs) / draws.sum()
        t = float(a)
        for d in draws[:-1]:
            t += d * fs
            out.append(int(round(t)))
    return np.array(sorted(set(out)), dtype=np.int64)


METHODS = {
    "none": recover_none,
    "mean_global": recover_mean_global,
    "mean_local": recover_mean_local,
    "random_sd": recover_random_sd,
}