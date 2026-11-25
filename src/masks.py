"""Block-deletion mask generation (uniform random, non-overlapping, in-bounds)."""
import numpy as np


def make_gaps(duration_s, block_s, loss_rate, rng):
    """Return (gaps, actual_loss). gaps = sorted list of (start_s, end_s).

    n_blocks = round(loss_rate * T / block_s); if 0 the combination is
    infeasible (one whole block would overshoot the target) and ([], 0.0)
    is returned. Starts drawn uniformly in [0, T - block_s] with overlap
    rejection, so blocks never cross record boundaries. If placement fails
    after the attempt budget, fewer blocks are kept and the actual loss
    ratio is returned.
    """
    T = float(duration_s)
    if block_s >= T:
        return [(0.0, T)], 1.0
    n_target = int(round(loss_rate * T / block_s))
    if n_target < 1:
        # infeasible: one whole block would exceed the target loss;
        # caller skips this (record, block, rate) combination
        return [], 0.0
    gaps = []
    max_attempts = min(20000, 500 * n_target)
    attempts = 0
    while len(gaps) < n_target and attempts < max_attempts:
        attempts += 1
        s = float(rng.uniform(0.0, T - block_s))
        e = s + block_s
        if all(e <= g0 or s >= g1 for g0, g1 in gaps):
            gaps.append((s, e))
    gaps.sort()
    actual = sum(e - s for s, e in gaps) / T
    return gaps, actual


def apply_mask(sig, gaps, fs, mode="bridge", taper_s=0.040):
    """Corrupt the signal inside deleted blocks.

    mode='bridge': replace each block by the straight chord between the
      samples just outside its edges. Value-continuous at both edges (no
      step -> no bandpass ringing), slow ramp inside -> detector misses all
      beats in the block, and surviving samples are untouched.
    mode='zero': zero-fill with a raised-cosine taper of taper_s at edges.
    mode='zero_notaper': plain zero-fill (step edges, kept for comparison).
    """
    masked = np.array(sig, dtype=np.float64, copy=True)
    n = len(masked)
    ramp = int(round(taper_s * fs))
    for s, e in gaps:
        g0, g1 = int(round(s * fs)), int(round(e * fs))
        if mode == "bridge":
            lv = masked[g0 - 1] if g0 > 0 else masked[min(g1, n - 1)]
            rv = masked[g1] if g1 < n else masked[g0 - 1]
            masked[g0:g1] = np.linspace(lv, rv, g1 - g0, endpoint=False)
        elif mode == "zero":
            masked[g0:g1] = 0.0
            if ramp > 0:
                w = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1)))
                lo = max(0, g0 - ramp)
                left_len = g0 - lo
                masked[lo:g0] *= w[:left_len][::-1]
                hi = min(n, g1 + ramp)
                masked[g1:hi] *= w[: hi - g1]
        else:
            masked[g0:g1] = 0.0
    return masked