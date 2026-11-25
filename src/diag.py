import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins
from masks import make_gaps

FS = 500
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "eegmat")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def load(name):
    return np.load(os.path.join(DATA, name + ".npy")).astype(float)


def main():
    import neurokit2 as nk

    # --- Subject07_1: where do mine and nk disagree?
    sig = load("Subject07_1")
    mine = pan_tompkins(sig, FS)
    _, info = nk.ecg_peaks(sig, sampling_rate=FS, method="pantompkins1985")
    theirs = np.asarray(info["ECG_R_Peaks"])
    d = np.abs(mine[:, None] - theirs[None, :])
    unmatched = mine[d.min(axis=1) > 25]
    print("Subject07_1 unmatched mine peaks (s):", np.round(unmatched / FS, 2)[:20])
    print("their RR range:", np.round(np.min(np.diff(theirs)) / FS, 2),
          np.round(np.max(np.diff(theirs)) / FS, 2))
    print("my RR range:", np.round(np.min(np.diff(mine)) / FS, 2),
          np.round(np.max(np.diff(mine)) / FS, 2))
    # plot around first unmatched region
    c = unmatched[0] if len(unmatched) else int(20 * FS)
    lo, hi = max(0, c - 3 * FS), min(len(sig), c + 3 * FS)
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(lo, hi) / FS
    ax.plot(t, sig[lo:hi], lw=0.5)
    m = mine[(mine >= lo) & (mine < hi)]
    th = theirs[(theirs >= lo) & (theirs < hi)]
    ax.plot(m / FS, sig[m], "r.", ms=8, label="mine")
    ax.plot(th / FS, sig[th], "gx", ms=8, label="neurokit2")
    ax.legend()
    ax.set_title("Subject07_1 disagreement region")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diag_s07.png"), dpi=110)

    # --- Subject13_1: over-detection?
    sig = load("Subject13_1")
    mine = pan_tompkins(sig, FS)
    _, info = nk.ecg_peaks(sig, sampling_rate=FS, method="pantompkins1985")
    theirs = np.asarray(info["ECG_R_Peaks"])
    print("\nSubject13_1: mine=%d nk=%d" % (len(mine), len(theirs)))
    rr = np.diff(mine) / FS
    print("my RR percentiles 1/5/50:", np.round(np.percentile(rr, [1, 5, 50]), 3))
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for ax, (lo, hi) in zip(axes, [(0, 10 * FS), (60 * FS, 70 * FS)]):
        t = np.arange(lo, hi) / FS
        ax.plot(t, sig[lo:hi], lw=0.5)
        m = mine[(mine >= lo) & (mine < hi)]
        th = theirs[(theirs >= lo) & (theirs < hi)]
        ax.plot(m / FS, sig[m], "r.", ms=6, label="mine")
        ax.plot(th / FS, sig[th], "gx", ms=6, label="nk")
        ax.legend()
    axes[0].set_title("Subject13_1 first 10 s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diag_s13.png"), dpi=110)

    # --- 50%/2s extras: distance to nearest gap edge
    sig = load("Subject00_1")
    full = pan_tompkins(sig, FS)
    rng = np.random.default_rng(123)
    gaps, _ = make_gaps(len(sig) / FS, 2, 0.50, rng)
    masked = sig.copy()
    for s, e in gaps:
        masked[int(round(s * FS)):int(round(e * FS))] = 0
    pm = pan_tompkins(masked, FS)
    edges = np.array([x for s, e in gaps for x in (s * FS, e * FS)])
    in_gap = np.zeros(len(pm), bool)
    for s, e in gaps:
        in_gap |= (pm >= s * FS) & (pm < e * FS)
    d_full = np.abs(pm[:, None] - full[None, :]).min(axis=1)
    extras = pm[(d_full > 25) & ~in_gap]
    de = np.abs(extras[:, None] - edges[None, :]).min(axis=1) / FS
    print("\n50%/2s extras distance to nearest gap edge (s):", np.round(np.sort(de), 3))
    spur = pm[in_gap]
    ds = np.abs(spur[:, None] - edges[None, :]).min(axis=1) / FS
    print("spurious in-gap distance to edge (s):", np.round(ds, 3))


if __name__ == "__main__":
    main()