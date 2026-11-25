"""Detector sanity checks before the full grid run."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins
from masks import make_gaps, apply_mask
from hrv import time_domain

FS = 500
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "eegmat")


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.npy")))
    print("== per-record QC (all 72) ==")
    print("name  n_peaks  HR  fracRR<0.30  fracRR>2.0  RMSSD_ms  SDNN_ms")
    hrs = {"1": [], "2": []}
    flagged = []
    for f in files:
        name = os.path.basename(f)[:-4]
        seg = name.split("_")[1]
        sig = np.load(f)
        pk = pan_tompkins(sig, FS)
        rr = np.diff(pk) / FS
        hr = 60.0 / np.mean(rr)
        hrs[seg].append(hr)
        f_short = np.mean(rr < 0.30)
        f_long = np.mean(rr > 2.0)
        tm = time_domain(pk, FS)
        flag = f_short > 0.05 or f_long > 0.05 or not (40 <= hr <= 160)
        if flag:
            flagged.append(name)
        print(f"{name}  {len(pk)}  {hr:.1f}  {f_short:.3f}  {f_long:.3f}  "
              f"{tm['RMSSD']:.1f}  {tm['SDNN']:.1f}  {'EXCLUDE' if flag else ''}")
    print("mean HR seg1 (rest): %.1f, seg2 (stress): %.1f" %
          (np.mean(hrs["1"]), np.mean(hrs["2"])))
    print("flagged for exclusion:", flagged)

    print("\n== neurokit2 pantompkins cross-check (3 files) ==")
    import neurokit2 as nk
    for name in ["Subject00_1", "Subject07_1", "Subject17_2"]:
        sig = np.load(os.path.join(DATA, name + ".npy")).astype(float)
        mine = pan_tompkins(sig, FS)
        _, info = nk.ecg_peaks(sig, sampling_rate=FS, method="pantompkins1985")
        theirs = np.asarray(info["ECG_R_Peaks"])
        if len(mine) and len(theirs):
            d = np.abs(mine[:, None] - theirs[None, :])
            matched = np.sum(d.min(axis=1) <= 25)
            print(f"{name}: mine={len(mine)} nk={len(theirs)} "
                  f"matched<=50ms={matched}/{len(mine)}")

    print("\n== masked-signal detection QC (Subject00_1) ==")
    sig = np.load(os.path.join(DATA, "Subject00_1.npy"))
    full = pan_tompkins(sig, FS)
    for block, rate in [(5, 0.10), (2, 0.50), (30, 0.50)]:
        rng = np.random.default_rng(123)
        gaps, actual = make_gaps(len(sig) / FS, block, rate, rng)
        masked = apply_mask(sig, gaps, FS)
        pm = pan_tompkins(masked, FS)
        in_gap = np.zeros(len(pm), bool)
        full_in_gap = np.zeros(len(full), bool)
        for s, e in gaps:
            g0, g1 = s * FS, e * FS
            in_gap |= (pm >= g0) & (pm < g1)
            full_in_gap |= (full >= g0) & (full < g1)
        outside_full = full[~full_in_gap]
        outside_mask = pm[~in_gap]
        d = np.abs(outside_full[:, None] - outside_mask[None, :])
        match = np.mean(d.min(axis=1) <= 25) if len(outside_full) else np.nan
        extras = np.sum(d.min(axis=0) > 25) if len(outside_mask) else 0
        print(f"block={block}s rate={rate}: actual={actual:.3f} "
              f"full={len(full)} masked={len(pm)} spurious_in_gaps={in_gap.sum()} "
              f"outside_match={match:.3f} extras_outside={extras}")

    print("\n== plot for visual check ==")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = np.load(os.path.join(DATA, "Subject00_1.npy")).astype(float)
    full = pan_tompkins(sig, FS)
    rng = np.random.default_rng(123)
    gaps, _ = make_gaps(len(sig) / FS, 5, 0.10, rng)
    masked = apply_mask(sig, gaps, FS)
    pm = pan_tompkins(masked, FS)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t = np.arange(len(sig)) / FS
    axes[0].plot(t, sig, lw=0.4)
    axes[0].plot(full / FS, sig[full], "r.", ms=4)
    axes[0].set_title("full + detected R peaks (first 30 s)")
    axes[1].plot(t, masked, lw=0.4)
    axes[1].plot(pm / FS, masked[pm], "r.", ms=4)
    for s, e in gaps:
        axes[1].axvspan(s, e, color="orange", alpha=0.3)
    axes[1].set_title("masked (10%, 5 s blocks, tapered edges) + detected R peaks")
    axes[1].set_xlim(0, 30)
    axes[1].set_xlabel("s")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "out", "validate_detector.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()