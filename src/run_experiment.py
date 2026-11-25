"""Main grid: block deletion -> re-detect -> recover -> HRV -> compare to truth.

Outputs out/results_long.csv, out/truth_metrics.csv, out/detection_qc.csv.
Run:  python src/run_experiment.py
"""
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins
from masks import make_gaps, apply_mask
from recover import METHODS
from hrv import all_metrics, ALL_METRICS

FS = 500
BLOCKS = [2, 5, 10, 30]
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
REPS = 3
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "eegmat")
OUT = os.path.join(ROOT, "out")


def derive_seed(subject, block, rate, rep, stream="mask"):
    key = f"eegmat-loss-v1|{subject}|{block}|{rate}|{rep}|{stream}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def qc_flag(peaks):
    rr = np.diff(peaks) / FS
    if len(rr) < 2:
        return True, {}
    hr = 60.0 / np.mean(rr)
    stats = {
        "n_peaks": len(peaks),
        "mean_hr": hr,
        "frac_rr_lt_0p30": float(np.mean(rr < 0.30)),
        "frac_rr_gt_2p0": float(np.mean(rr > 2.0)),
    }
    flag = (stats["frac_rr_lt_0p30"] > 0.05 or stats["frac_rr_gt_2p0"] > 0.05
            or not (40.0 <= hr <= 160.0))
    return flag, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "*.npy")))
    t0 = time.time()

    truth = {}
    truth_rows = []
    kept = []
    for f in files:
        name = os.path.basename(f)[:-4]
        subject, seg = name.split("_")
        sig = np.load(f)
        pk = pan_tompkins(sig, FS)
        flag, stats = qc_flag(pk)
        dur = len(sig) / FS
        m = all_metrics(pk, FS, dur)
        truth[name] = m
        truth_rows.append({"file": name, "subject": subject, "segment": int(seg),
                           "duration_s": dur, "excluded": flag, **stats, **m})
        if not flag:
            kept.append((f, name, subject, int(seg), sig, dur))
    pd.DataFrame(truth_rows).to_csv(os.path.join(OUT, "truth_metrics.csv"), index=False)
    print(f"truth done: {len(kept)}/{len(files)} records kept "
          f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    qc_rows = []
    for fi, (f, name, subject, seg, sig, dur) in enumerate(kept):
        full = pan_tompkins(sig, FS)
        for block in BLOCKS:
            for rate in RATES:
                for rep in range(REPS):
                    seed = derive_seed(subject, block, rate, rep)
                    gaps, actual = make_gaps(dur, block, rate,
                                             np.random.default_rng(seed))
                    if not gaps:
                        continue  # infeasible combo (block longer than target loss)
                    masked = apply_mask(sig, gaps, FS, mode="bridge")
                    pm = pan_tompkins(masked, FS)

                    in_gap = np.zeros(len(pm), bool)
                    full_in_gap = np.zeros(len(full), bool)
                    for s, e in gaps:
                        in_gap |= (pm >= s * FS) & (pm < e * FS)
                        full_in_gap |= (full >= s * FS) & (full < e * FS)
                    of, om = full[~full_in_gap], pm[~in_gap]
                    if len(of) and len(om):
                        d = np.abs(of[:, None] - om[None, :])
                        match = float(np.mean(d.min(axis=1) <= 25))
                        extras = int(np.sum(d.min(axis=0) > 25))
                    else:
                        match, extras = np.nan, 0
                    qc_rows.append({
                        "file": name, "block_s": block, "loss_target": rate,
                        "rep": rep, "loss_actual": actual,
                        "n_full": len(full), "n_masked": len(pm),
                        "spurious_in_gaps": int(in_gap.sum()),
                        "outside_match": match, "extras_outside": extras,
                    })
                    # detections inside deleted blocks are artifacts (no signal
                    # there); discard uniformly for all methods incl. baseline
                    pm = pm[~in_gap]

                    rec_seed = derive_seed(subject, block, rate, rep, "recover")
                    for mname, fn in METHODS.items():
                        pk_r = fn(pm, gaps, FS, np.random.default_rng(rec_seed))
                        est = all_metrics(pk_r, FS, dur)
                        for metric in ALL_METRICS:
                            tv, ev = truth[name][metric], est[metric]
                            rel = ((ev - tv) / tv
                                   if np.isfinite(tv) and np.isfinite(ev) and tv != 0
                                   else np.nan)
                            rows.append({
                                "subject": subject, "segment": seg, "file": name,
                                "duration_s": dur, "block_s": block,
                                "loss_target": rate, "loss_actual": actual,
                                "rep": rep, "seed": seed, "method": mname,
                                "metric": metric, "truth": tv, "estimate": ev,
                                "rel_err": rel,
                                "abs_err": ev - tv if np.isfinite(ev) and np.isfinite(tv) else np.nan,
                            })
        print(f"[{fi + 1}/{len(kept)}] {name} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results_long.csv"), index=False)
    pd.DataFrame(qc_rows).to_csv(os.path.join(OUT, "detection_qc.csv"), index=False)
    print(f"results_long: {len(df)} rows; detection_qc: {len(qc_rows)} rows; "
          f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()