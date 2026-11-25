"""Revision-A control: how much does the 62 s window alone change HRV?

The `_2` records are 62 s long. A 62 s window contains <2.5 cycles of the LF
band (0.04-0.15 Hz -> 25 s .. 6.7 s per cycle), so LF computed on `_2` is not
the same quantity as LF computed on a 182 s `_1` record -- even when the signal
is perfectly clean and nothing was ever deleted.

This script quantifies exactly that, with NO packet loss involved:

  for every `_1` record with duration >= 62 s,
      truth LF_full  <- all_metrics(detected peaks, duration = whole record)
      for 6 window offsets spread over the record,
          LF_62 <- all_metrics(same peaks restricted to [t0, t0+62 s), 62 s)
      rel = (LF_62 - LF_full) / LF_full

The median |rel| is the *window-only* error floor: any `_2` LF error at or
below this number cannot be blamed on block loss or on interpolation.

Peaks are restricted (not re-detected) inside the window on purpose: this
isolates the window effect from detector variability. Detector-attributable
error is handled separately by src/fantasia_validate.py.

Outputs:
  out_v2/window_control_windows.csv    one row per (record, offset, metric)
  out_v2/window_control_summary.csv    per metric: median / quartiles / n
  out_v2/loss_actual_table.csv         realised loss rate by stratum x block x rate
  window_control.log

Run:  python src/window_control.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins
from hrv import all_metrics, ALL_METRICS
from masks import make_gaps
from run_experiment import derive_seed, qc_flag, FS, BLOCKS, RATES, REPS

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "eegmat")
OUT = os.path.join(ROOT, "out_v2")
WIN = 62.0
N_OFFSETS = 6
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "*.npy")))
    rows = []
    skipped = []
    tdf = pd.read_csv(os.path.join(OUT, "truth_metrics.csv")) \
        if os.path.exists(os.path.join(OUT, "truth_metrics.csv")) else None
    excl = set(tdf.loc[tdf.excluded, "file"]) if tdf is not None else set()

    for f in files:
        name = os.path.basename(f)[:-4]
        subject, seg = name.split("_")
        if seg != "1":
            continue                      # only the long rest segment
        if name in excl:
            skipped.append((name, "QC-excluded in the v1 split"))
            continue
        sig = np.load(f)
        dur = len(sig) / FS
        if dur < WIN:
            skipped.append((name, "shorter than %.0f s" % WIN))
            continue
        pk = pan_tompkins(sig, FS)
        full = all_metrics(pk, FS, dur)
        offsets = np.linspace(0.0, dur - WIN, N_OFFSETS)
        for off in offsets:
            lo, hi = off * FS, (off + WIN) * FS
            sel = pk[(pk >= lo) & (pk < hi)] - lo
            m = all_metrics(sel, FS, WIN)
            for metric in ALL_METRICS:
                tv, ev = full[metric], m[metric]
                rel = ((ev - tv) / tv
                       if np.isfinite(tv) and np.isfinite(ev) and tv != 0
                       else np.nan)
                rows.append({"file": name, "subject": subject,
                             "duration_s": dur, "offset_s": float(off),
                             "metric": metric, "full": tv, "win62": ev,
                             "rel_err": rel})
        say("%s: dur=%.0f s, %d peaks, %d offsets"
            % (name, dur, len(pk), len(offsets)))

    d = pd.DataFrame(rows)
    d["abs_rel"] = d["rel_err"].abs()
    d.to_csv(os.path.join(OUT, "window_control_windows.csv"), index=False)

    g = d.groupby("metric")
    summ = g.agg(
        n=("rel_err", lambda x: int(np.isfinite(x).sum())),
        n_records=("file", "nunique"),
        bias_median=("rel_err", "median"),
        median_abs_relerr=("abs_rel", "median"),
        q25_abs_relerr=("abs_rel", lambda x: float(np.nanpercentile(x, 25))),
        q75_abs_relerr=("abs_rel", lambda x: float(np.nanpercentile(x, 75))),
        frac_abs_gt_10pct=("abs_rel", lambda x: float(np.nanmean(x > 0.10))),
        frac_abs_gt_20pct=("abs_rel", lambda x: float(np.nanmean(x > 0.20))),
    ).reset_index()
    order = {m: i for i, m in enumerate(ALL_METRICS)}
    summ = summ.sort_values("metric", key=lambda s: s.map(order))
    summ.to_csv(os.path.join(OUT, "window_control_summary.csv"), index=False)

    # within-record dispersion of LF across the 62 s windows
    lf = d[(d.metric == "LF") & np.isfinite(d.win62)]
    cv = lf.groupby("file")["win62"].agg(
        lambda x: (np.std(x, ddof=1) / np.mean(x)) if len(x) > 1 else np.nan)
    say("")
    say("== window-only control: 62 s window vs the same record's full length ==")
    say("(no packet loss, no interpolation -- window length/estimator alone)")
    for metric in ["LF", "HF", "LF_HF", "TP", "RMSSD", "SDNN"]:
        r = summ[summ.metric == metric]
        if r.empty:
            continue
        r = r.iloc[0]
        say("  %-6s median|rel| %6.1f%%  IQR %5.1f-%5.1f%%  "
            "|rel|>10%%: %5.1f%%  n=%d (%d records)"
            % (metric, r.median_abs_relerr * 100, r.q25_abs_relerr * 100,
               r.q75_abs_relerr * 100, r.frac_abs_gt_10pct * 100,
               r.n, r.n_records))
    say("  LF within-record CV across %d windows of 62 s: median %.1f%% "
        "(IQR %.1f-%.1f%%, n=%d records)"
        % (N_OFFSETS, np.nanmedian(cv) * 100,
           np.nanpercentile(cv, 25) * 100, np.nanpercentile(cv, 75) * 100,
           int(cv.notna().sum())))
    for nm, why in skipped:
        say("  skipped %s: %s" % (nm, why))

    # ---- realised loss rate: integer-block quantisation, per stratum ------
    say("")
    say("== realised loss rate (integer-block quantisation) ==")
    lrows = []
    for f in files:
        name = os.path.basename(f)[:-4]
        subject, seg = name.split("_")
        if name in excl:
            continue
        sig = np.load(f)
        dur = len(sig) / FS
        for block in BLOCKS:
            for rate in RATES:
                for rep in range(REPS):
                    seed = derive_seed(subject, block, rate, rep)
                    gaps, actual = make_gaps(dur, block, rate,
                                             np.random.default_rng(seed))
                    if not gaps:
                        continue
                    lrows.append({"file": name, "segment": int(seg),
                                  "block_s": block, "loss_target": rate,
                                  "loss_actual": actual})
    lf_df = pd.DataFrame(lrows)
    lt = lf_df.groupby(["segment", "block_s", "loss_target"]).agg(
        n_runs=("loss_actual", "size"),
        loss_actual_mean=("loss_actual", "mean"),
        loss_actual_min=("loss_actual", "min"),
        loss_actual_max=("loss_actual", "max"),
    ).reset_index()
    lt.to_csv(os.path.join(OUT, "loss_actual_table.csv"), index=False)
    for seg in (1, 2):
        say("-- segment _%d --" % seg)
        p = lt[lt.segment == seg].pivot_table(
            index="block_s", columns="loss_target", values="loss_actual_mean")
        say((p * 100).round(1).to_string() + "   [realised loss %, nominal 5/10/20/30/50]")
        q = lt[lt.segment == seg].pivot_table(
            index="block_s", columns="loss_target", values="n_runs")
        say(q.to_string() + "   [runs per cell; blank = infeasible, skipped]")

    with open(os.path.join(OUT, "window_control.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()