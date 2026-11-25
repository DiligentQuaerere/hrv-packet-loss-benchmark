"""v2 grid runner: identical pipeline to run_experiment.py (v1) but

  * scans ALL 72 records (no record-level QC exclusion at scan time), so the
    sensitivity analysis of revision B is one single grid, not two;
  * writes into out_v2/ under explicit names, leaving v1's out/ untouched;
  * emits both split tables:
        results_long_all.csv       = 72 records
        results_long_excluded.csv  = the v1 67-record subset (5 QC-excluded
                                     files filtered out *after* the grid)
    Filtering after the fact is valid *and* stronger than rerunning, because the
    deletion mask of a given record depends only on
    sha256(subject|block|rate|rep) -- never on which other records are present.
    So results_long_excluded.csv must be row-identical to v1's
    out/results_long.csv; verify_v1_parity() checks exactly that.

Everything else (detector, mask model, recovery methods, HRV pipeline, seeds)
is imported from the v1 modules unchanged. See README_v2 for the v2-vs-v1
口径差异.

Run:  python src/run_experiment_v2.py
"""
import glob
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
from run_experiment import derive_seed, qc_flag, FS, BLOCKS, RATES, REPS

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "eegmat")
OUT_V1 = os.path.join(ROOT, "out")
OUT = os.path.join(ROOT, "out_v2")


def _tag(meta):
    return "%s_%d" % (meta["subject"], meta["segment"])


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "*.npy")))
    t0 = time.time()
    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say("v2 grid: %d records found in %s" % (len(files), DATA))

    # ---- pass 1: truth metrics + QC flags for every record ----------------
    truth = {}
    truth_rows = []
    recs = []
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
        recs.append((name, subject, int(seg), sig, dur, flag))
    tdf = pd.DataFrame(truth_rows)
    tdf.to_csv(os.path.join(OUT, "truth_metrics.csv"), index=False)
    excluded = sorted(tdf.loc[tdf.excluded, "file"].tolist())
    say("truth done: %d records, %d QC-excluded (%s) [%.0fs]"
        % (len(recs), len(excluded), ", ".join(excluded) if excluded else "none",
           time.time() - t0))

    # ---- pass 2: the full grid over all records ---------------------------
    rows = []
    qc_rows = []
    for i, (name, subject, seg, sig, dur, flag) in enumerate(recs):
        full = pan_tompkins(sig, FS)
        for block in BLOCKS:
            for rate in RATES:
                for rep in range(REPS):
                    seed = derive_seed(subject, block, rate, rep)
                    gaps, actual = make_gaps(dur, block, rate,
                                             np.random.default_rng(seed))
                    if not gaps:
                        continue  # infeasible combo
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
                        "rec_excluded": flag,
                    })
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
                                "abs_err": (ev - tv
                                            if np.isfinite(ev) and np.isfinite(tv) else np.nan),
                                "rec_excluded": flag,
                            })
        if (i + 1) % 10 == 0 or i + 1 == len(recs):
            say("[%d/%d] %s [%.0fs]" % (i + 1, len(recs), name, time.time() - t0))

    df = pd.DataFrame(rows)
    qdf = pd.DataFrame(qc_rows)
    df.to_csv(os.path.join(OUT, "results_long_all.csv"), index=False)
    qdf.to_csv(os.path.join(OUT, "detection_qc_all.csv"), index=False)

    keep = ~df["rec_excluded"]
    df[keep].to_csv(os.path.join(OUT, "results_long_excluded.csv"), index=False)
    qdf[~qdf["rec_excluded"]].to_csv(os.path.join(OUT, "detection_qc_excluded.csv"),
                                     index=False)
    say("results_long_all: %d rows (%d records)"
        % (len(df), df.file.nunique()))
    say("results_long_excluded: %d rows (%d records)"
        % (int(keep.sum()), df.loc[keep, "file"].nunique()))

    # ---- pass 3: parity check against v1 ---------------------------------
    say("")
    say("== v1 parity check ==")
    v1_path = os.path.join(OUT_V1, "results_long.csv")
    if os.path.exists(v1_path):
        v1 = pd.read_csv(v1_path)
        a = df[keep].reset_index(drop=True)
        cols = [c for c in v1.columns if c in a.columns]
        if len(v1) != len(a):
            say("ROW COUNT MISMATCH: v1=%d v2_excluded=%d" % (len(v1), len(a)))
        else:
            say("row count identical: %d" % len(v1))
        bad = []
        for c in cols:
            sv, sa = v1[c], a[c]
            if sv.dtype.kind in "fc" and sa.dtype.kind in "fc":
                neq = ~(np.isclose(sv.to_numpy(float), sa.to_numpy(float),
                                   rtol=0, atol=0, equal_nan=True))
            else:
                neq = (sv.astype(str).to_numpy() != sa.astype(str).to_numpy())
            if neq.sum():
                bad.append("%s:%d" % (c, int(neq.sum())))
        say("columns compared: %d; columns with differences: %s"
            % (len(cols), ", ".join(bad) if bad else "none"))
        # truth table parity
        v1t = pd.read_csv(os.path.join(OUT_V1, "truth_metrics.csv"))
        shared = [c for c in v1t.columns if c in tdf.columns]
        same = v1t[shared].reset_index(drop=True).equals(
            tdf[shared].reset_index(drop=True))
        say("truth_metrics.csv identical on %d shared columns: %s"
            % (len(shared), same))
    else:
        say("v1 results_long.csv not found; skipped")

    say("")
    say("total %.0fs" % (time.time() - t0))
    with open(os.path.join(OUT, "run_v2.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()