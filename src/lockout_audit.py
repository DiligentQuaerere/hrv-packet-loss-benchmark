"""Lock-out audit: how close did the detector get to the failure mode that
src/fantasia_validate.py found on 2 of 20 Fantasia records?

That failure is a *threshold lock-out*: `spki` only updates when a beat is
accepted, `npki` only rises on rejected candidates, so `thr1` can climb above
the real QRS amplitude and there is no re-initialisation path. The signature is
a stretch with no detections at all, i.e. at least one very long RR interval.

This script measures that signature directly on all 72 EEGGMAT records, instead
of arguing that the QC rule would have caught it. (It would not: a single
lock-out contributes ONE long RR, which on 87-338 intervals is 0.3-1.1% --
far below the QC rule's 5% cut-off.)

  rr_max          longest detected RR in the record
  n_rr_gt_2       how many RR intervals exceed 2 s
  max_ratio       rr_max / median RR
  n_rr_gt_1p66med number of intervals that would have triggered Pan-Tompkins
                  searchback (1.66 x mean RR) -- an upper bound on the number of
                  missed-beat clusters

Outputs: out_v2/lockout_audit.csv, lockout_audit.log

Run:  python src/lockout_audit.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins
from run_experiment import FS

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "eegmat")
OUT = os.path.join(ROOT, "out_v2")
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def main():
    os.makedirs(OUT, exist_ok=True)
    tdf = pd.read_csv(os.path.join(OUT, "truth_metrics.csv"))
    excl = set(tdf.loc[tdf.excluded, "file"])
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.npy"))):
        name = os.path.basename(f)[:-4]
        sig = np.load(f)
        pk = pan_tompkins(sig, FS)
        rr = np.diff(pk) / float(FS)
        med = float(np.median(rr))
        rows.append({
            "file": name, "segment": int(name.split("_")[1]),
            "qc_excluded": name in excl,
            "duration_s": len(sig) / float(FS), "n_peaks": int(len(pk)),
            "rr_median": med, "rr_min": float(rr.min()), "rr_max": float(rr.max()),
            "max_ratio": float(rr.max() / med),
            "n_rr_gt_2": int((rr > 2.0).sum()),
            "n_rr_gt_1p66med": int((rr > 1.66 * med).sum()),
        })
    d = pd.DataFrame(rows).sort_values("max_ratio", ascending=False)
    d.to_csv(os.path.join(OUT, "lockout_audit.csv"), index=False)

    say("== lock-out audit over all %d EEGMAT records ==" % len(d))
    say("  longest RR anywhere     : %.3f s  (%s)"
        % (d.rr_max.max(), d.loc[d.rr_max.idxmax(), "file"]))
    say("  records with any RR>2 s : %d" % int((d.n_rr_gt_2 > 0).sum()))
    say("  records with RR>1.66x median: %d (of which QC-excluded: %d)"
        % (int((d.n_rr_gt_1p66med > 0).sum()),
           int(d[(d.n_rr_gt_1p66med > 0)].qc_excluded.sum())))
    say("  max_ratio quantiles (p50/p90/p99/max): %s"
        % np.round(np.percentile(d.max_ratio, [50, 90, 99, 100]), 3).tolist())
    say("  shortest RR anywhere    : %.3f s" % d.rr_min.min())
    say("")
    say("  top 8 by max_ratio:")
    say(d.head(8)[["file", "qc_excluded", "n_peaks", "duration_s",
                   "rr_median", "rr_max", "max_ratio", "n_rr_gt_1p66med"]]
        .round(3).to_string(index=False))
    say("")
    say("  worst case for the QC rule: a single lock-out gives ONE long RR,"
        " i.e. %.2f%%-%.2f%% of intervals on these record lengths ->"
        " the 5%% rule cannot flag it."
        % (100.0 / d.n_peaks.max(), 100.0 / d.n_peaks.min()))

    with open(os.path.join(OUT, "lockout_audit.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()