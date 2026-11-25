"""Matched-composition contrast between `_1` (182 s) and `_2` (62 s).

The plain stratified comparison in summary_v2.py pools over block lengths, but
the two segments do NOT get the same mix of block lengths: on a 62 s record a
2, 5, 10 s block grid is feasible but a 30 s block is only feasible at 30/50%
nominal loss (and then it realises 48.4%, see loss_actual_table.csv). So the
plain `_1` vs `_2` gap bundles three things:

    window length  +  which block lengths are feasible  +  integer-block
                                                           quantisation

Here the first two confounds are removed by keeping only the (block, rate)
cells that are feasible in BOTH segments -- blocks of 2 s and 5 s at all five
nominal rates -- and recomputing the same statistic on that matched subset.
What is left is (window length) + (any quantisation difference that survives
matching); the latter is reported separately in loss_actual_table.csv.

Outputs:
  out_v2/seg_contrast_<split>.csv   one row per (metric, method, rate):
                                    seg1 vs seg2 median |rel err| on matched cells
  seg_contrast.log

Run:  python src/seg_contrast.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out_v2")
METHOD_ORDER = ["none", "mean_global", "mean_local", "random_sd"]
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
MATCH_BLOCKS = [2, 5]
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def med(df, metric, method, rate):
    q = df[(df.metric == metric) & (df.method == method)
           & np.isclose(df.loss_target, rate)]
    r = q["rel_err"].abs()
    r = r[np.isfinite(r)]
    return (float(r.median()), int(len(r))) if len(r) else (np.nan, 0)


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in ["excluded", "all"]:
        path = os.path.join(OUT, f"results_long_{split}.csv")
        if not os.path.exists(path):
            say(f"MISSING {path}")
            continue
        df = pd.read_csv(path)
        m = df[df.block_s.isin(MATCH_BLOCKS)]
        d1 = m[m.segment == 1]
        d2 = m[m.segment == 2]
        say(f"== {split}: matched cells = blocks {MATCH_BLOCKS} s x all 5 rates ==")
        say(f"   records: _1 = {d1.file.nunique()}, _2 = {d2.file.nunique()}; "
            f"rows: {len(d1)} / {len(d2)}")
        rows = []
        for metric in ["LF", "HF", "LF_HF", "TP", "RMSSD", "SDNN"]:
            for method in METHOD_ORDER:
                for rate in RATES:
                    a, na = med(d1, metric, method, rate)
                    b, nb = med(d2, metric, method, rate)
                    rows.append({"split": split, "metric": metric,
                                 "method": method, "loss_target": rate,
                                 "seg1_matched": a, "n_seg1": na,
                                 "seg2_matched": b, "n_seg2": nb,
                                 "delta": b - a,
                                 "ratio": (b / a if np.isfinite(a) and a > 0
                                           else np.nan)})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(OUT, f"seg_contrast_{split}.csv"), index=False)

        pd.set_option("display.width", 200)
        for metric in ["LF", "HF"]:
            q = t[t.metric == metric].pivot_table(
                index="method", columns="loss_target", values="seg1_matched")
            r = t[t.metric == metric].pivot_table(
                index="method", columns="loss_target", values="seg2_matched")
            say(f"  {metric} matched | median abs rel err % , segment _1:")
            say((q * 100).round(1).to_string())
            say(f"  {metric} matched | segment _2:")
            say((r * 100).round(1).to_string())
        say("")

    with open(os.path.join(OUT, "seg_contrast.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()