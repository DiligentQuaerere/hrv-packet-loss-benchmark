"""v2 aggregation: same statistics as summarize.py (v1) but stratified by
segment (`_1` rest vs `_2` mental-arithmetic stress) and repeated for both
record splits (excluded = v1's 67-record set, all = 72 records).

Outputs (all under out_v2/):
  summary_<split>_<stratum>.csv        method x loss_rate x metric, with n
  summary_by_block_<split>_<stratum>.csv   ... additionally split by block_s
  thresholds_<split>_<stratum>.csv     first crossing of 10% / 20% median |rel err|
  thresholds_compare_excluded_vs_all.csv   revision B difference table
  lf_attribution_<split>.csv           revision A decomposition for LF/HF
  fig_heatmap_hf_<split>_<stratum>.png
  fig_threshold_<split>_<stratum>.png
  fig_stratified_<split>.png           seg1 vs seg2 curves for LF and HF
  fig_threshold_compare.png            revision B
  summarize_v2.log

split   in {excluded, all}
stratum in {pooled, seg1, seg2}

dir_agree / flip_rate are paired-segment statistics (they compare each
subject's _1 against its own _2) and are therefore only defined for
stratum=pooled; they are NaN elsewhere by construction, not by failure.

Run:  python src/summarize_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out_v2")
METHOD_ORDER = ["none", "mean_global", "mean_local", "random_sd"]
LABEL = {
    "none": "no recovery (baseline)",
    "mean_global": "M1 global-mean RR",
    "mean_local": "M2 local edge RR",
    "random_sd": "M3 mean±1SD random",
}
SHORT = {"none": "none", "mean_global": "M1", "mean_local": "M2",
         "random_sd": "M3"}
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
FLIP_RULE = {"LF_HF": "up", "HF": "down"}
STRATA = {"pooled": None, "seg1": 1, "seg2": 2}
SPLITS = ["excluded", "all"]
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def aggregate(df):
    """Same aggregation as v1 summarize.main(), minus the paired stats."""
    d = df.copy()
    d["abs_rel"] = d["rel_err"].abs()
    g = d.groupby(["method", "loss_target", "metric"], sort=False)
    s = g.agg(
        n=("rel_err", lambda x: int(np.isfinite(x).sum())),
        n_records=("file", "nunique"),
        bias=("rel_err", "mean"),
        bias_median=("rel_err", "median"),
        sd_relerr=("rel_err", "std"),
        median_abs_relerr=("abs_rel", "median"),
        mean_abs_relerr=("abs_rel", "mean"),
        loss_actual_mean=("loss_actual", "mean"),
    ).reset_index()
    return s


def paired_stats(df):
    """v1's dir_agree / flip_rate: needs both segments per subject."""
    p = df.pivot_table(
        index=["subject", "block_s", "loss_target", "rep", "method", "metric"],
        columns="segment", values=["truth", "estimate"],
    )
    t1, t2 = p[("truth", 1)], p[("truth", 2)]
    e1, e2 = p[("estimate", 1)], p[("estimate", 2)]
    ok = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(e1) & np.isfinite(e2)
    dt, de = np.sign(t2 - t1), np.sign(e2 - e1)
    valid = ok & (dt != 0)
    pair = pd.DataFrame({
        "method": np.asarray(p.index.get_level_values("method")),
        "loss_target": np.asarray(p.index.get_level_values("loss_target")),
        "metric": np.asarray(p.index.get_level_values("metric")),
        "dir_agree": np.where(valid, (dt == de).astype(float), np.nan),
    })
    flips = []
    for metric, rule in FLIP_RULE.items():
        m = np.asarray(ok & (p.index.get_level_values("metric") == metric))
        if rule == "up":
            dec_t, dec_e = np.asarray(t2 > t1), np.asarray(e2 > e1)
        else:
            dec_t, dec_e = np.asarray(t2 < t1), np.asarray(e2 < e1)
        flips.append(pd.DataFrame({
            "method": np.asarray(p.index.get_level_values("method"))[m],
            "loss_target": np.asarray(p.index.get_level_values("loss_target"))[m],
            "metric": metric,
            "flip": (dec_t[m] != dec_e[m]).astype(float),
        }))
    fl = pd.concat(flips).groupby(["method", "loss_target", "metric"])["flip"].mean()
    da = pair.groupby(["method", "loss_target", "metric"])["dir_agree"].mean()
    return da.rename("dir_agree").reset_index(), fl.rename("flip_rate").reset_index()


def thresholds(summ, metrics):
    rows = []
    for method in METHOD_ORDER:
        for metric in metrics:
            s = summ[(summ.method == method) & (summ.metric == metric)]
            if s.empty:
                continue
            s = s.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
            cross10 = next((r for r in RATES
                            if np.isfinite(s.get(r, np.nan)) and s.get(r) > 0.10),
                           np.nan)
            cross20 = next((r for r in RATES
                            if np.isfinite(s.get(r, np.nan)) and s.get(r) > 0.20),
                           np.nan)
            rows.append({"method": method, "metric": metric,
                         "thr10_loss": cross10, "thr20_loss": cross20})
    return pd.DataFrame(rows)


def thresholds_by_block(sb, metrics):
    """Same crossing rule, but along loss rate *within* one block length.

    Added in v2 because the pooled table hides a large block-length effect:
    at a nominal 5% loss the median LF error ranges from 5.5% (10 s blocks) to
    178% (2 s blocks) on the same records. A threshold that pools block lengths
    is therefore not a property of the loss rate alone.
    """
    rows = []
    for method in METHOD_ORDER:
        for block in sorted(sb.block_s.unique()):
            for metric in metrics:
                s = sb[(sb.method == method) & (sb.metric == metric)
                       & (sb.block_s == block)]
                if s.empty:
                    continue
                s = s.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
                cross10 = next((r for r in RATES
                                if np.isfinite(s.get(r, np.nan)) and s.get(r) > 0.10),
                               np.nan)
                cross20 = next((r for r in RATES
                                if np.isfinite(s.get(r, np.nan)) and s.get(r) > 0.20),
                               np.nan)
                rows.append({"method": method, "metric": metric,
                             "block_s": block,
                             "thr10_loss": cross10, "thr20_loss": cross20,
                             "n_rates_evaluated": int(s.notna().sum())})
    return pd.DataFrame(rows)


def block_effect_fig(sb, path, title):
    """x = block length (categorical positions), series = nominal loss rate."""
    metrics = ["LF", "HF"]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(13, 7), sharex=True)
    blocks = sorted(sb.block_s.unique())
    pos = {b: i for i, b in enumerate(blocks)}
    for i, metric in enumerate(metrics):
        for j, method in enumerate(["mean_global", "mean_local", "random_sd"]):
            ax = axes[i, j]
            for rate in RATES:
                q = sb[(sb.metric == metric) & (sb.method == method)
                       & np.isclose(sb.loss_target, rate)]
                q = q.set_index("block_s")["median_abs_relerr"].reindex(blocks)
                ax.plot([pos[b] for b in blocks], q.values * 100, "o-",
                        label=f"{rate:.0%}")
            ax.set_yscale("log")
            ax.set_xticks(range(len(blocks)), [str(b) for b in blocks])
            ax.set_xlim(-0.3, len(blocks) - 0.7)
            ax.minorticks_off()
            ax.grid(alpha=0.3, which="major")
            ax.set_title(f"{metric} — {LABEL[method]}", fontsize=9)
            if i == len(metrics) - 1:
                ax.set_xlabel("block length (s)")
            if j == 0:
                ax.set_ylabel("median |relative error| (%)")
            if i == 0 and j == 2:
                ax.legend(fontsize=7, title="nominal loss", ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(summ, path, title, metric="HF"):
    sub = summ[summ.metric == metric].pivot_table(
        index="method", columns="loss_target", values="median_abs_relerr")
    sub = sub.reindex(index=METHOD_ORDER, columns=RATES)
    vals = sub.values * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vmin = np.nanmin(vals[vals > 0]) if np.any(vals > 0) else 1e-3
    im = ax.imshow(vals, cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=np.nanmax(vals)))
    ax.set_xticks(range(len(RATES)), [f"{r:.0%}" for r in RATES])
    ax.set_yticks(range(len(METHOD_ORDER)), [LABEL[m] for m in METHOD_ORDER])
    mid = np.sqrt(vmin * np.nanmax(vals))
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0f}" if np.isfinite(v) else "-",
                    ha="center", va="center",
                    color="white" if v < mid else "black", fontsize=10)
    ax.set_xlabel("packet loss rate (nominal)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"{metric} median |relative error| (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def threshold_curves(summ, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, metric in zip(axes, ["HF", "LF_HF"]):
        for method in METHOD_ORDER:
            s = summ[(summ.method == method) & (summ.metric == metric)]
            s = s.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
            ax.plot(np.array(RATES) * 100, s.values * 100, "o-", label=LABEL[method])
        ax.axhline(10, color="gray", ls="--", lw=1)
        ax.axhline(20, color="gray", ls=":", lw=1)
        ax.text(51, 10, "10%", va="center", fontsize=9, color="gray")
        ax.text(51, 20, "20%", va="center", fontsize=9, color="gray")
        ax.set_yscale("log")
        ax.set_xlabel("packet loss rate (%)")
        ax.set_xticks([5, 10, 20, 30, 50])
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{metric}: median |relative error|")
    axes[0].set_ylabel("median |relative error| (%)")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stratified_fig(split, summ_by_stratum, path):
    """key revision-A evidence: same metric, seg1 vs seg2, per method."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, metric in zip(axes, ["LF", "HF"]):
        for method in METHOD_ORDER:
            style = {"color": None}
            for stratum, ls, mk in [("seg1", "-", "o"), ("seg2", "--", "s")]:
                s = summ_by_stratum[stratum]
                s = s[(s.method == method) & (s.metric == metric)]
                s = s.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
                line, = ax.plot(np.array(RATES) * 100, s.values * 100,
                                ls=ls, marker=mk, ms=4, lw=1.6,
                                label=f"{SHORT[method]} {'_1' if stratum=='seg1' else '_2'}")
                style["color"] = line.get_color()
        ax.axhline(10, color="gray", ls="--", lw=1)
        ax.axhline(20, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("packet loss rate (%)")
        ax.set_xticks([5, 10, 20, 30, 50])
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{metric}: seg_1 (182 s rest, solid) vs seg_2 (62 s stress, dashed)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    axes[0].set_ylabel("median |relative error| (%)")
    fig.suptitle(f"revision A — stratified by segment ({split} records)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    all_summ = {}
    all_thr = {}

    for split in SPLITS:
        path = os.path.join(OUT, f"results_long_{split}.csv")
        if not os.path.exists(path):
            say(f"MISSING {path} -- run src/run_experiment_v2.py first")
            continue
        df = pd.read_csv(path)
        say(f"== split '{split}': {len(df)} rows, {df.file.nunique()} records ==")
        metrics = sorted(df.metric.unique())
        da = fl = None
        if df.segment.nunique() == 2:
            da, fl = paired_stats(df)

        per_stratum = {}
        for stratum, seg in STRATA.items():
            d = df if seg is None else df[df.segment == seg]
            s = aggregate(d)
            if da is not None:
                s = s.merge(da, on=["method", "loss_target", "metric"], how="left")
                s = s.merge(fl, on=["method", "loss_target", "metric"], how="left")
            else:
                s["dir_agree"] = np.nan
                s["flip_rate"] = np.nan
            s["method"] = pd.Categorical(s.method, METHOD_ORDER, ordered=True)
            s = s.sort_values(["method", "loss_target", "metric"])
            s.to_csv(os.path.join(OUT, f"summary_{split}_{stratum}.csv"), index=False)

            # by-block breakdown (needed to separate window from block effects)
            dd = d.copy()
            dd["abs_rel"] = dd["rel_err"].abs()
            sb = dd.groupby(["method", "loss_target", "block_s", "metric"],
                            sort=False).agg(
                n=("rel_err", lambda x: int(np.isfinite(x).sum())),
                n_records=("file", "nunique"),
                median_abs_relerr=("abs_rel", "median"),
                bias_median=("rel_err", "median"),
                loss_actual_mean=("loss_actual", "mean"),
            ).reset_index().sort_values(["method", "loss_target", "block_s", "metric"])
            sb.to_csv(os.path.join(OUT, f"summary_by_block_{split}_{stratum}.csv"),
                      index=False)

            thr = thresholds(s, metrics)
            thr.to_csv(os.path.join(OUT, f"thresholds_{split}_{stratum}.csv"),
                       index=False)
            thrb = thresholds_by_block(sb, metrics)
            thrb.to_csv(os.path.join(OUT,
                                     f"thresholds_by_block_{split}_{stratum}.csv"),
                        index=False)
            per_stratum[stratum] = s
            all_summ[(split, stratum)] = s
            all_summ[(split, "byblock_" + stratum)] = sb
            all_thr[(split, stratum)] = thr
            nrec = d.file.nunique()
            say(f"  stratum {stratum:6s}: {nrec:2d} records, "
                f"{len(d)} rows, {len(s)} cells -> "
                f"summary_{split}_{stratum}.csv")

        heatmap(per_stratum["pooled"],
                os.path.join(OUT, f"fig_heatmap_hf_{split}_pooled.png"),
                f"HF median |relative error| (%) -- {split}, both segments pooled")
        for stratum, seg in [("seg1", 1), ("seg2", 2)]:
            d = df[df.segment == seg]
            heatmap(per_stratum[stratum],
                    os.path.join(OUT, f"fig_heatmap_hf_{split}_{stratum}.png"),
                    f"HF median |relative error| (%) -- {split}, segment _{seg} "
                    f"({d.file.nunique()} records)")
        threshold_curves(per_stratum["pooled"],
                         os.path.join(OUT, f"fig_threshold_{split}_pooled.png"),
                         f"{split} records, both segments pooled")
        for stratum, seg in [("seg1", 1), ("seg2", 2)]:
            threshold_curves(per_stratum[stratum],
                             os.path.join(OUT, f"fig_threshold_{split}_{stratum}.png"),
                             f"{split} records, segment _{seg} only")
        stratified_fig(split, per_stratum,
                       os.path.join(OUT, f"fig_stratified_{split}.png"))
        block_effect_fig(all_summ[(split, "byblock_pooled")],
                         os.path.join(OUT, f"fig_block_effect_{split}.png"),
                         f"block length dominates loss rate ({split} records, "
                         f"both segments pooled)")
        say("")

    # ---------------- revision A: LF attribution -------------------------
    say("== revision A: is LF error still driven by interpolation after "
        "stratification? ==")
    for split in SPLITS:
        if (split, "seg1") not in all_summ:
            continue
        s1 = all_summ[(split, "seg1")]
        s2 = all_summ[(split, "seg2")]
        rows = []
        for metric in ["LF", "HF", "LF_HF", "TP", "RMSSD", "SDNN"]:
            for method in METHOD_ORDER:
                for rate in RATES:
                    def get(s, m=method, r=rate):
                        q = s[(s.method == m) & (s.metric == metric)
                              & (np.isclose(s.loss_target, r))]
                        return float(q["median_abs_relerr"].iloc[0]) if len(q) else np.nan
                    a, b = get(s1), get(s2)
                    rows.append({"metric": metric, "method": method,
                                 "loss_target": rate,
                                 "med_abs_rel_seg1": a, "med_abs_rel_seg2": b,
                                 "ratio_seg2_over_seg1": (b / a if a and np.isfinite(a)
                                                          and a > 0 else np.nan),
                                 "delta_seg2_minus_seg1": b - a})
        att = pd.DataFrame(rows)
        # per (metric, rate): spread across the three interpolation methods,
        # per segment -- the "does the method matter?" signal
        interp = ["mean_global", "mean_local", "random_sd"]
        sp = []
        for metric in att.metric.unique():
            for rate in RATES:
                sel = att[(att.metric == metric) & (np.isclose(att.loss_target, rate))]
                for stratum, col in [("seg1", "med_abs_rel_seg1"),
                                     ("seg2", "med_abs_rel_seg2")]:
                    sel_i = sel[sel.method.isin(interp)].dropna(subset=[col])
                    if len(sel_i):
                        best = sel_i.loc[sel_i[col].idxmin()]
                        sp.append({"metric": metric, "loss_target": rate,
                                   "stratum": stratum,
                                   "method_spread": float(sel_i[col].max()
                                                          - sel_i[col].min()),
                                   "method_max": float(sel_i[col].max()),
                                   "method_min": float(sel_i[col].min()),
                                   "method_best": best["method"],
                                   "method_best_value": float(best[col])})
                    else:
                        sp.append({"metric": metric, "loss_target": rate,
                                   "stratum": stratum,
                                   "method_spread": np.nan, "method_max": np.nan,
                                   "method_min": np.nan, "method_best": None,
                                   "method_best_value": np.nan})
        spdf = pd.DataFrame(sp)
        att.to_csv(os.path.join(OUT, f"lf_attribution_{split}.csv"), index=False)
        spdf.to_csv(os.path.join(OUT, f"method_spread_{split}.csv"), index=False)
        say(f"  wrote lf_attribution_{split}.csv, method_spread_{split}.csv")

    # ---------------- revision B: excluded vs all ------------------------
    say("")
    say("== revision B: does excluding dirty records make the safe range "
        "look too good? ==")
    cmp_rows = []
    for stratum in STRATA:
        te, ta = all_thr.get(("excluded", stratum)), all_thr.get(("all", stratum))
        if te is None or ta is None:
            continue
        t = te.merge(ta, on=["method", "metric"], suffixes=("_excluded", "_all"))
        for _, r in t.iterrows():
            cmp_rows.append({
                "stratum": stratum, "method": r.method, "metric": r.metric,
                "thr10_excluded": r.thr10_loss_excluded,
                "thr10_all": r.thr10_loss_all,
                "thr10_delta_all_minus_excluded":
                    (r.thr10_loss_all - r.thr10_loss_excluded
                     if np.isfinite(r.thr10_loss_all) and np.isfinite(r.thr10_loss_excluded)
                     else np.nan),
                "thr20_excluded": r.thr20_loss_excluded,
                "thr20_all": r.thr20_loss_all,
                "thr20_delta_all_minus_excluded":
                    (r.thr20_loss_all - r.thr20_loss_excluded
                     if np.isfinite(r.thr20_loss_all) and np.isfinite(r.thr20_loss_excluded)
                     else np.nan),
                "verdict": ("more pessimistic when dirty records kept"
                            if (np.isfinite(r.thr10_loss_all) and np.isfinite(r.thr10_loss_excluded)
                                and r.thr10_loss_all < r.thr10_loss_excluded)
                            else ("same" if (np.isfinite(r.thr10_loss_all) and np.isfinite(r.thr10_loss_excluded)
                                             and r.thr10_loss_all == r.thr10_loss_excluded)
                                  else "other/NaN")),
            })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(OUT, "thresholds_compare_excluded_vs_all.csv"), index=False)

    # summary of how often the safe range shrinks
    for stratum in ["pooled", "seg1", "seg2"]:
        c = cmp[cmp.stratum == stratum]
        if c.empty:
            continue
        fin = c[np.isfinite(c.thr10_delta_all_minus_excluded)]
        say(f"  {stratum:6s}: {len(fin):2d} comparable cells; "
            f"threshold shrinks when dirty records are kept in "
            f"{int((fin.thr10_delta_all_minus_excluded < 0).sum())}, "
            f"unchanged in {int((fin.thr10_delta_all_minus_excluded == 0).sum())}, "
            f"grows in {int((fin.thr10_delta_all_minus_excluded > 0).sum())}")

    # figure: HF/LF threshold curves, excluded vs all
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, metric in zip(axes, ["LF", "HF"]):
        for split, ls in [("excluded", "-"), ("all", "--")]:
            if (split, "pooled") not in all_summ:
                continue
            s = all_summ[(split, "pooled")]
            for method in METHOD_ORDER:
                q = s[(s.method == method) & (s.metric == metric)]
                q = q.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
                ax.plot(np.array(RATES) * 100, q.values * 100, ls=ls, marker="o",
                        ms=3.5, lw=1.5,
                        label=f"{SHORT[method]} ({split})")
        ax.axhline(10, color="gray", ls="--", lw=1)
        ax.axhline(20, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("packet loss rate (%)")
        ax.set_xticks([5, 10, 20, 30, 50])
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"{metric}: solid = QC-excluded (v1), dashed = all 72 records")
        ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    axes[0].set_ylabel("median |relative error| (%)")
    fig.suptitle("revision B — sensitivity of the error curve to QC exclusion",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_threshold_compare.png"), dpi=150)
    plt.close(fig)
    say("  wrote thresholds_compare_excluded_vs_all.csv, fig_threshold_compare.png")

    # ---------------- console tables for the write-up --------------------
    pd.set_option("display.width", 220)
    for split in SPLITS:
        for stratum in STRATA:
            key = (split, stratum)
            if key not in all_summ:
                continue
            s = all_summ[key]
            say("")
            say(f"---- median |rel err| (%) and n : split={split} "
                f"stratum={stratum} ----")
            for metric in ["LF", "HF", "LF_HF", "TP", "RMSSD", "SDNN"]:
                q = s[s.metric == metric].pivot_table(
                    index="method", columns="loss_target",
                    values="median_abs_relerr").reindex(index=METHOD_ORDER,
                                                        columns=RATES)
                nn = s[s.metric == metric].pivot_table(
                    index="method", columns="loss_target",
                    values="n").reindex(index=METHOD_ORDER, columns=RATES)
                say(f"{metric}: median |rel err| %")
                say((q * 100).round(1).to_string())
                say(f"{metric}: n")
                say(nn.astype("Int64").to_string())

            # block-length effect (the v2 headline)
            sb = all_summ[(split, "byblock_" + stratum)]
            for metric in ["LF", "HF"]:
                for method in ["mean_global", "mean_local", "random_sd"]:
                    q = sb[(sb.metric == metric) & (sb.method == method)] \
                        .pivot_table(index="block_s", columns="loss_target",
                                     values="median_abs_relerr")
                    say(f"{metric} {method}: median |rel err| % by block_s")
                    say((q * 100).round(1).to_string())

            tb = pd.read_csv(os.path.join(
                OUT, f"thresholds_by_block_{split}_{stratum}.csv"))
            say(f"thresholds by block ({split}/{stratum}) — first rate crossing 10%:")
            for method in METHOD_ORDER:
                p = tb[(tb.method == method) & (tb.metric.isin(["LF", "HF", "LF_HF"]))] \
                    .pivot_table(index="block_s", columns="metric", values="thr10_loss")
                say(f"  {method}:")
                say(p.to_string())

    with open(os.path.join(OUT, "summarize_v2.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()