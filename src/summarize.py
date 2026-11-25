"""Aggregate results_long.csv -> summary.csv, thresholds.csv, figures.

Run:  python src/summarize.py   (after run_experiment.py)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out")
METHOD_ORDER = ["none", "mean_global", "mean_local", "random_sd"]
LABEL = {
    "none": "no recovery (baseline)",
    "mean_global": "M1 global-mean RR",
    "mean_local": "M2 local edge RR",
    "random_sd": "M3 mean±1SD random",
}
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
# stress decision: LF/HF up = stress up; HF down = stress up
FLIP_RULE = {"LF_HF": "up", "HF": "down"}


def main():
    df = pd.read_csv(os.path.join(OUT, "results_long.csv"))
    df["abs_rel"] = df["rel_err"].abs()

    g = df.groupby(["method", "loss_target", "metric"], sort=False)
    summ = g.agg(
        n=("rel_err", lambda s: int(np.isfinite(s).sum())),
        bias=("rel_err", "mean"),
        bias_median=("rel_err", "median"),
        sd_relerr=("rel_err", "std"),
        median_abs_relerr=("abs_rel", "median"),
        mean_abs_relerr=("abs_rel", "mean"),
        loss_actual_mean=("loss_actual", "mean"),
    ).reset_index()

    # paired-segment direction agreement and conclusion flip rate
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
    summ = summ.merge(da.reset_index(name="dir_agree"),
                      on=["method", "loss_target", "metric"], how="left")
    summ = summ.merge(fl.reset_index(name="flip_rate"),
                      on=["method", "loss_target", "metric"], how="left")
    summ["method"] = pd.Categorical(summ["method"], METHOD_ORDER, ordered=True)
    summ = summ.sort_values(["method", "loss_target", "metric"])
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # thresholds: first loss rate (ascending) where median |rel err| crosses 10/20%
    thr_rows = []
    for method in METHOD_ORDER:
        for metric in sorted(df["metric"].unique()):
            s = summ[(summ.method == method) & (summ.metric == metric)]
            s = s.set_index("loss_target")["median_abs_relerr"].reindex(RATES)
            cross10 = next((r for r in RATES if s.get(r, np.nan) > 0.10), np.nan)
            cross20 = next((r for r in RATES if s.get(r, np.nan) > 0.20), np.nan)
            thr_rows.append({"method": method, "metric": metric,
                             "thr10_loss": cross10, "thr20_loss": cross20})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(os.path.join(OUT, "thresholds.csv"), index=False)

    # heatmap: HF median |relative error|
    hf = summ[summ.metric == "HF"].pivot_table(
        index="method", columns="loss_target", values="median_abs_relerr")
    hf = hf.reindex(index=METHOD_ORDER, columns=RATES)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    from matplotlib.colors import LogNorm
    vals = hf.values * 100
    im = ax.imshow(vals, cmap="viridis", aspect="auto",
                   norm=LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    ax.set_xticks(range(len(RATES)), [f"{r:.0%}" for r in RATES])
    ax.set_yticks(range(len(METHOD_ORDER)), [LABEL[m] for m in METHOD_ORDER])
    for i in range(hf.shape[0]):
        for j in range(hf.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:.0f}" if np.isfinite(v) else "-",
                    ha="center", va="center",
                    color="white" if v < np.sqrt(np.nanmin(vals) * np.nanmax(vals)) else "black",
                    fontsize=10)
    ax.set_xlabel("packet loss rate (nominal)")
    ax.set_title("HF median |relative error| (%) — pooled over block lengths,\n"
                 "67 records × 3 reps; truth = same pipeline on intact signal (log color)")
    fig.colorbar(im, ax=ax, label="HF median |relative error| (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_heatmap_hf.png"), dpi=150)
    plt.close(fig)

    # threshold curves
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
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_threshold.png"), dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("== thresholds (first loss rate crossing 10% / 20% median |rel err|) ==")
    print(thr[thr.metric.isin(["HF", "LF_HF", "LF", "TP", "RMSSD", "SDNN"])]
          .to_string(index=False))
    print("\n== HF signed MEDIAN rel err (%) by method x rate ==")
    hb = summ[summ.metric == "HF"].pivot_table(
        index="method", columns="loss_target", values="bias_median").reindex(
        index=METHOD_ORDER, columns=RATES)
    print((hb * 100).round(1).to_string())
    print("\n== flip rate (LF_HF / HF) by method x rate ==")
    for metric in ["LF_HF", "HF"]:
        fp = summ[summ.metric == metric].pivot_table(
            index="method", columns="loss_target", values="flip_rate").reindex(
            index=METHOD_ORDER, columns=RATES)
        print(metric)
        print((fp * 100).round(1).to_string())
    print("\nwrote summary.csv, thresholds.csv, fig_heatmap_hf.png, fig_threshold.png")


if __name__ == "__main__":
    main()