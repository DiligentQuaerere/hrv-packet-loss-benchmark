"""Revision C: validate the project's own R-peak detector against the Fantasia
reference beat annotations, to bound how much of the "truth" error is
detector-attributable.

Facts established by reading the local copy in
data/fantasia/ (read-only, nothing written):

  * 40 records, each with .hea + .ecg (WFDB annotation stream, annotator name
    "ecg").  ANNOTATORS says:
        ecg <tab> corrected beat annotations from an automated detector
    i.e. these are *machine* annotations that were corrected, NOT independent
    human expert labels.  A high agreement here means "agrees with another
    detector on an easier signal", not "matches cardiologist truth".
  * Only 20 of the 40 records (all f2*) have a .dat waveform, and those .dat
    files are 660 s PREFIXES (see waveform_prefix_manifest.json:
    covered_samples).  The 20 f1* records have header + annotations but no
    signal file at all, so the detector cannot be run on them.
  * The annotation files describe the FULL record (~7000 s), so annotations
    must be truncated to the covered sample range before comparison.
  * f2y02 runs at 333 Hz, not 250 Hz -- the sampling rate is taken from each
    header, never assumed.

Matching: one-to-one, order-preserving two-pointer pairing within
+-TOL_MS (default 50 ms, as specified in the task).

Outputs:
  out_v2/fantasia_detector_validation.csv   one row per record
  out_v2/fantasia_detector_summary.csv      aggregate distribution
  out_v2/fantasia_notes.txt                 data-availability + caveats
  fantasia_validate.log

Run:  python src/fantasia_validate.py
"""
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ecg_detect import pan_tompkins

FANTASIA = r"\26-11-23\PNS_SNS\data\fantasia"
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "out_v2")
TOL_MS = 50.0
LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def parse_hea(path):
    """Minimal WFDB header parser: fs, signal names, formats."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    head = lines[0].split()
    n_sig = int(head[1])
    fs = float(head[2].split("/")[0]) if len(head) > 2 else np.nan
    names, fmts = [], []
    for ln in lines[1:1 + n_sig]:
        parts = ln.split()
        fmts.append(parts[1])
        names.append(parts[-1])
    return fs, names, fmts


def frames_available(rec_path, n_sig, fmt):
    """How many frames the (possibly truncated) .dat file really holds."""
    dat = rec_path + ".dat"
    if not os.path.exists(dat):
        return None, "missing .dat"
    b = os.path.getsize(dat)
    if fmt == "212":
        return int(b // (1.5 * n_sig)), None
    if fmt == "16":
        return int(b // (2 * n_sig)), None
    return None, f"unsupported format {fmt}"


def match(ref, det, tol):
    """Order-preserving one-to-one pairing within +-tol samples."""
    i = j = 0
    pairs = []
    missed = 0
    extra = 0
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            pairs.append((int(ref[i]), int(det[j]), int(d)))
            i += 1
            j += 1
        elif d > 0:            # detection is late -> this ref beat was missed
            missed += 1
            i += 1
        else:                  # detection is early -> extra detection
            extra += 1
            j += 1
    missed += len(ref) - i
    extra += len(det) - j
    return pairs, missed, extra


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        import wfdb
    except Exception as e:                       # pragma: no cover
        say("FATAL: wfdb not importable: %r" % (e,))
        say("Cannot read the WFDB annotation streams without it. No numbers "
            "reported.")
        with open(os.path.join(OUT, "fantasia_validate.log"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(LOG) + "\n")
        return

    say("wfdb version: %s" % getattr(wfdb, "__version__", "?"))
    recs = []
    recfile = os.path.join(FANTASIA, "RECORDS")
    if os.path.exists(recfile):
        with open(recfile, "r", encoding="utf-8", errors="replace") as fh:
            recs = [ln.strip() for ln in fh if ln.strip()]
    else:
        recs = sorted(os.path.basename(p)[:-4]
                      for p in glob.glob(os.path.join(FANTASIA, "*.hea")))
    say("records listed: %d" % len(recs))

    rows = []
    notes = []
    hists = {}
    for rec in recs:
        path = os.path.join(FANTASIA, rec)
        hea = path + ".hea"
        if not os.path.exists(hea):
            notes.append("%s: no .hea" % rec)
            continue
        try:
            fs, names, fmts = parse_hea(hea)
        except Exception as e:
            notes.append("%s: header parse failed (%r)" % (rec, e))
            continue
        n_sig = len(names)
        fmt = fmts[0]
        if len(set(fmts)) > 1:
            notes.append("%s: mixed formats %s; using first" % (rec, fmts))
        avail, why = frames_available(path, n_sig, fmt)
        if avail is None:
            rows.append({"record": rec, "fs": fs, "status": why,
                         "n_ref": np.nan, "n_det": np.nan,
                         "matched": np.nan, "missed": np.nan, "extra": np.nan,
                         "match_rate": np.nan, "precision": np.nan,
                         "f1": np.nan, "n_samples": np.nan,
                         "duration_covered_s": np.nan,
                         "n_ann_outside_covered": np.nan})
            notes.append("%s: %s -- detector NOT run (no waveform in local copy)"
                         % (rec, why))
            continue
        try:
            r = wfdb.rdrecord(path, physical=False, sampto=avail)
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()[-1]
            rows.append({"record": rec, "fs": fs, "status": "rdrecord failed: " + tb,
                         "n_ref": np.nan, "n_det": np.nan, "matched": np.nan,
                         "missed": np.nan, "extra": np.nan,
                         "match_rate": np.nan, "precision": np.nan, "f1": np.nan,
                         "n_samples": np.nan, "duration_covered_s": np.nan,
                         "n_ann_outside_covered": np.nan})
            notes.append("%s: rdrecord failed (%s)" % (rec, tb))
            continue
        ch = names.index("ECG") if "ECG" in names else (1 if n_sig > 1 else 0)
        sig = np.asarray(r.d_signal[:, ch], dtype=np.float64)
        n_samp = len(sig)
        try:
            ann = wfdb.rdann(path, "ecg")
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()[-1]
            rows.append({"record": rec, "fs": fs, "status": "rdann failed: " + tb,
                         "n_ref": np.nan, "n_det": np.nan, "matched": np.nan,
                         "missed": np.nan, "extra": np.nan, "match_rate": np.nan,
                         "precision": np.nan, "f1": np.nan,
                         "n_samples": n_samp,
                         "duration_covered_s": n_samp / fs,
                         "n_ann_outside_covered": np.nan})
            notes.append("%s: rdann failed (%s)" % (rec, tb))
            continue

        ref_all = np.asarray(ann.sample, dtype=np.int64)
        ref = ref_all[(ref_all >= 0) & (ref_all < n_samp)]
        det = pan_tompkins(sig, fs)
        tol = int(round(TOL_MS / 1000.0 * fs))
        pairs, missed, extra = match(ref, det, tol)
        nm = len(pairs)
        offs = np.array([p[2] for p in pairs], dtype=float)
        rows.append({
            "record": rec, "fs": fs, "status": "ok",
            "n_ref": len(ref), "n_det": len(det),
            "matched": nm, "missed": int(missed), "extra": int(extra),
            "match_rate": nm / len(ref) if len(ref) else np.nan,
            "precision": nm / len(det) if len(det) else np.nan,
            "f1": (2 * nm / (len(ref) + len(det))
                   if (len(ref) + len(det)) else np.nan),
            "n_samples": n_samp, "duration_covered_s": n_samp / fs,
            "n_ann_outside_covered": int(len(ref_all) - len(ref)),
            "tol_samples": tol,
            "median_offset_ms": float(np.median(offs) / fs * 1000) if nm else np.nan,
            "p95_abs_offset_ms": (float(np.percentile(np.abs(offs), 95) / fs * 1000)
                                  if nm else np.nan),
            "hr_ref_bpm": 60.0 * fs / float(np.mean(np.diff(ref))) if len(ref) > 2 else np.nan,
            "hr_det_bpm": 60.0 * fs / float(np.mean(np.diff(det))) if len(det) > 2 else np.nan,
        })
        say("%s fs=%5.0f %6.0f s | ref %4d det %4d | match %4d (%.4f) "
            "missed %3d extra %3d | HR ref %.1f det %.1f"
            % (rec, fs, n_samp / fs, len(ref), len(det), nm,
               nm / len(ref) if len(ref) else np.nan, missed, extra,
               rows[-1]["hr_ref_bpm"], rows[-1]["hr_det_bpm"]))
        bins = np.arange(0, n_samp / fs + 60.0, 60.0)
        hists[rec] = (bins, np.histogram(ref / fs, bins)[0],
                      np.histogram(det / fs, bins)[0])

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "fantasia_detector_validation.csv"), index=False)

    ok = t[t.status == "ok"]
    summary_rows = []
    if len(ok):
        tot_ref = int(ok.n_ref.sum())
        tot_det = int(ok.n_det.sum())
        tot_match = int(ok.matched.sum())
        agg = {
            "n_records_validated": len(ok),
            "n_records_no_waveform": int((t.status != "ok").sum()),
            "total_ref_beats": tot_ref,
            "total_det_beats": tot_det,
            "total_matched": tot_match,
            "pooled_match_rate": tot_match / tot_ref,
            "pooled_precision": tot_match / tot_det,
            "pooled_f1": 2 * tot_match / (tot_ref + tot_det),
            "total_missed": int(ok.missed.sum()),
            "total_extra": int(ok.extra.sum()),
            "per_record_match_rate_median": float(ok.match_rate.median()),
            "per_record_match_rate_q25": float(ok.match_rate.quantile(0.25)),
            "per_record_match_rate_q75": float(ok.match_rate.quantile(0.75)),
            "per_record_match_rate_min": float(ok.match_rate.min()),
            "per_record_match_rate_max": float(ok.match_rate.max()),
            "per_record_precision_median": float(ok.precision.median()),
            "per_record_precision_min": float(ok.precision.min()),
            "median_abs_offset_ms_median": float(ok.median_offset_ms.abs().median()),
            "p95_abs_offset_ms_median": float(ok.p95_abs_offset_ms.median()),
            "tolerance_ms": TOL_MS,
            "covered_duration_s_total": float(ok.duration_covered_s.sum()),
        }
        summary_rows.append(agg)
        say("")
        say("== pooled over validated records ==")
        for k, v in agg.items():
            say("  %-34s %s" % (k, ("%.6f" % v) if isinstance(v, float) else v))
        say("")
        say("== agreement distribution across records (match rate) ==")
        for q in [0, 10, 25, 50, 75, 90, 100]:
            say("  p%-3d %.4f" % (q, float(np.percentile(ok.match_rate, q))))
    S = pd.DataFrame(summary_rows)
    S.to_csv(os.path.join(OUT, "fantasia_detector_summary.csv"), index=False)

    # ---- figure: agreement per record + the lock-out signature -----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ok = t[t.status == "ok"].sort_values("match_rate")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                                 gridspec_kw={"width_ratios": [1.05, 1.25]})
        ax = axes[0]
        cols = ["crimson" if v < 0.9 else "steelblue" for v in ok.match_rate]
        ax.barh(range(len(ok)), ok.match_rate * 100, color=cols)
        ax.set_yticks(range(len(ok)),
                      [f"{r} (fs={f:.0f})" for r, f in zip(ok.record, ok.fs)],
                      fontsize=7)
        ax.axvline(90, color="gray", ls="--", lw=1)
        ax.set_xlabel("matched / reference beats (%)")
        ax.set_title("per-record agreement (red < 90%)", fontsize=10)
        ax = axes[1]
        for rec in ["f2o05", "f2y05"]:
            if rec in hists:
                bins, hr, hd = hists[rec]
                c = bins[:-1] + 30
                ax.plot(c, hr, "-", lw=1.6, label=f"{rec} reference")
                ax.plot(c, hd, "--", lw=1.6, label=f"{rec} detected")
        ax.set_xlabel("time in covered prefix (s)")
        ax.set_ylabel("beats per 60 s")
        ax.set_title("threshold lock-out signature: detections stop mid-record",
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.suptitle("revision C — detector vs Fantasia reference annotations "
                     "(+-50 ms)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_fantasia_validation.png"), dpi=150)
        plt.close(fig)
        say("")
        say("wrote fig_fantasia_validation.png")
    except Exception as e:                       # pragma: no cover
        say("figure generation failed: %r" % (e,))

    say("")
    say("== records where the detector could not be run ==")
    for rec in t.loc[t.status != "ok", "record"].tolist():
        say("  " + rec)
    say("")
    say("== notes ==")
    for n in notes:
        say("  " + n)
    with open(os.path.join(OUT, "fantasia_notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("Fantasia local copy: %s\n" % FANTASIA)
        fh.write("ANNOTATORS file content: 'corrected beat annotations from an "
                 "automated detector'\n")
        fh.write("=> machine annotations with correction, NOT independent human "
                 "expert labels.\n\n")
        fh.write("\n".join(notes) + "\n")
    with open(os.path.join(OUT, "fantasia_validate.log"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()