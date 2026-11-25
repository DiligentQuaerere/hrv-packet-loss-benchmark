"""HRV metrics. Frequency pipeline (fixed): RR tachogram -> cubic-spline
resample at 4 Hz -> subtract mean -> Welch PSD (Hann, nperseg=min(256, N),
50% overlap) -> trapezoid band integration with interpolated band edges.
LF 0.04-0.15 Hz, HF 0.15-0.40 Hz, TP = LF + HF (0.04-0.40 Hz).
Powers in ms^2, time-domain in ms (pNN50 in %).

Duration gate: records shorter than FREQ_MIN_DURATION get NaN frequency
metrics. The original protocol used 120 s; on 2026-11-25 the user approved
relaxing it to 60 s because all EEGMAT stress segments (_2) are 62 s.
Truth and estimates use the identical pipeline, so relative errors remain
internally consistent; absolute LF values on 62 s records are noisy.
"""
import numpy as np
from scipy.signal import welch
from scipy.interpolate import CubicSpline

FREQ_MIN_DURATION = 60.0
TIME_METRICS = ["SDNN", "RMSSD", "pNN50"]
FREQ_METRICS = ["LF", "HF", "LF_HF", "LFnu", "HFnu", "TP"]
ALL_METRICS = TIME_METRICS + FREQ_METRICS


def time_domain(peaks, fs):
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / fs
    if len(rr) < 2:
        return {k: np.nan for k in TIME_METRICS}
    d = np.diff(rr)
    return {
        "SDNN": float(np.std(rr, ddof=1) * 1000.0),
        "RMSSD": float(np.sqrt(np.mean(d * d)) * 1000.0),
        "pNN50": float(np.mean(np.abs(d) > 0.05) * 100.0),
    }


def _band_power(f, p, lo, hi):
    inside = (f > lo) & (f < hi)
    fb = np.concatenate(([lo], f[inside], [hi]))
    pb = np.concatenate(([np.interp(lo, f, p)], p[inside], [np.interp(hi, f, p)]))
    return float(np.trapezoid(pb, fb))


def freq_domain(peaks, fs, duration_s):
    nan = {k: np.nan for k in FREQ_METRICS}
    peaks = np.asarray(peaks, dtype=np.float64)
    if duration_s < FREQ_MIN_DURATION or len(peaks) < 4:
        return nan
    rr = np.diff(peaks) / fs
    t = peaks[1:] / fs
    cs = CubicSpline(t, rr)
    n = int((t[-1] - t[0]) * 4.0) + 1
    if n < 16:
        return nan
    x = cs(t[0] + np.arange(n) / 4.0)
    # physiological guard against spline overshoot at unfilled/long gaps;
    # applied identically to truth and estimates
    x = np.clip(x, 0.3, 2.0)
    x = x - np.mean(x)
    nper = min(256, len(x))
    f, p = welch(
        x, fs=4.0, window="hann", nperseg=nper,
        noverlap=nper // 2, detrend=False, scaling="density",
    )
    lf = _band_power(f, p, 0.04, 0.15) * 1e6
    hf = _band_power(f, p, 0.15, 0.40) * 1e6
    tp = lf + hf
    return {
        "LF": lf,
        "HF": hf,
        "LF_HF": lf / hf if hf > 0 else np.nan,
        "LFnu": lf / tp * 100.0 if tp > 0 else np.nan,
        "HFnu": hf / tp * 100.0 if tp > 0 else np.nan,
        "TP": tp,
    }


def all_metrics(peaks, fs, duration_s):
    out = time_domain(peaks, fs)
    out.update(freq_domain(peaks, fs, duration_s))
    return out