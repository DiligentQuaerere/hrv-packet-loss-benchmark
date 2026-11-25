"""Pan-Tompkins R-peak detector (offline, unified for full and masked signals).

Pipeline: 5-15 Hz Butterworth bandpass -> 5-point derivative -> squaring ->
150 ms moving-window integration -> adaptive dual threshold with searchback
and T-wave discrimination -> refine location on |bandpassed| signal.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def pan_tompkins(x, fs):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < int(2 * fs):
        return np.array([], dtype=np.int64)

    b, a = butter(2, [5.0, 15.0], btype="band", fs=fs)
    bp = filtfilt(b, a, x)
    deriv = np.convolve(bp, np.array([1.0, 2.0, 0.0, -2.0, -1.0]) / 8.0, mode="same")
    sq = deriv * deriv
    win = int(round(0.150 * fs))
    mwi = np.convolve(sq, np.ones(win) / win, mode="same")

    # 0.25 s refractory: below any plausible RR here; kills 0.2 s double counts
    refr = int(round(0.250 * fs))
    cands, _ = find_peaks(mwi, distance=refr)
    if len(cands) == 0:
        return np.array([], dtype=np.int64)
    heights = mwi[cands]

    n_init = max(4, int(np.searchsorted(cands, 2 * fs)))
    init = heights[:n_init]
    spki = float(np.max(init))
    npki = 0.5 * float(np.median(init))
    thr1 = npki + 0.25 * (spki - npki)
    thr2 = 0.5 * thr1

    slope_win = int(round(0.075 * fs))

    def slope_at(idx):
        lo = max(0, idx - slope_win)
        hi = min(n, idx + slope_win + 1)
        return float(np.max(np.abs(deriv[lo:hi])))

    status = np.zeros(len(cands), dtype=np.int8)  # 0 unknown, 1 QRS, -1 noise
    accepted = []
    rr_hist = []
    rr_avg = float(fs)  # prior: 1 s

    def accept(j):
        nonlocal spki, thr1, thr2, rr_avg
        status[j] = 1
        accepted.append(int(cands[j]))
        spki = 0.125 * heights[j] + 0.875 * spki
        thr1 = npki + 0.25 * (spki - npki)
        thr2 = 0.5 * thr1
        if len(accepted) >= 2:
            rr_hist.append(accepted[-1] - accepted[-2])
            rr_hist[:] = rr_hist[-8:]
            rr_avg = float(np.mean(rr_hist))

    def is_twave(j, prev_idx):
        if cands[j] - prev_idx >= int(round(0.360 * fs)):
            return False
        return slope_at(cands[j]) < 0.5 * slope_at(prev_idx)

    i = 0
    while i < len(cands):
        if status[i] != 0:
            i += 1
            continue
        c = cands[i]
        h = heights[i]
        last = accepted[-1] if accepted else None
        if last is not None and c - last < refr:
            status[i] = -1
            i += 1
            continue
        if h > thr1 and not (last is not None and is_twave(i, last)):
            accept(i)
        else:
            status[i] = -1
            npki = 0.125 * h + 0.875 * npki
            thr1 = npki + 0.25 * (spki - npki)
            thr2 = 0.5 * thr1
        # searchback: long pause since last QRS -> look for missed beat above thr2
        last = accepted[-1] if accepted else None
        if last is not None and c - last > 1.66 * rr_avg:
            lo = last + refr
            hi = c - refr
            eligible = [
                j for j in range(i)
                if status[j] == -1 and lo < cands[j] < hi and heights[j] > thr2
            ]
            if eligible:
                j = max(eligible, key=lambda k: heights[k])
                accept(j)
                # re-evaluate current candidate against the newly accepted beat
                last = accepted[-1]
                if c - last < refr:
                    status[i] = -1
                elif h > thr1 and not is_twave(i, last):
                    accept(i)
                else:
                    status[i] = -1
                    npki = 0.125 * h + 0.875 * npki
                    thr1 = npki + 0.25 * (spki - npki)
                    thr2 = 0.5 * thr1
        i += 1

    # refine each accepted MWI peak to the local |bandpassed| maximum,
    # then snap to the dominant raw deflection within +-30 ms so biphasic
    # QRS complexes get a consistent fiducial point
    peaks = []
    half = win // 2
    snap = int(round(0.030 * fs))
    for c in accepted:
        lo = max(0, c - half)
        hi = min(n, c + half + 1)
        p = lo + int(np.argmax(np.abs(bp[lo:hi])))
        lo2 = max(0, p - snap)
        hi2 = min(n, p + snap + 1)
        peaks.append(lo2 + int(np.argmax(np.abs(x[lo2:hi2]))))
    peaks = np.array(sorted(set(peaks)), dtype=np.int64)
    # enforce refractory on refined locations
    keep = []
    for p in peaks:
        if not keep or p - keep[-1] >= refr:
            keep.append(int(p))
    return np.array(keep, dtype=np.int64)