"""Univariate time-series anomaly detection against a seasonal baseline.

Each point is compared with the median + MAD (median absolute deviation) of
the SAME hour over the previous days, so daily seasonality does not inflate
the baseline and the anomalies themselves cannot either. Evaluates precision
and recall against injected ground truth.
"""
import math
import random


def seasonal_baseline(x, period=24, days=3):
    """Median and MAD of the values at the same phase on previous days."""
    n = len(x)
    med = [None] * n
    mad = [None] * n
    for i in range(n):
        hist = [x[i - j * period] for j in range(1, days + 1) if i - j * period >= 0]
        if len(hist) < 2:
            continue
        hist.sort()
        m = hist[len(hist) // 2]
        dev = sorted(abs(v - m) for v in hist)
        med[i] = m
        mad[i] = dev[len(dev) // 2] or 1e-9
    return med, mad


def detect(x, period=24, days=5, threshold=4.0):
    """Flag points whose robust z-score vs the seasonal baseline exceeds threshold.

    The MAD of a few historical samples is a noisy scale estimate, so it is
    floored at the median MAD across the series; otherwise ordinary noise
    would be flagged whenever the local MAD happens to come out tiny.
    """
    med, mad = seasonal_baseline(x, period, days)
    valid = sorted(m for m in mad if m is not None)
    floor = valid[len(valid) // 2] if valid else 1e-9
    return [
        med[i] is not None and abs(v - med[i]) / max(mad[i], floor) > threshold
        for i, v in enumerate(x)
    ]


def make_series(n=24 * 14, seed=3, n_anoms=8):
    """Two weeks of hourly traffic: daily seasonality + noise + spikes."""
    rnd = random.Random(seed)
    x = [100 + 40 * math.sin(2 * math.pi * (t % 24) / 24) + rnd.gauss(0, 4)
         for t in range(n)]
    idx = sorted(rnd.sample(range(60, n - 10), n_anoms))
    for i in idx:
        x[i] += rnd.choice([-1, 1]) * rnd.uniform(45, 80)
    return x, set(idx)


def score(flags, truth):
    pred = {i for i, f in enumerate(flags) if f}
    # tolerant matching: a detection within +-2h of an injected spike counts
    hits = sum(1 for i in pred if any(abs(i - t) <= 2 for t in truth))
    covered = sum(1 for t in truth if any(abs(i - t) <= 2 for i in pred))
    precision = hits / max(len(pred), 1)
    recall = covered / max(len(truth), 1)
    return precision, recall, len(pred)


def main():
    x, truth = make_series()
    print(f"series: {len(x)} hourly points, {len(truth)} injected anomalies\n")
    for threshold in (3.0, 4.0, 5.0):
        flags = detect(x, threshold=threshold)
        p, r, n_pred = score(flags, truth)
        print(f"threshold={threshold}: flagged {n_pred:2d} points -> "
              f"precision {p:.2f}, recall {r:.2f}")
    flags = detect(x)
    print("\nDetected timestamps (hour index) at threshold=4.0:",
          sorted(i for i, f in enumerate(flags) if f))
    print("Injected timestamps:                    ", sorted(truth))


if __name__ == "__main__":
    main()
