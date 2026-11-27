# ts-anomaly-detector

Univariate time-series anomaly detection using a seasonal robust z-score: each point is compared with the median and MAD (median absolute deviation) of the *same hour* on previous days. Daily seasonality does not inflate the baseline, and the anomalies themselves cannot either (that is why median/MAD instead of mean/std).

One non-obvious detail: the MAD of a handful of historical samples is a noisy scale estimate, so it is floored at the median MAD across the whole series. Without the floor, ordinary noise gets flagged whenever the local MAD comes out tiny - in early experiments that produced 90+ false alarms on a 336-point series.

## Files
- `detector.py` - `seasonal_baseline()`, `detect()`, synthetic series generator with injected spikes (fixed seed), precision/recall scoring with +-2h tolerant matching
- `tests/` - 4 pytest cases

## Quickstart
```
python detector.py
pytest tests -q
```

## Real output (from `python detector.py` on this machine)

```
series: 336 hourly points, 8 injected anomalies

threshold=3.0: flagged 36 points -> precision 0.22, recall 1.00
threshold=4.0: flagged 18 points -> precision 0.44, recall 1.00
threshold=5.0: flagged 12 points -> precision 0.67, recall 1.00

Detected timestamps (hour index) at threshold=4.0: [57, 81, 82, 102, 149, 153, 180, 184, 186, 187, 188, 239, 248, 252, 263, 274, 316, 325]
Injected timestamps:                     [102, 180, 239, 248, 252, 263, 274, 316]
```

All 8 injected spikes are recovered at every threshold; precision rises as the threshold tightens. The remaining false positives (e.g. hours 184-188) are clusters where several consecutive noisy points drift together - a real behavior worth knowing about.

## Limits
- Synthetic data with known ground truth; real series need the period tuned (`period=24` assumes hourly data with a daily cycle).
- Univariate only, no trend handling - a strong trend would need differencing first.
