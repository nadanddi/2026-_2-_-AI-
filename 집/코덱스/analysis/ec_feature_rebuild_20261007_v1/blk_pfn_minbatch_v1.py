"""Numerical inference experiment: pad small query batches using their own last row.

No fitting, labels, new features, or other query rows enter this adapter.
This is experimental until all frozen-context audits pass.
"""
import numpy as np


def predict_minbatch8(model, rows):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError('Expected a nonempty two-dimensional query matrix')
    count = len(rows)
    if count < 8:
        rows = np.concatenate([rows, np.repeat(rows[-1:], 8-count, axis=0)], axis=0)
    prediction = np.asarray(model.predict(rows), dtype=float)
    if prediction.shape != (len(rows),) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape or nonfinite output')
    return prediction[:count].copy()
