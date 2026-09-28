"""Exact convex-blend leaderboard RMSE intervals from prior aggregate scores."""
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FILES = {
    'round1': ROOT / 'Claude/submissions/submission_01_scored.csv',
    'round2': ROOT / 'research/submissions/submission_03.csv',
    'round3': ROOT / 'research/submissions/submission_04.csv',
    'round5': ROOT / 'research/submissions/submission_06.csv',
}
SCORES = np.array([.2442, .2287, .2055, .2134])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = [pd.read_csv(path, usecols=['row_id', 'sub_ec']) for path in FILES.values()]
    ids = data[0].row_id.tolist()
    assert len(ids) == 1440 and len(set(ids)) == 1440
    assert all(z.row_id.tolist() == ids for z in data)
    p = np.column_stack([z.sub_ec.to_numpy(float) for z in data])
    assert np.isfinite(p).all()
    d = np.mean((p[:, :, None] - p[:, None, :])**2, axis=0)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    # Nonnegative weights on the closed 0.05 simplex grid.
    rows = []
    lo = np.maximum(0, SCORES - .00005)**2
    hi = (SCORES + .00005)**2
    for a in range(21):
        for b in range(21-a):
            for c in range(21-a-b):
                w = np.array([a,b,c,20-a-b-c], float)/20
                penalty = sum(w[i]*w[j]*d[i,j] for i,j in itertools.combinations(range(4),2))
                center = float(np.sqrt(max(0,np.dot(w,SCORES**2)-penalty)))
                lower = float(np.sqrt(max(0,np.dot(w,lo)-penalty)))
                upper = float(np.sqrt(max(0,np.dot(w,hi)-penalty)))
                rows.append((center, lower, upper, w))
    best = min(rows, key=lambda z: z[2])
    best_exact = min(rows, key=lambda z: z[0])
    # The best published baseline itself could be as low as 0.20545.
    passes = best[2] <= .20545 - .001
    result = dict(formula='sum(w_i*RMSE_i^2)-sum_{i<j}(w_i*w_j*D_ij)',
                  files={key:dict(path=str(path),sha256=sha(path)) for key,path in FILES.items()},
                  scores_displayed=dict(zip(FILES,SCORES.tolist())),
                  pairwise_prediction_mse=d.tolist(), grid_count=len(rows),
                  best_worst_case=dict(weights=best[3].tolist(),center=best[0],
                                       lower=best[1],upper=best[2]),
                  best_center=dict(weights=best_exact[3].tolist(),center=best_exact[0],
                                   lower=best_exact[1],upper=best_exact[2]),
                  best_existing_round3_interval=[.20545,.20555],
                  passes_screen=passes, hidden_labels_read=False,
                  no_submission_file_created=True)
    out = ROOT / 'analysis/local/ec_leaderboard_blend' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
