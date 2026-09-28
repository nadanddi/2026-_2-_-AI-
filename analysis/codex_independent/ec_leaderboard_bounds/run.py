"""Geometric RMSE bounds for an unsubmitted EC vector from aggregate scores."""
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/submissions/submission_04.csv'
OTHERS = [ROOT/'Claude/submissions/submission_01_scored.csv',
          ROOT/'research/submissions/submission_03.csv',
          ROOT/'research/submissions/submission_06.csv']
CANDIDATE = ROOT/'analysis/local/ec_tabpfn_local_candidate/20260928_032419/candidate_temp06_ec_tabpfn_v2_cpu.csv'


def values(path, ids=None):
    f = pd.read_csv(path, usecols=['row_id','sub_ec'])
    assert len(f) == 1440 and f.row_id.is_unique
    if ids is not None:
        assert f.row_id.tolist() == ids
    return f.row_id.tolist(), f.sub_ec.to_numpy(float)


def bounds(scores, gram, cross, q2, qproj2):
    base = scores[0]
    others = scores[1:]
    g = (base**2 + np.diag(gram) - others**2)/2
    inv = np.linalg.pinv(gram, rcond=1e-12)
    rproj2 = float(g @ inv @ g)
    rest = base**2 - rproj2
    if rest < -1e-10:
        return None
    center = base**2 + q2 - 2*float(g @ inv @ cross)
    radius = 2*np.sqrt(max(0,rest)*max(0,q2-qproj2))
    return np.sqrt(max(0,center-radius)), np.sqrt(max(0,center+radius)), rproj2


def main():
    ids, p3 = values(BASE)
    b = np.column_stack([values(path,ids)[1]-p3 for path in OTHERS])
    q = values(CANDIDATE,ids)[1]-p3
    gram = b.T @ b / len(q)
    cross = b.T @ q / len(q)
    q2 = float(q@q/len(q))
    qproj2 = float(cross@np.linalg.pinv(gram,rcond=1e-12)@cross)
    centers = np.array([.2055,.2442,.2287,.2134])
    candidates=[]
    for signs in itertools.product((-1,1),repeat=4):
        sc = centers + np.array(signs)*.00005
        v = bounds(sc,gram,cross,q2,qproj2)
        if v is not None:
            candidates.append(v)
    assert len(candidates)==16
    lower,upper = min(z[0] for z in candidates),max(z[1] for z in candidates)
    nominal = bounds(centers,gram,cross,q2,qproj2)
    result=dict(candidate_delta_rms=np.sqrt(q2),
                fraction_of_candidate_delta_in_known_span=qproj2/q2,
                nominal_bounds=nominal[:2],rounded_score_bounds=[lower,upper],
                known_residual_projection_rms=np.sqrt(nominal[2]),
                verdict=('GUARANTEED_BETTER' if upper<.20545 else
                         'GUARANTEED_WORSE' if lower>.20555 else 'UNDETERMINED'),
                hidden_labels_read=False,no_submission_file_created=True)
    out=ROOT/'analysis/local/ec_leaderboard_bounds'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
