"""Cached metadata and algebra only; no fitting or raw labels."""
from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
CACHE=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1'
bounds=[]
for p in sorted(CACHE.glob('*_r3_*.json')):
    m=json.loads(p.read_text(encoding='utf-8'))
    lo,hi=m['provenance']['train_target_bounds']
    assert math.isfinite(lo) and math.isfinite(hi) and lo>0 and hi>=lo
    bounds.append((lo,hi))
assert len(bounds)==66
# bq remains inside training EC range, yet normalized leaf output may leave it.
y_train=np.array([1.,4.]); b_train=np.array([1.,1.]); b_query=4.
ratio=y_train/b_train
leaf_arithmetic=float(np.mean(ratio)); raw=b_query*leaf_arithmetic
assert raw==10. and raw>float(y_train.max())
baseline=2.;old_et=2.;weight=.48
clip_first=float(np.clip(baseline+weight*(np.clip(raw,1.,4.)-old_et),1.,4.))
clip_last=float(np.clip(baseline+weight*(raw-old_et),1.,4.))
assert abs(clip_first-2.96)<1e-14 and clip_last==4.
result=dict(status='PASS',scope='66 public cache metadata bounds and synthetic arithmetic only.',metadata_cells=len(bounds),minimum_saved_train_lower_bound=min(x[0] for x in bounds),maximum_saved_train_upper_bound=max(x[1] for x in bounds),all_saved_training_ranges_strictly_positive=True,synthetic_clip_counterexample=dict(y_train=y_train.tolist(),b_train=b_train.tolist(),b_query=b_query,leaf_ratio_mean=leaf_arithmetic,raw=raw,clip_component_first=clip_first,clip_final_only=clip_last))
(OUT/'log_ratio_bounds_check_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
