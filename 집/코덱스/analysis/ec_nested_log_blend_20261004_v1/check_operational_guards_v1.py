"""Synthetic provenance/partial-first-cell checks; no model fit or real labels."""
from pathlib import Path
import importlib.util, json
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
spec = importlib.util.spec_from_file_location('nested_log_guard_test', H/'run_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
import pandas as pd

records = []
frame = pd.DataFrame({'a':[1.,np.nan], 'season':[2.,3.]})
original = m.features_sha(frame,['a','season'])
for name, changed in [('raw_input',frame.assign(a=[5.,np.nan])),
                       ('season',frame.assign(season=[7.,3.]))]:
    assert m.features_sha(changed,['a','season']) != original
    records.append(dict(case=name,status='PASS'))
assert m.features_sha(frame.copy(),['a','season']) == original
assert m.features_sha(frame,['season','a']) != original
records.append(dict(case='copy_and_column_order',status='PASS'))
toy = ROOT/'집/코덱스/local/ec_nested_log_blend_guard_synthetic_20261004_v1'
assert toy.resolve().is_relative_to((ROOT/'집/코덱스/local').resolve())
toy.mkdir(exist_ok=False)
m.H = toy
d,meta,inner = toy/'cell.csv',toy/'cell.json',toy/'cell_inner.npz'
signature = {'source':'synthetic','ids':'synthetic'}
m.first_artifact_guard(d,meta,inner,signature)
records.append(dict(case='empty_first_cell',status='PASS'))
d.write_text('synthetic',encoding='utf-8')
try:
    m.first_artifact_guard(d,meta,inner,signature)
    raise RuntimeError('partial first cell accepted')
except AssertionError:
    records.append(dict(case='partial_first_cell_rejected',status='PASS'))
meta.write_text('synthetic',encoding='utf-8')
inner.write_text('synthetic',encoding='utf-8')
first = toy/'first_fold_verification_v2.json'
first.write_text(json.dumps({'status':'PASS','signature':signature}),encoding='utf-8')
m.first_artifact_guard(d,meta,inner,signature)
records.append(dict(case='four_first_artifacts',status='PASS'))
try:
    m.first_artifact_guard(d,meta,inner,{'source':'different','ids':'synthetic'})
    raise RuntimeError('changed signature accepted')
except AssertionError:
    records.append(dict(case='changed_signature_rejected',status='PASS'))
dest = H/'operational_guard_check_v1.json'
assert not dest.exists()
dest.write_text(json.dumps(dict(status='PASS',run_sha256=m.S.sha(H/'run_v2.py'),
                               cases=records,model_fit=0,scope='synthetic operational guards only'),
                           ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
