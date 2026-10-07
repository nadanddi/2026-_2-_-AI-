"""Synthetic API boundary cases, never a performance or actual BLK layout change."""
import json,copy,warnings
from blk_sg2_refonly_v2 import *
from blk_context_v1 import BLKContext,RAW as ALL_RAW

layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
ctx=BLKContext(layout);m=RefOnlySG2(ctx)
checks=[]
rid='F13_000_00'
first=next(r for r in ctx.query_ids if key(r)[0]=='F13')
ctx.query_ids.add(rid);ctx._query[rid]=dict(ctx._query[first])
try:
    out,diag=m.predict_one(rid,{rid:.7},'BLK_QUERY_ROLE')
    assert np.isfinite(out)
    assert diag.get('query_calendar',0)==0
    checks.append('synthetic first record: no negative-index wrap to later record')
finally:
    ctx.query_ids.remove(rid);del ctx._query[rid]
rid=min(r for r in ctx.query_ids if key(r)[2]==5)
f,d,h=key(rid);prefix={f'{f}_{d:03d}_{j:02d}':.7 for j in range(h+1)}
obs={j:{c:None for c in ALL_RAW} for j in range(h+1)}
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning)
    assert m.twin(f,d,h,obs) is None
checks.append('all-missing weather does not yield a false exact twin')
saved=m.fit_signature[f,h]
try:
    m.fit_signature[f,h]=(saved[0],np.full(len(saved[1]),np.nan))
    out,diag=m.predict_one(rid,prefix,'BLK_QUERY_ROLE')
    assert out==.7 and not diag['has_candidate']
    checks.append('no finite standardized signature falls back to baseline')
finally:m.fit_signature[f,h]=saved
try:m.predict_one(rid,{rid:.7},'BLK_QUERY_ROLE')
except AssertionError:checks.append('incomplete baseline prediction prefix rejected')
else:raise AssertionError('missing prefix accepted')
try:m.predict_one(rid,{**prefix,rid:float('nan')},'BLK_QUERY_ROLE')
except AssertionError:checks.append('nonfinite baseline prediction rejected')
else:raise AssertionError('NaN accepted')
out=HERE/'BLK_SG2_edge_audit_v1.json';assert not out.exists()
out.write_text(json.dumps({'status':'PASS','synthetic_checks':checks,'actual_layout_changed':False,
    'heldout_truth_loaded':False,'performance_evaluated':False,'code_sha256':{n:sha(HERE/n) for n in ['blk_sg2_refonly_v1.py','blk_sg2_refonly_v2.py']}},ensure_ascii=False,indent=2),encoding='utf-8')
print('SG2 synthetic boundary cases PASS',flush=True)
