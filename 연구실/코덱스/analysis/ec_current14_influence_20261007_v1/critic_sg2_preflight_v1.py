"""Independent SG2 mechanical tests; no model fitting or forbidden data reads."""
from pathlib import Path
import runpy, sys, json
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
sys.argv = [str(H/'test_sg2_reference_v2.py'), str(H/'sg2_ref_v2.py')]
d = runpy.run_path(sys.argv[0])
m, old, raw, ref = (d[n] for n in ['m', 'old', 'raw', 'ref'])
np, pd = d['np'], d['pd']
allref = d['allref']; a, b = d['a'], d['b']
ca, cb = m.ref_calendar(a, allref), old.ref_calendar(b, allref)
assert ca == cb
ec = pd.Series(np.linspace(.5,.9,len(allref)), index=pd.MultiIndex.from_tuples(sorted(allref)))
q = d['q']; p = d['p']
pa, trace = m.correct(q,p,a,ec,allref,ca,return_trace=True)
pb = old.correct(q,p,b,ec,allref,cb)
np.testing.assert_array_equal(pa,pb)
base_ec = pd.Series([.8]*len(ref), index=pd.MultiIndex.from_tuples(sorted(ref)))
_, original_trace = m.correct(q,p,d['base'],base_ec,ref,m.ref_calendar(d['base'],ref),return_trace=True)
tested = [r for r in original_trace if int(r['row_id'][4:7]) == 179 and int(r['row_id'][8:10]) <= 5]
assert len(tested)==12 and all(r['gate'] and abs(r['delta'])>0 for r in tested)
receipt = dict(status='PASS_TOY_PREPARE_CALENDAR_CORRECT_EQUIVALENCE_AND_ACTIVE_PREFIX',
    fullref_query_rows=len(q), exact_output_maxdiff=float(np.max(abs(pa-pb))),
    original_tested_prefix_rows=12, original_tested_active_rows=len(tested),
    fit=0, limitation='Synthetic inputs only; no claim of full real-data pathway coverage.')
with (H/'critic_sg2_preflight_v1.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(receipt,ensure_ascii=False))
