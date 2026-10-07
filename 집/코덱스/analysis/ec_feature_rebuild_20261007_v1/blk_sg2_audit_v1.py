"""Independent SG2 source comparison with reference-only weather normalization."""
import ast,json
from blk_sg2_refonly_v1 import *
from blk_context_v1 import BLKContext
from checkpoint_v1 import digest

layout=json.loads((HERE/'BLK_layout_v2.json').read_text(encoding='utf-8'))
ctx=BLKContext(layout)
m=RefOnlySG2(ctx)
ids=sorted(ctx.query_ids)
queries={}
for f in ['F13','F47']:
    queries.update(ctx.query_prefix(max(r for r in ids if key(r)[0]==f)))
q=dataframe(queries)
S=sgns['prepare'](pd.concat([m.x,q],ignore_index=True))
# Independently replace the unfit query+train weather pivot with fixed reference scaling.
raw=sgns['_ident'](pd.concat([m.x,q],ignore_index=True)).pivot_table(index=['farm','day'],columns='hour',values=W)
for f in ['F13','F47']:
    rows=raw.index.get_level_values(0)==f
    for c in W:raw.loc[rows,c]=(raw.loc[rows,c].values-m.mu[f,c])/m.sd[f,c]
S['WV']=raw
source=ast.parse(SG2_SOURCE.read_text(encoding='utf-8-sig'))
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='correct')
# The ONLY source-rule modification is the explicit query-role activation gate.
for n in ast.walk(fn):
    if isinstance(n,ast.If) and ast.unparse(n.test)=='d < 179':n.test=ast.Constant(False)
ns=dict(np=np,pd=pd,_ident=sgns['_ident'])
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SG2_SOURCE),'exec'),ns)
baseline=np.full(len(ids),.7)
ec=pd.Series(m.ec)
expected=ns['correct'](pd.DataFrame({'row_id':ids}),baseline,S,ec,m.ref,m.cal)
actual=[];diagnostics=[]
for rid in ids:
    f,d,h=key(rid)
    bp={f'{f}_{d:03d}_{i:02d}':.7 for i in range(h+1)}
    pred,diag=m.predict_one(rid,bp,'BLK_QUERY_ROLE')
    actual.append(pred);diagnostics.append(diag)
    rawpass,_=m.predict_one(rid,bp,'BLK_RAW_PASS')
    assert rawpass==.7
error=float(np.max(np.abs(expected-np.array(actual))))
assert error<=1e-10,(error,'source equivalence failed')
probes=[f'{b["farm"]}_{b["query_days"][i]:03d}_{h:02d}' for b in layout['blocks'] for i in [0,len(b['query_days'])-1] for h in [0,12,23]]
for rid in probes:
    f,d,h=key(rid);bp={f'{f}_{d:03d}_{i:02d}':.7 for i in range(h+1)}
    before=m.predict_one(rid,bp,'BLK_QUERY_ROLE')
    future=[r for r in ctx._query if key(r)[0]!=f or key(r)[1:]>(d,h)]
    saved={r:ctx._query[r] for r in future}
    try:
        for r in future:ctx._query[r]={c:99999.0 for c in ctx._query[r]}
        assert before==m.predict_one(rid,bp,'BLK_QUERY_ROLE')
    finally:ctx._query.update(saved)
out=HERE/'BLK_SG2_refonly_audit_v1.json';assert not out.exists()
out.write_text(json.dumps({'status':'PASS','reference_fit_rows':len(ctx.train_ids),'query_rows':len(ids),
    'independent_source_max_difference':error,'future_other_farm_checks':len(probes),
    'query_role_activation':sum(d['active'] for d in diagnostics),'constant_baseline_changed':sum(d['changed'] for d in diagnostics),
    'constant_baseline_reference_available':sum(d['has_candidate'] for d in diagnostics),'RAW_PASS_active':0,
    'heldout_truth_loaded':False,'performance_evaluated':False,'source_sha256':sha(SG2_SOURCE),
    'adapter_sha256':sha(HERE/'blk_sg2_refonly_v1.py'),'reference_calendar_sha256':digest(sorted((f,d,c) for (f,d),c in m.cal.items())),
    'limits':['constant baseline audit only; real CPU PFN/R3 final output checks pending','ref-only fit policy differs from naïve BLK query+train prepare','explicit no-finite/first-record fallback; original source may fail those edge cases']},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'SG2 reference-only/source equivalence PASS maxdiff={error}',flush=True)
