"""Exercise lazy context without constructor/competition CSV access."""
from pathlib import Path
import json,hashlib
from types import MappingProxyType
from ch2_query_context_v2 import CH2QueryContext,RAW

HERE=Path(__file__).resolve().parent
class ExplodesOnStrip:
    def strip(self):raise RuntimeError('forbidden numeric access occurred')

def fixture():
    ctx=object.__new__(CH2QueryContext)
    ctx.blocks=({'farm':'F13','query_days':(14,15)}, {'farm':'F13','query_days':(44,45)}, {'farm':'F47','query_days':(14,15)})
    ctx.query_to_block={f'{b["farm"]}_{d:03d}_{h:02d}':i for i,b in enumerate(ctx.blocks) for d in b['query_days'] for h in range(24)}
    ctx.query_ids=set(ctx.query_to_block);ctx._query={rid:{c:'1.5' for c in RAW} for rid in ctx.query_ids};ctx.consumed=[]
    return ctx

def main():
    checks=[]
    def check(name):checks.append(name)
    ctx=fixture();packet=ctx.query_prefix('F13_015_02')
    assert len(packet)==27 and set(packet)=={f'F13_014_{h:02d}' for h in range(24)}|{f'F13_015_{h:02d}' for h in range(3)}
    assert all(v==1.5 for row in packet.values() for v in row.values());check('exact_complete_prefix_numeric_values')
    for target in ['F13_015_02','F13_044_00','F47_015_02']:
        ctx=fixture();allowed=set(ctx.query_prefix(target));ctx.consumed=[]
        for rid in set(ctx.query_ids)-allowed:ctx._query[rid]={c:ExplodesOnStrip() for c in RAW}
        assert set(ctx.query_prefix(target))==allowed;check(f'{target}_all_forbidden_values_never_accessed')
    for violation in ['bad_schema','missing_row']:
        ctx=fixture();ctx._query['F13_014_00']={c:ExplodesOnStrip() for c in RAW}
        if violation=='bad_schema':ctx._query['F13_015_02']['sub_ec']=ExplodesOnStrip()
        else:del ctx._query['F13_015_02']
        rejected=False
        try:ctx.query_prefix('F13_015_02')
        except AssertionError:rejected=True
        assert rejected;check(f'{violation}_last_row_rejected_before_first_row_numeric_access')
    ctx=fixture();ctx._query['F13_014_00']['out_temp']=''
    assert ctx.query_prefix('F13_014_00')['F13_014_00']['out_temp'] is None;check('blank_value_preserved_as_missing')
    ctx=fixture();ctx._query['F13_014_00']['out_temp']='nan'
    rejected=False
    try:ctx.query_prefix('F13_014_00')
    except AssertionError:rejected=True
    assert rejected;check('nonfinite_allowed_observation_rejected')
    paths=[Path(__file__),HERE/'ch2_query_context_v2.py',HERE/'blk_context_v1.py']
    result={'status':'SYNTHETIC_LAZY_CONTEXT_BOUNDARIES_PASS_NOT_REAL_QUERY_GATE','check_count':len(checks),'checks':checks,
            'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'competition_rows_parsed':0,'model_fit':False,'performance_evaluated':False}
    out=HERE/'CH2_query_context_synthetic_audit_v1.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Lazy query context synthetic{len(checks)} checks PASS; no competition input access')

if __name__=='__main__':main()
