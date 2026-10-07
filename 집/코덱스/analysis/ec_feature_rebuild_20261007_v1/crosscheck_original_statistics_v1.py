"""Independent Counter/struct draw replay and synthetic statistic boundary checks."""
from pathlib import Path
from collections import Counter
import hashlib,json,random,struct,importlib.util
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
regpath=HERE/'DOMAIN24_original_statistics_registration_v2.json'
reg=json.loads(regpath.read_text(encoding='utf-8'))
assert reg['status']=='ORIGINAL_DOMAIN_DIAGNOSTIC_DRAWS_SEALED_BEFORE_FIT_NO_LABELS'
assert all(sha(p)==v for p,v in reg['source_sha256'].items())
drawpath=HERE/'DOMAIN24_original_bootstrap_draws_v2.bin'
assert sha(drawpath)==reg['draw_file_sha256']
layout=reg['layout'];ids=layout['ordered_DIAG_ids'];tm=set(layout['ordered_TM_ids'])
assert len(ids)==len(set(ids))==8640 and len(tm)==2664
blocks=layout['block_keys'];n=Counter();nt=Counter()
for rid,i in zip(ids,layout['membership']):
    farm,day,hour=rid.split('_');bucket,_=divmod(int(day),5)
    assert blocks[i]==[farm,bucket]
    n[i]+=1;nt[i]+=int(rid in tm)
assert [n[i] for i in range(len(blocks))]==layout['rows_per_block']
assert [nt[i] for i in range(len(blocks))]==layout['TM_rows_per_block']
groups=[[i for i,b in enumerate(blocks) if b[0]==farm] for farm in ['F13','F47']]
assert groups==layout['groups'] and reg['alpha']==.025/84
rng=random.Random(2026100703);width=len(blocks)*2;fmt='<'+str(len(blocks))+'H'
accepted=0;rejected=0
with drawpath.open('rb') as handle:
    while accepted<200000:
        counted=Counter()
        for group in groups:
            counted.update(group[rng.randrange(len(group))] for _ in range(len(group)))
        if sum(counted[i]*nt[i] for i in range(len(blocks)))==0:
            rejected+=1;continue
        raw=handle.read(width);assert len(raw)==width
        counts=struct.unpack(fmt,raw)
        assert counts==tuple(counted[i] for i in range(len(blocks)))
        for group in groups:assert sum(counts[i] for i in group)==len(group)
        assert sum(counts[i]*n[i] for i in range(len(blocks)))>0
        accepted+=1
    assert handle.read(1)==b''
assert rejected==reg['rejected_zero_TM_denominator']
path=HERE/'original_domain_statistics_v2.py'
spec=importlib.util.spec_from_file_location('frozen_statistics',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
tests=[]
for subset in ['DIAG10','TM111']:
    for value,expected in [(0.,1.),(1.,1.),(-1.,1/200001)]:
        output=module.bootstrap_loss(drawpath,layout,dict.fromkeys(ids,value),subset)
        assert output['p_worse']==expected
        assert output['individual95_MSE_delta_CI']==[value,value]
        tests.append({'subset':subset,'synthetic_constant_delta':value,'p_worse':expected,'PASS':True})
bad=dict.fromkeys(ids,0.);bad.pop(ids[0])
try:module.bootstrap_loss(drawpath,layout,bad,'DIAG10')
except AssertionError:tests.append({'case':'missing_loss_row_rejected','PASS':True})
else:raise AssertionError('missing row accepted')
bad=dict.fromkeys(ids,0.);bad[ids[0]]=float('nan')
try:module.bootstrap_loss(drawpath,layout,bad,'DIAG10')
except AssertionError:tests.append({'case':'nonfinite_loss_rejected','PASS':True})
else:raise AssertionError('nonfinite loss accepted')
result={'status':'INDEPENDENT_ORIGINAL_STATISTICS_DRAW_AND_SYNTHETIC_PASS',
        'code_sha256':sha(__file__),'registration_sha256':sha(regpath),'draws_replayed':accepted,
        'blocks_checked':len(blocks),'row_membership_checked':len(ids),'synthetic_tests':tests,
        'target_values_read':False,'model_fit':False,
        'limits':['Full draw replay uses same specified random generator; count implementation differs',
                  'Synthetic loss cases test statistic boundaries, not model effects or fresh validation']}
with (HERE/'DOMAIN24_original_statistics_independent_crosscheck_v1.json').open('x',encoding='utf-8') as h:
    json.dump(result,h,ensure_ascii=False,indent=2)
print('Independent all200k Counter replay + 8 synthetic statistic checks PASS; no model/target values')
