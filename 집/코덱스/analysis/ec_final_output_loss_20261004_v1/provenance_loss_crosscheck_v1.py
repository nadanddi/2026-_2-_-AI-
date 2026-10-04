"""Independent read-only digest/complete metadata binding. No fitting or prediction."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
r=read(H/'verification_registration_v1.json');p=read(H/'preparation_v3.json');fit=read(H/'fit_audit_v1.json');whole=read(H/'full_verification_v3.json')
assert r['status']=='PINNED_BEFORE_ACTUAL_FIT' and r['models_fit']==r['score_count']==0
for rel,digest in r['files'].items():assert sha(ROOT/rel)==digest,rel
assert hashlib.sha256(json.dumps(p['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==r['config_sha256']
assert p['config']['alpha']==.025/24 and p['config']['epochs']==400 and p['config']['strict_15']
assert fit['preparation']==p and fit['config']==p['config'] and fit['score_count']==0 and len(fit['cells'])==132
assert whole['receipt_sha256']==sha(H/'verification_registration_v1.json') and whole['fit_audit_sha256']==sha(H/'fit_audit_v1.json')
for name,digest in whole['outputs_sha256'].items():assert sha(H/name)==digest
metas=[];inits={};firsts=[]
for mode,family in [('FINAL_LOSS',22),('RAW_LOSS',23)]:
 for sig in p['manifest']:
  v,k=sig['validator'],sig['fold']
  for seed in (7,101,2024):
   name=f'{mode}_{v}_{k}_{seed}';m=read(OUT/(name+'.json'))
   expected=dict(sig,seed=seed,mode=mode,family=family,preparation_sha256=sha(H/'preparation_v3.json'),preregistration_sha256=sha(H/'preregistration_v1.md'))
   assert m['signature']==expected and m['status']=='PASS'
   assert m['csv_sha256']==sha(OUT/(name+'_pred.csv')) and m['checkpoint_sha256']==sha(OUT/(name+'_model.npz'))
   t=m['train'];assert t['config']==p['config'] and t['epochs']==400 and t['all_gradients_finite'] is True and t['all_parameters_finite'] is True
   init=t['initial_weights_sha256'];assert init==sig['initial_weights_sha256'][str(seed)]
   if seed in inits:assert inits[seed]==init
   inits[seed]=init
   if (v,k,seed)==('DIAG10',0,7):
    fp=H/f'first_{mode}_v1.json';f=read(fp);assert m['first_audit_sha256']==sha(fp) and f['signature']==expected and f['atol']==1e-12
    assert f['all_gradients_finite'] is True and f['all_parameters_finite'] is True
    assert all(0<=x<=1e-12 for x in f['errors'].values())
    firsts.append(dict(mode=mode,sha256=sha(fp),errors=f['errors']))
   metas.append(m)
assert metas==fit['cells']
assert fit['aggregate_sha256']==sha(OUT/'oof.csv')
for mode,digest in fit['mode_aggregate_sha256'].items():assert sha(OUT/f'oof_{mode}.csv')==digest
result=dict(status='PASS_132_METADATA_AND_REGISTERED_DIGESTS',cells=132,registered_files=len(r['files']),config_sha256=r['config_sha256'],init_same_across_modes=inits,firsts=firsts,actual_fit=0,actual_predict=0,actual_score=0,raw_ec_reads=0,test_reads=0,EL1_reads=0,limitations='Timing of preregistration still requires parent git evidence; this is digest verification, not optimizer replay. Public/cache feature/purge reconstruction is parent whole evidence.')
with (H/'provenance_loss_crosscheck_result_v1.json').open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
print(result['status'])
