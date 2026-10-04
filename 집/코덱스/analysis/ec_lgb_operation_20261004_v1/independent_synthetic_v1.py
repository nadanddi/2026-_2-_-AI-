"""v3 독립 합성 stress: 실제 candidate/score/cache 읽기0, fit/predict0."""
from pathlib import Path
import importlib.util, json, math, hashlib, copy
H=Path(__file__).resolve().parent
path=H/'verify_full_v3.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='a888c7831fa0647e305e825b728bf8fc88cc0744efd812beb420873d4ce3a1da'
sp=importlib.util.spec_from_file_location('synthetic_lgb_verifier',path)
V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V);V.bootstrap(False)
np,pd=V.np,V.pd
day=np.tile(np.repeat(np.arange(180),24),2)
farm=np.array(['F13']*4320+['F47']*4320)
# 양/음 손실차가 farm/day별로 다르며 표본 방향이 모두 존재하는 fixture.
diff=np.where((day//5)%2==0,-.2,.19)+np.where(farm=='F13',.002,-.001)
q=pd.DataFrame(dict(farm=farm,day=day,hour=np.tile(np.arange(24),360),y=np.zeros(8640),baseline=np.ones(8640),candidate=np.sqrt(1+diff)))
boots={}
for seed in V.SEEDS:
    b=V.boot(q,seed)
    assert .1<b['p_worse']<.9 and b['ci_adjusted'][0]<0<b['ci_adjusted'][1]
    boots[seed]=b
# 블록 내 일부일만 남아 tail block이3일인8640행 fixture(180->183일구조는 행수고정때문에 별도 sample-summary).
mixed=np.r_[np.linspace(-1,-.00001,9999),0.,np.linspace(.00001,1,10000)]
b=V.summary_boot(mixed)
assert b['p_worse']==10001/20000 and b['ci_adjusted'][0]<0<b['ci_adjusted'][1]
rng=np.random.default_rng(21)
small=q.iloc[:48].copy().reset_index(drop=True)
et,lgb,mlp,pfn=rng.normal(size=(4,48))
scalar=V.scalar_candidate(small,et,lgb,mlp,pfn,-.2,.4)
order=rng.permutation(48)
shuffled=V.scalar_candidate(small.iloc[order].reset_index(drop=True),et[order],lgb[order],mlp[order],pfn[order],-.2,.4)
V.compare(shuffled,scalar[order])
signature=dict(seed=7,train_new23='A',query_new23='B')
meta=dict(status='PASS',signature=signature,csv_sha256='a'*64,fit_predict_seconds=1.,first_audit_sha256='b'*64)
V.validate_meta(meta,signature,'a'*64,'b'*64)
rejected=0
for kind in ['signature','csv_digest','extra','nan_seconds','first_sha','nonfirst_sha']:
    bad=copy.deepcopy(meta)
    if kind=='signature':bad['signature']['train_new23']='X'
    elif kind=='csv_digest':bad['csv_sha256']='c'*64
    elif kind=='extra':bad['unexpected']=1
    elif kind=='nan_seconds':bad['fit_predict_seconds']=float('nan')
    elif kind=='first_sha':bad['first_audit_sha256']='c'*64
    try:V.validate_meta(bad,signature,'a'*64,None if kind=='nonfirst_sha' else 'b'*64)
    except (AssertionError,KeyError):rejected+=1
    else:raise AssertionError(('accepted corruption',kind))
result=dict(status='PASS_SYNTHETIC_STRESS_ONLY',verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mixed_day_loss_bootstrap=boots,ties_sample_summary=b,meta_corruptions_rejected=rejected,checks=V.CHECKS,actual_candidate_reads=0,real_score=0,fit=0,predict=0)
with (H/'independent_synthetic_result_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print('PASS_SYNTHETIC_STRESS_ONLY',rejected,V.CHECKS)
