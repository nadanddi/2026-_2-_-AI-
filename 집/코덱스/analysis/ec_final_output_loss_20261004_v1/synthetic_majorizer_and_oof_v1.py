"""合成 local Hessian majorizer/inner OOF source-schema contamination reject."""
from pathlib import Path
import importlib.util,json,copy,hashlib
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('derivative_synthetic',H/'synthetic_derivatives_v1.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
def audit_inner(record,outer_train,outer_query):
    assert set(record)=={'inner_train','inner_query','pfn_context','season_train_days','raw_members','stage'}
    tr=set(map(tuple,record['inner_train']));q=set(map(tuple,record['inner_query']));ctx=set(map(tuple,record['pfn_context']))
    assert tr and q and len(tr)==len(record['inner_train']) and len(q)==len(record['inner_query'])
    assert tr<=outer_train and q<=outer_train and not (tr|q)&outer_query and not tr&q
    days_tr={(f,d) for f,d,h in tr};days_q={(f,d) for f,d,h in q}
    assert not days_tr&{(f,d+j) for f,d in days_q for j in [-1,0,1]}
    assert ctx and len(ctx)==len(record['pfn_context']) and ctx<=tr
    assert set(map(tuple,record['season_train_days']))==days_tr
    assert record['raw_members']==['ET','MLP','PFN'] and record['stage']=='INNER_OOF_HELDOUT_PUBLIC_ONLY'
    assert all({h for ff,dd,h in rows if (ff,dd)==(f,d)}==set(range(24)) for rows,days in [(tr,days_tr),(q,days_q)] for f,d in days)
def main():
    target=H/'synthetic_majorizer_oof_result_v1.json';assert not target.exists()
    rng=np.random.default_rng(20261004);eps=1e-9;min_eig=1e99
    for _ in range(100):
        f=rng.uniform(-2,2,12);b=rng.uniform(-2,2,12);y=rng.uniform(0,3,12);s=M.block_s([6,6])
        _,g,h,gn,active,a,z=M.derivatives(f,b,y,s,.1,2.)
        hh=gn.sum(axis=1)+abs(g)+eps
        assert (hh>0).all()
        eig=float(np.linalg.eigvalsh(np.diag(hh)-h).min());assert eig>=-1e-10;min_eig=min(min_eig,eig)
        d=.24*np.exp(f)
        # matrix-free row-sum through forward and reverse causal prefix.
        row_sum=d*(s.T@(active*(s@d)))
        M.near(row_sum,gn.sum(axis=1),1e-12)
    outer_train={(f,d,h) for f in ['F13','F47'] for d in range(1,11) for h in range(24)}
    outer_query={(f,20,h) for f in ['F13','F47'] for h in range(24)}
    tr=sorted((f,d,h) for f,d,h in outer_train if d not in [4,5,6]);q=sorted((f,d,h) for f,d,h in outer_train if d==5)
    record=dict(inner_train=[list(x) for x in tr],inner_query=[list(x) for x in q],pfn_context=[list(x) for x in tr[:24]],season_train_days=[list(x) for x in sorted({(f,d) for f,d,h in tr})],raw_members=['ET','MLP','PFN'],stage='INNER_OOF_HELDOUT_PUBLIC_ONLY')
    audit_inner(record,outer_train,outer_query);rejected=0
    for kind in ['query_label_context','neighbor_day','outer_query','season_heldout','insample_stage','member_omission','duplicate_train']:
        bad=copy.deepcopy(record)
        if kind=='query_label_context':bad['pfn_context'].append(list(q[0]))
        elif kind=='neighbor_day':bad['inner_train']+= [['F13',4,h] for h in range(24)]
        elif kind=='outer_query':bad['inner_query']=[list(x) for x in sorted(outer_query)]
        elif kind=='season_heldout':bad['season_train_days'].append(['F13',5])
        elif kind=='insample_stage':bad['stage']='OUTER_FIT_TRAIN_PREDICTION'
        elif kind=='member_omission':bad['raw_members']=['ET','MLP']
        else:bad['inner_train'].append(bad['inner_train'][0])
        try:audit_inner(bad,outer_train,outer_query)
        except AssertionError:rejected+=1
        else:raise AssertionError(('contamination accepted',kind))
    result=dict(status='PASS_SYNTHETIC_LOCAL_MAJORIZE_AND_OOF_SCHEMA',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),majorizer_cases=100,minimum_q_minus_exact_hessian_eigenvalue=min_eig,oof_corruptions_rejected=rejected,formula='h=GN rowsum + abs(g) + eps; PSD bound at same smooth-region point only',limitations=['not a global clipped loss upper bound','synthetic provenance schema rejects contamination; real provenance requires source/cache checks'],fit=0,predict=0,real_score=0,raw_ec_reads=0,test_reads=0,EL1_rescore=0)
    with target.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2)
    print('PASS_SYNTHETIC_LOCAL_MAJORIZE_AND_OOF_SCHEMA',min_eig,rejected)
if __name__=='__main__':main()
