"""Independent arithmetic verifier, never fits a model or loads locked labels."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys,csv,json,math,hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=ROOT/'집/코덱스/local/ec_member_ablation_20261002_v1'
PHASE=ROOT/'집/코덱스/local/ec_restart_phase3_20261001_v1'
W={'et':.48,'lgb':.24,'mlp':.08,'pfn':.20}
ARMS=['drop_et','drop_lgb','drop_mlp','drop_pfn']
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load_json(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(errors):return math.sqrt(math.fsum(e*e for e in errors)/len(errors))
def transform(ids,values,inverse=False):
    groups=defaultdict(list)
    for i,rid in enumerate(ids):
        f,d,h=rid.split('_');groups[(f,int(d))].append((int(h),i))
    out=[0.]*len(ids)
    for g in groups.values():
        total=0.
        for n,(_,i) in enumerate(sorted(g),1):
            val=float(values[i])
            if inverse:
                p=(2*n*val-total)/(n+1);out[i]=p;total+=p
            else:
                total+=val;out[i]=.5*val+.5*total/n
    return out
def clip(values,lo,hi):return [min(hi,max(lo,float(v))) for v in values]
def main():
    audits=load_json(HERE/'reconstruction_audit.json')
    locks={(r['farm'],int(r['day'])) for r in load_json(Path(env.CODEX)/'ec_final_lock/locked_days.json')['selected']}
    truth={};excluded=0
    for r in read(Path(env.DATA)/'train_y.csv'):
        f,d,_=r['row_id'].split('_')
        if f not in ['F13','F47']:continue
        if (f,int(d)) in locks:excluded+=1;continue
        truth[r['row_id']]=float(r['sub_ec'])
    assert excluded==960 and len(truth)==8640
    independent=[];predicted={}
    for a in audits:
        name=a['validator'];fold=int(a['validation_fold']);seed=int(a['seed'])
        with np.load(OUT/f'{name}_{fold}_seed{seed}_members.npz') as z:
            ids=z['row_id'].tolist();e=z['et'].tolist();l=z['lgb'].tolist();m=z['mlp'].tolist()
        assert all(rid in truth for rid in ids)
        with np.load(PHASE/f'{name}_{fold}.npz') as z:
            assert ids==z['row_id'].tolist()
            oldr=z[f'r3_{seed}'].tolist();oldv=z[f'v2_{seed}'].tolist()
        lo=float(a['train_label_min']);hi=float(a['train_label_max'])
        r3raw=[math.fsum([.6*et,.3*lg,.1*ml]) for et,lg,ml in zip(e,l,m)]
        fresh=clip(transform(ids,r3raw),lo,hi)
        r3gap=max(abs(x-y) for x,y in zip(fresh,oldr))
        assert abs(r3gap-a['refit_finished_r3_max_difference'])<1e-10
        cr=sum(v<=lo or v>=hi for v in oldr);cv=sum(v<=lo or v>=hi for v in oldv)
        assert cr==a['clip_r3_count'] and cv==a['clip_v2_count']
        assert sha(PHASE/f'{name}_{fold}.npz')==a['cache_hash']
        rec={'validator':name,'fold':fold,'seed':seed,'rows':len(ids),'r3_gap_independent':r3gap,
             'clip_r3_count_independent':cr,'clip_v2_count_independent':cv}
        if r3gap>1e-6 or cr+cv>0:
            rec['gate_pass']=False;independent.append(rec)
            continue
        q=[(v-.8*r)/.2 for r,v in zip(oldr,oldv)]
        rawp=transform(ids,q,inverse=True)
        roundtrip=max(abs(x-y) for x,y in zip(q,transform(ids,rawp)))
        assert roundtrip<1e-9
        rec['gate_pass']=True;rec['pfn_roundtrip_independent']=roundtrip
        for arm in ARMS:
            dropped=arm[5:];den=1-W[dropped]
            raw=[math.fsum(W[k]*z for k,z in [('et',et),('lgb',lg),('mlp',ml),('pfn',p)] if k!=dropped)/den
                 for et,lg,ml,p in zip(e,l,m,rawp)]
            pred=clip(transform(ids,raw),lo,hi)
            for rid,p in zip(ids,pred):predicted[(name,fold,seed,rid,arm)]=p
        independent.append(rec)
    if (HERE/'gate_failure.json').exists() and not (HERE/'completion.json').exists():
        assert any(not r['gate_pass'] for r in independent)
        result={'status':'PASS_BLOCKED_GATE','method':'csv + explicit per-day shrink/inverse + math.fsum',
                'audits':independent,'locked_rows_skipped_before_float':excluded,
                'model_fitted':False,'candidate_adoption_claimed':False}
        save(HERE/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return
    assert len(independent)==66 and all(r['gate_pass'] for r in independent)
    oof=read(OUT/'oof_predictions.csv');errors=defaultdict(list);seen=set();max_pred=0.
    for r in oof:
        name=r['validator'];fold=int(r['validation_fold']);seed=int(r['seed']);rid=r['row_id']
        ident=(name,fold,seed,rid);assert ident not in seen;seen.add(ident)
        assert abs(float(r['sub_ec'])-truth[rid])<1e-12
        for arm in ['v2']+ARMS:
            p=float(r[arm]);errors[(name,seed,arm)].append(p-truth[rid])
            if arm!='v2':
                delta=abs(p-predicted[(name,fold,seed,rid,arm)]);max_pred=max(max_pred,delta)
                assert delta<1e-9
    scoremap={(r['validator'],int(r['seed']),r['arm']):r for r in load_json(HERE/'evaluation.json')['scores']}
    scores=[]
    for key,err in sorted(errors.items()):
        score=rmse(err);assert abs(score-scoremap[key]['rmse'])<1e-12
        scores.append({'validator':key[0],'seed':key[1],'arm':key[2],'rmse_independent':score,'n':len(err)})
    result={'status':'PASS','method':'csv + explicit per-day shrink/inverse + math.fsum',
            'audits':independent,'verified_oof_rows':len(oof),'max_independent_candidate_difference':max_pred,
            'scores':scores,'locked_rows_skipped_before_float':excluded,'model_fitted':False,'final_lock_scored':False}
    save(HERE/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['audits','scores']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
