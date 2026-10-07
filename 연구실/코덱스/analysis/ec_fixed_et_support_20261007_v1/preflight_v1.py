from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
import numpy as np,pandas as pd
def main():
    raw,jobs,pairs,ct,cq,public,vec=R.prep();rows=[]
    ens=pd.read_csv(R.B.H/'baseline_score_v1/days.csv',float_precision='round_trip');ens=ens[ens.seed.astype(str)=='ensemble']
    for p in pairs:
        if p.get('historical'):continue
        farm,day,k=p['farm'],p['day'],p['fold'];t,q,_=jobs[k];pool=ens[(ens.farm==farm)&(ens.k==k)&ens.ordinary&(ens.rmse<=.1)].copy()
        for seed in R.B.SEEDS:
            with np.load(R.B.L/f'base/{k}_{seed}.npz') as z:pred=R.B.M.shrink(z['et'],q)
            v=q[['farm','day','sub_ec']].copy();v['pred']=pred;gg=v.groupby(['farm','day']);bias=abs(gg.pred.mean()-gg.sub_ec.mean())
            pool=pool[[bias.loc[(f,d)]<=.1 for f,d in pool[['farm','day']].itertuples(index=False,name=None)]]
        cc=['season']+[c+'_h0' for c in R.B.M.RAW];xt=t[cc].to_numpy(float);med=np.nanmedian(xt,axis=0);std=np.nan_to_num(np.where(np.isnan(xt),med,xt)).std(axis=0);std[std==0]=1
        def x0(d):
            v=q[(q.farm==farm)&(q.day==d)&(q.hour==0)][cc].to_numpy(float)[0];return np.where(np.isnan(v),med,v)
        a=x0(day);v=[]
        for rr in pool.itertuples():
            gap=abs(rr.truth-p['truth']);step=1 if rr.pass2==(day>=179) and gap<=.2 else 2 if gap<=.2 else 3 if gap<=.3 else 4
            v.append(dict(pair=p['tag'],candidate=int(rr.day),step=step,ec_gap=float(gap),distance=float(np.mean(((x0(rr.day)-a)/std)**2)),selected=int(rr.day)==p['control']))
        chosen=min(v,key=lambda x:(x['step'],x['distance'],x['candidate']));assert chosen['candidate']==p['control'];rows.extend(v)
    pd.DataFrame(rows).to_csv(H/'control_candidates_v1.csv',index=False)
    source=R.B.L/'trace/BASE.npz';meta=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'));assert meta['sha']==R.B.sha(source)
    originals=[]
    for k in [1,4,8]:
        t,q,_=jobs[k];p=R.B.L/f'base/{k}_7.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==R.B.sha(p)
        originals.append(dict(fold=k,rows=len(t),days=len(t)//24,train_hash=R.B.S.fhash(t,['row_id','sub_ec']+R.COLS),query_hash=R.B.S.fhash(q,['row_id','sub_ec']+R.COLS),cache_sha=R.B.sha(p)))
    R.write(H/'preflight_v1.json',dict(status='PASS_FIT0',fit=0,source_snapshot_sha=R.B.sha(source),source_snapshot_meta_sha=R.B.sha(source.with_suffix('.json')),originals=originals,common_train_days=len(ct)//24,candidate_rows=len(rows),candidate_sha=R.B.sha(H/'control_candidates_v1.csv'),prep_sha=R.B.sha(H/'preparation_v1.json'),code_sha=R.B.sha(__file__)))
    print('PREFLIGHT_PASS',len(rows),flush=True)
if __name__=='__main__':main()
