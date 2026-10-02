from pathlib import Path
import sys,json,math,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np,pandas as pd
TP=ROOT/'집/코덱스/local/temp_tk_season_20261003_v1/TK1_predictions.csv';EP=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
temp=pd.read_csv(TP);ec=pd.read_csv(EP);records=[];details=[]
def examine(d,p,blend,target,label):
    lo,hi=p.min(axis=1),p.max(axis=1);y=d[target].to_numpy();near=np.minimum(np.maximum(y,lo),hi);outside=(y<lo)|(y>hi);floor=(near-y)**2;errors=p-y[:,None];base=(blend-y)**2
    for seg,mask in {'all':np.ones(len(d),bool),'early':d.day.to_numpy()<179,'late':d.day.to_numpy()>=179,'F47_late':(d.day.to_numpy()>=179)&(d.farm.to_numpy()=='F47')}.items():
        if not mask.sum():continue
        a,b=base[mask],floor[mask];fs=float(b.sum()/a.sum());n=int(mask.sum());manual_floor=[];manual_base=[];manual_out=0
        for preds,truth,bp in zip(p[mask],y[mask],blend[mask]):
            small,big=min(map(float,preds)),max(map(float,preds));distance=max(small-float(truth),0,float(truth)-big);manual_floor.append(distance**2);manual_base.append((float(bp)-float(truth))**2);manual_out+=int(float(truth)<small or float(truth)>big)
        assert abs(fs-math.fsum(manual_floor)/math.fsum(manual_base))<1e-12;assert manual_out==int(outside[mask].sum());assert abs(math.sqrt(np.mean(a))-math.sqrt(math.fsum(manual_base)/n))<1e-12
        corr=np.corrcoef(errors[mask].T)
        records.append(dict(target=label,segment=seg,n=n,days=len(d.loc[mask,['farm','day']].drop_duplicates()),outside_n=manual_out,outside_pct=100*manual_out/n,baseline_rmse=float(np.sqrt(a.mean())),oracle_hull_rmse=float(np.sqrt(b.mean())),unavoidable_sse_pct=100*fs,pairwise_error_correlation=corr[np.triu_indices(p.shape[1],1)].tolist(),fsum_pass=True))
for seed in [7,101]:
    for context in ['1-8','17-24']:
        t=temp[(temp.validator=='DIAG10')&(temp.base_seed==seed)&(temp.context==context)];d=t[t.member=='W30G'].set_index('row_id');p=np.column_stack([t[t.member==m].set_index('row_id').prediction.reindex(d.index) for m in ['BASE','CODEX','PFN']]);assert len(d)==9600 and np.isfinite(p).all();examine(d.reset_index(),p,d.prediction.to_numpy(),'sub_temp',f'TEMP_{seed}_{context}')
for seed in [7,101,2024]:
    d=ec[(ec.validator=='DIAG10')&(ec.seed==seed)].copy();assert len(d)==8640;examine(d,d[['season_r3','season_pfn']].to_numpy(),d.season_v2.to_numpy(),'sub_ec',f'EC_{seed}_two_superexperts')
(HERE/'diagnostic.json').write_text(json.dumps(dict(status='PASS',findings=records,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [TP,EP,HERE/'diagnose.py']}),ensure_ascii=False,indent=2),encoding='utf-8');pd.DataFrame(records).to_csv(HERE/'diagnostic.csv',index=False)
for r in records:
    if r['target'] in ['TEMP_7_1-8','EC_7_two_superexperts']:print(json.dumps(r))
