from pathlib import Path
import sys,json,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/local/ec_model_packages_20261002_v1'))
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/temp_season_20261002_v1/reference'))
import numpy as np,pandas as pd
import common,harness,temp_mask_v1 as TM,train_flags_v6 as TF
from anal_q1_errors import diag_folds
from resid_reset_features import build_features,FEATURE_COLUMNS,PHYSICS_COLUMNS,USABLE
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
OUT=ROOT/'집/코덱스/local/temp_group_loss_20261003_v1';OUT.mkdir(parents=True,exist_ok=True)
DAYCOL=[c+'_h0' for c in USABLE]+['farm_id','day','second']
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

from model import fit_fold
def main():
    start=time.time();common.load_raw=TM.masked_loader
    try:lab,ct,phc=TM.build_world()
    finally:common.load_raw=TM.ORIG;harness._CACHE.clear()
    tx,_,sx=TM.masked_loader();cf=build_features(tx,sx).set_index('row_id')
    for c in FEATURE_COLUMNS:
        if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
    w=TF.row_weights(lab,.2,w_noisy=.2);zpath=Path(env.LOCAL)/'temp_mask_v1_oof.npz';z=dict(np.load(zpath,allow_pickle=True));assert np.array_equal(lab.row_id.to_numpy(),z['row_id'])
    gp=ROOT/'집/코덱스/analysis/temp_validation_structure_20261003_v1/weather_groups.csv';groups=pd.read_csv(gp);gm={(r.farm,int(r.day)):int(r.weather_group) for r in groups.itertuples()};gids=np.array([gm[(f,int(d))] for f,d in zip(lab.farm,lab.day)])
    dmin=lab.groupby(['farm','day']).ph_in_temp_3.min();fds=diag_folds(lab);sets=[('DIAG10',fds)]
    for th in (10,12):
        ix=dmin[dmin<th].index;sets.append((f'EXT{th}',[{f:set(int(d) for ff,d in ix if ff==f) for f in common.TARGET_FARMS}]))
    records=[];stats=[]
    for name,folds in sets:
        arrays={tag:{s:np.full(len(lab),np.nan) for s in (726,727)} for tag in (['H1','H2','CODEX','CB','ET'] if name=='GUARD' else ['L1','S1','D1'])};fid=np.full(len(lab),-1)
        for k,fd in enumerate(folds):
            tm,vm=common.split_mask(lab,fd);old_n=int(tm.sum())
            if name=='GUARD':tm&=~np.isin(gids,np.unique(gids[vm]));assert not set(gids[tm])&set(gids[vm])
            tr,va=lab[tm].copy(),lab[vm].copy();cp=OUT/f'{name}_{k}.npz';stp=OUT/f'{name}_{k}_training.json'
            if cp.exists():payload=dict(np.load(cp,allow_pickle=True));training=json.loads(stp.read_text(encoding='utf-8'));assert np.array_equal(payload['row_id'],va.row_id.to_numpy())
            else:
                payload={'row_id':va.row_id.to_numpy(dtype=str)};training={}
                for seed in (726,727):
                    predictions,ts=fit_fold(tr,va,w[tm],seed,name=='GUARD');training[str(seed)]=ts
                    for tag,p in predictions.items():payload[f'{tag}_{seed}']=p
                    print(f'{name}/{k}/{seed} trained {len(tr)//24} days, H1 {rmse(predictions["L1"],va.sub_temp.to_numpy()):.5f} H2 {rmse(predictions["S1"],va.sub_temp.to_numpy()):.5f}, elapsed {time.time()-start:.0f}s',flush=True)
                np.savez(cp,**payload);stp.write_text(json.dumps(training,ensure_ascii=False,indent=2),encoding='utf-8')
            for seed in (726,727):
                for tag in arrays:arrays[tag][seed][vm]=payload[f'{tag}_{seed}']
                stats.append(dict(validator=name,fold=k,seed=seed,training_rows=len(tr),validation_rows=len(va),weather_extra_removed=old_n-len(tr),scores=training[str(seed)]))
            fid[vm]=k
        if name=='GUARD':
            for seed in (726,727):
                ok=fid>=0
                for tag in ['H1','H2','CB','ET']:
                    frame=lab.loc[ok,['row_id','farm','day','hour','in_temp','sub_temp']].copy();frame['validator']=name;frame['scope']='member_only';frame['member']=tag;frame['seed']=seed;frame['context']='none';frame['fold']=fid[ok];frame['base']=arrays['CODEX'][seed][ok];frame['candidate']=arrays[tag][seed][ok];records.append(frame)
            continue
        pfs=[np.load(Path(env.LOCAL)/f'web_tabpfn_v2_temp_{name}.npy').mean(0),np.load(Path(env.LOCAL)/f'web_tabpfn_v6_temp_{name}.npy')[:8].mean(0)];g=np.where(np.isnan(lab.in_temp),1.,np.clip((lab.in_temp.to_numpy()-8)/2,0,1))
        for bs,cs in [(7,726),(101,727)]:
            base,cx=z[f'{name}__MASK__{bs}'],z[f'{name}__CODEX__{cs}']
            for context,pfn in zip(['1-8','17-24'],pfs):
                ref=(.4+.1*g)*base+(.6-.4*g)*cx+.3*g*pfn
                for tag in ['L1','S1','D1']:
                    cand=(.4+.1*g)*base+(.6-.4*g)*arrays[tag][cs]+.3*g*pfn;ok=np.isfinite(ref)&np.isfinite(cand);assert np.array_equal(ok,fid>=0)
                    frame=lab.loc[ok,['row_id','farm','day','hour','in_temp','sub_temp']].copy();frame['validator']=name;frame['scope']='W30G';frame['member']=tag;frame['seed']=cs;frame['context']=context;frame['fold']=fid[ok];frame['base']=ref[ok];frame['candidate']=cand[ok];records.append(frame)
    oof=pd.concat(records,ignore_index=True);oof.to_csv(OUT/'oof.csv',index=False);summary=[];rng=np.random.default_rng(20261003)
    for (scope,tag,name,seed,context),d in oof.groupby(['scope','member','validator','seed','context'],sort=True):
        a=rmse(d.base.to_numpy(),d.sub_temp.to_numpy());b=rmse(d.candidate.to_numpy(),d.sub_temp.to_numpy());item=dict(scope=scope,member=tag,validator=name,seed=int(seed),context=context,n=len(d),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1))
        if name in ['DIAG10','GUARD']:
            d=d.copy();d['block']=d.farm+'_'+(d.day//5).astype(str);d['dmse']=(d.candidate-d.sub_temp)**2-(d.base-d.sub_temp)**2;blocks=d.groupby('block').dmse.agg(['sum','count']);ix=rng.integers(0,len(blocks),size=(20000,len(blocks)));samples=blocks['sum'].to_numpy()[ix].sum(1)/blocks['count'].to_numpy()[ix].sum(1);alpha=.025/10;item.update(p_worse=float(np.mean(samples>=0)),ci=np.quantile(samples,[alpha,1-alpha]).tolist())
        summary.append(item);print(json.dumps(item),flush=True)
    verdicts={tag:('PASS_REQUIRES_FULL_GUARD' if all(s['delta_pct']<0 for s in summary if s['scope']=='W30G' and s['member']==tag) and all(s['p_worse']<.025/10 and s['ci'][1]<0 for s in summary if s['scope']=='W30G' and s['member']==tag and s['validator']=='DIAG10') else 'REJECT') for tag in ['L1','S1','D1']}
    result=dict(verdicts=verdicts,summary=summary,training_scores=stats,elapsed=time.time()-start,day_columns=DAYCOL,feature_columns=FEATURE_COLUMNS,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'run.py',HERE/'model.py',HERE/'PROTOCOL.md',gp,zpath]})
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(verdicts),flush=True)
if __name__=='__main__':main()
