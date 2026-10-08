"""Classify record-order proxy roles, never real sensor or greenhouse labels."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H))
import audit_v2 as A
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
C=['in_temp','in_hum','in_co2','act_heating','act_vent','act_thermal','act_shade','act_circfan','act_fog','act_co2']
def main():
    O=H/'role_v1';O.mkdir(exist_ok=False)
    x,te,y,b,m=A.read_inputs()
    roles=pd.read_csv(H/'rest_v2/roles.csv')
    fits=0;rows=[];stats=[]
    for scope,xx in [('h0',x[x.hour==0]),('full_day_posthoc',x)]:
        z=xx.groupby(['farm','day'])[C].mean().reset_index().merge(roles,on=['farm','day'],validate='one_to_one')
        for farm,g in z.groupby('farm'):
            g=g.sort_values('day').reset_index(drop=True);gid=g.groupby('gid').day.min().sort_values().index.to_numpy()
            assert (g.groupby('gid').size()==2).all()
            for layout in ['contiguous_record_blocks','interleaved_groups']:
                blocks=np.array_split(gid,5) if layout=='contiguous_record_blocks' else [gid[k::5] for k in range(5)]
                pp=np.full(len(g),np.nan);label=g.role.eq('B').astype(int).to_numpy()
                for k,block in enumerate(blocks):
                    val=g.gid.isin(block).to_numpy();tr=~val
                    assert set(g.loc[val,'gid']).isdisjoint(g.loc[tr,'gid'])
                    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=1000))
                    model.fit(g.loc[tr,C],label[tr]);fits+=1
                    assert model[-1].n_iter_.max()<1000
                    pp[val]=model.predict_proba(g.loc[val,C])[:,1]
                    for i in np.flatnonzero(val):rows.append(dict(farm=farm,day=int(g.loc[i,'day']),gid=int(g.loc[i,'gid']),role=g.loc[i,'role'],scope=scope,layout=layout,fold=k,p=float(pp[i])))
                assert np.isfinite(pp).all()
                stats.append(dict(farm=farm,scope=scope,layout=layout,n=len(g),pairs=len(gid),auc=roc_auc_score(label,pp)))
    pd.DataFrame(rows).to_csv(O/'role_oof.csv',index=False);pd.DataFrame(stats).to_csv(O/'role_auc.csv',index=False)
    days=pd.read_csv(H/'first_v2/ec_days.csv',dtype={'seed':str}).merge(roles,on=['farm','day'],validate='many_to_one')
    bias=[]
    for (seed,farm,role,sealed),g in days.groupby(['seed','farm','role','sealed']):
        bias.append(dict(seed=seed,farm=farm,role=role,sealed=bool(sealed),n=len(g),bias=g.bias.mean(),rmse=np.sqrt(g.sse.sum()/g.n.sum()),high=int((g.truth>=1).sum())))
    pd.DataFrame(bias).to_csv(O/'role_bias.csv',index=False)
    m.update(source_sha=A.sha(__file__),audit_source_sha=A.sha(H/'audit_v2.py'),classification_fits=fits,ec_model_fits=0,
             target='order within adjacent exact-weather input pairs; real dong unobserved',
             diagnostic_only=True,layouts='group-disjoint five folds, no record-neighbor purge; not prospective evaluation or model adoption',
             output_sha={p.name:A.sha(p) for p in O.iterdir() if p.is_file()})
    A.save(O/'manifest.json',m)
    print(pd.DataFrame(stats).to_string(index=False));print(pd.DataFrame(bias).query("seed=='ensemble'").to_string(index=False));print('COMPLETE role',fits)
if __name__=='__main__':main()
