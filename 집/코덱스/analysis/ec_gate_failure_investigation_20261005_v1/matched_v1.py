"""Control date composition: same IDs/days across inner and outer contexts."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local'/H.name
def read(scope,mode):
    fs=sorted(OUT.glob(f'{scope}_{mode}_DIAG10_*.csv'));assert len(fs)==30;d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in fs],ignore_index=True);d['P']=d.A+d.g*(d.B-d.A);return d
def main():
    v=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'));assert v['status']=='PASS_DIAGNOSIS_AND_META270';all_days=[];summaries=[]
    for mode in ['LR','LGB','MLP']:
        inner=read('inner_resub',mode);meta=read('inner_meta_oof',mode);outer=read('outer_actual',mode)
        assert inner[['row_id','A','B','y']].equals(meta[['row_id','A','B','y']]);keys=['row_id','farm','day','hour'];columns=['A','B','P','g','y']
        ig=inner.groupby(['seed']+keys)[columns].mean().reset_index().rename(columns={c:'inner_'+c for c in columns});mg=meta.groupby(['seed']+keys)[columns].mean().reset_index().rename(columns={c:'meta_'+c for c in columns});og=outer[['seed']+keys+columns].rename(columns={c:'outer_'+c for c in columns});assert not og.duplicated(['seed','row_id']).any();matched=og.merge(ig,on=['seed']+keys,validate='one_to_one').merge(mg,on=['seed']+keys,validate='one_to_one')
        assert np.max(abs(matched.outer_y-matched.inner_y))<1e-14;assert np.max(abs(matched.meta_A-matched.inner_A))<1e-14
        # Independently check context averaging for three fixed shared IDs.
        for rid in sorted(matched.row_id.unique())[:3]:
            for seed in [7,101,2024]:
                records=inner[(inner.row_id==rid)&(inner.seed==seed)];expected=math.fsum(map(float,records.A))/len(records);actual=matched[(matched.row_id==rid)&(matched.seed==seed)].inner_A.iloc[0];assert abs(expected-actual)<1e-12
        numeric=[c for c in matched.columns if c.startswith(('inner_','meta_','outer_'))];days=matched.groupby(['seed','farm','day'])[numeric].mean().reset_index();days['mode']=mode;days['inner_r']=days.outer_y-days.inner_A;days['outer_r']=days.outer_y-days.outer_A;days['inner_d']=days.inner_B-days.inner_A;days['outer_d']=days.outer_B-days.outer_A;all_days.append(days)
        for seed in [7,101,2024]:
            d=days[days.seed==seed]
            groups=[('all',np.ones(len(d),bool)),('high',(d.outer_y>=1).to_numpy()),('ordinary',(d.outer_y<1).to_numpy()),('outer_A_ge_.9',(d.outer_A>=.9).to_numpy()),('pass2',(d.day>=179).to_numpy())]+[(f+'_pass'+str(p),((d.farm==f)&((d.day>=179)==(p==2))).to_numpy()) for f in ['F13','F47'] for p in [1,2]]
            for group,mask in groups:
                z=d[mask]
                if len(z)==0:continue
                summaries.append(dict(mode=mode,seed=seed,segment=group,days=len(z),inner_r=float(z.inner_r.mean()),outer_r=float(z.outer_r.mean()),inner_d=float(z.inner_d.mean()),outer_d=float(z.outer_d.mean()),inner_gate=float(z.inner_g.mean()),meta_gate=float(z.meta_g.mean()),outer_gate=float(z.outer_g.mean()),negative_to_positive_r_days=int(((z.inner_r<0)&(z.outer_r>0)).sum()),positive_to_negative_r_days=int(((z.inner_r>0)&(z.outer_r<0)).sum()),inner_wrong_down_days=int(((z.inner_d<0)&(z.inner_r>0)).sum()),outer_wrong_down_days=int(((z.outer_d<0)&(z.outer_r>0)).sum())))
    pd.concat(all_days,ignore_index=True).to_csv(H/'matched_days_v1.csv',index=False)
    with (H/'matched_summary_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_MATCHED_IDS_SCALAR_MEAN',weighting='same calendar-record day equal weight; inner/meta context means; not a like-for-like CV score',results=summaries),f,ensure_ascii=False,indent=2)
    print('MATCHED_DAY_CONTROL_PASS',flush=True)
if __name__=='__main__':main()
