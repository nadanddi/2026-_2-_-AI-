from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import pandas as pd,numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'공용/대회자료/정형데이터/참가자_배포/train_X.csv';raw=pd.read_csv(source)
    raw['farm']=raw.row_id.str.split('_').str[0];raw['day']=raw.row_id.str.split('_').str[1].astype(int);raw['hour']=raw.row_id.str.split('_').str[2].astype(int)
    prep=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'));keys=set((p['farm'],d) for p in prep['pairs'] for d in [p['day'],p['control']])
    for fp in H.glob('support_original_*_failure.csv'):
        q=pd.read_csv(fp).head(5);keys|=set(q[['farm','day']].itertuples(index=False,name=None))
    cols=['in_temp','in_hum','in_co2','act_vent','act_heating','act_circfan','act_shade','act_thermal','act_co2','act_fog'];out=[]
    for farm,day in sorted(keys):
        q=raw[(raw.farm==farm)&(raw.day==day)].sort_values('hour');assert len(q)==24
        r=dict(farm=farm,day=day)
        for c in cols:r[c+'_h0']=None if pd.isna(q[c].iloc[0]) else float(q[c].iloc[0]);r[c+'_mean']=float(q[c].mean());r[c+'_missing']=int(q[c].isna().sum())
        r['fan_zero_fraction']=float((q.act_circfan==0).mean());r['vent_zero_fraction']=float((q.act_vent==0).mean());out.append(r)
    pd.DataFrame(out).to_csv(H/'raw_input_comparison_v1.csv',index=False)
    coverage=[];intersection=None
    for k in [1,4,8]:
        with np.load(ROOT/f'연구실/코덱스/local/ec_current14_influence_20261007_v1/base/{k}_7.npz') as z:ids=set(z['train_row_id'].tolist())
        intersection=ids if intersection is None else intersection&ids
    for label in ['original_1_7','original_4_7','original_8_7','common_7']:
        with np.load(L/f'{label}.npz') as z:f=pd.DataFrame(dict(row_id=z['train_row_id'],y=z['train_y']))
        f['farm']=f.row_id.str.split('_').str[0];f['day']=f.row_id.str.split('_').str[1].astype(int);days=f.groupby(['farm','day']).y.mean().reset_index()
        for farm in ['F13','F47']:
            for pass2 in [False,True]:
                q=days[(days.farm==farm)&((days.day>=179)==pass2)];coverage.append(dict(model=label,farm=farm,pass2=pass2,days=len(q),high_days=int((q.y>=1).sum())))
    pd.DataFrame(coverage).to_csv(H/'training_coverage_v1.csv',index=False)
    with (H/'input_evidence_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(raw_sha=sha(source),source_sha=sha(Path(__file__)),intersection_days=len(intersection)//24,common_days=len(prep['common_train_ids'])//24,raw_profiles=len(out)),f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
