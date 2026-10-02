from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np,pandas as pd
from anal_q1_errors import diag_folds
from common import split_mask,USABLE
def main():
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
    raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int)
    y=pd.read_csv(Path(env.DATA)/'train_y.csv',usecols=['row_id','sub_temp']);raw=raw.merge(y,on='row_id',validate='one_to_one').sort_values(['farm','day','hour']).reset_index(drop=True)
    assert len(raw)==9600 and raw.row_id.is_unique
    groups=list(raw.groupby(['farm','day'],sort=True));keys=[key for key,g in groups];W=['out_temp','out_hum','out_rad','out_wspd']
    weather=np.stack([g[W].to_numpy(float).T for key,g in groups]);mu=np.nanmean(weather,axis=(0,2));sd=np.nanstd(weather,axis=(0,2));sd[sd==0]=1
    norm=(weather-mu[None,:,None])/sd[None,:,None]
    distance=np.empty((len(keys),len(keys)))
    for i in range(len(keys)):distance[i]=np.sqrt(np.nanmean((norm[i]-norm)**2,axis=(1,2)))
    edge=distance<=.05;np.fill_diagonal(edge,False)
    parents=list(range(len(keys)))
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    for i,j in zip(*np.where(np.triu(edge,1))):parents[find(int(j))]=find(int(i))
    cid=np.array([find(i) for i in range(len(keys))]);label=pd.DataFrame(keys,columns=['farm','day']);label['weather_group']=cid
    label.to_csv(HERE/'weather_groups.csv',index=False)
    summaries=[]
    for k,fd in enumerate(diag_folds(raw)):
        tm,vm=split_mask(raw,fd);td=set(map(tuple,raw.loc[tm,['farm','day']].drop_duplicates().to_numpy()));vd=set(map(tuple,raw.loc[vm,['farm','day']].drop_duplicates().to_numpy()))
        ti=[i for i,key in enumerate(keys) if key in td];vi=[i for i,key in enumerate(keys) if key in vd];vgroup=set(cid[vi]);exclude=[i for i in ti if cid[i] in vgroup]
        exposed=[i for i in vi if any(cid[i]==cid[j] for j in ti)]
        summaries.append(dict(fold=k,train_days=len(ti),validation_days=len(vi),validation_weather_shared_train=len(exposed),weather_guard_extra_train_days_removed=len(exclude)))
    identical_weather_pairs=0;identical_inputs_pairs=0;identical_labels_pairs=0
    for i,j in zip(*np.where(np.triu(edge,1))):
        gi,gj=groups[int(i)][1],groups[int(j)][1]
        if np.array_equal(gi[W].to_numpy(),gj[W].to_numpy(),equal_nan=True):identical_weather_pairs+=1
        if np.array_equal(gi[USABLE].to_numpy(),gj[USABLE].to_numpy(),equal_nan=True):identical_inputs_pairs+=1
        if np.array_equal(gi.sub_temp.to_numpy(),gj.sub_temp.to_numpy()):identical_labels_pairs+=1
    result=dict(rows=len(raw),days=len(keys),weather_groups=len(set(cid)),near_weather_pairs=int(np.triu(edge,1).sum()),exact_weather_pairs=identical_weather_pairs,exact_full_input_pairs=identical_inputs_pairs,exact_temperature_label_pairs=identical_labels_pairs,threshold=.05,weather_standardization='all provided TRAIN inputs, split audit/group definition only, no model transform',folds=summaries,interpretation='Shared exogenous weather is not evidence of label leakage; guard is a dependence stress test. Exact input/target copies checked separately.')
    (HERE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
