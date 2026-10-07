from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
import numpy as np,pandas as pd
L=ROOT/'연구실/코덱스/local'/H.name
def main():
    e=pd.read_csv(H/'endpoints_v1.csv');s=pd.read_csv(H/'shapley_v1.csv');ii=pd.read_csv(H/'interactions_v1.csv');p=pd.read_csv(H/'profiles_v1.csv')
    output={}
    for pair in e.pair.unique():
        out={}
        for model in e[e.pair==pair].model.unique():
            ep=e[(e.pair==pair)&(e.model==model)].to_dict('records');ss=s[(s.pair==pair)&(s.model==model)].sort_values('smooth_phi',ascending=False).to_dict('records');it=ii[(ii.pair==pair)&(ii.model==model)&(ii.metric=='smooth')].copy();it['abs']=it.nonadditivity.abs();it=it.sort_values('abs',ascending=False).head(6).drop(columns='abs').to_dict('records')
            with np.load(L/f'coalition_{model}_{pair}.npz') as z:
                groups=z['groups'].tolist();v=z['smooth'];high=z['high_support'];full=len(v)-1;sw=[]
                for j,g in enumerate(groups):sw.append(dict(group=g,control_to_failure=float(v[1<<j]-v[0]),failure_to_control=float(v[full^(1<<j)]-v[full]),control_to_failure_high=float(high[1<<j]-high[0]),failure_to_control_high=float(high[full^(1<<j)]-high[full])))
            out[model]=dict(endpoints=ep,shapley=ss,largest_pair_nonadditivity=it,single_group_swaps=sw,support_failure_top=pd.read_csv(H/f'support_{model}_{pair}_failure.csv').head(8).to_dict('records'),support_control_top=pd.read_csv(H/f'support_{model}_{pair}_control.csv').head(5).to_dict('records'))
        output[pair]=out
    pathsummary=[]
    for fp in H.glob('paths_*.json'):
        d=json.loads(fp.read_text(encoding='utf-8'));rr=pd.DataFrame(d['records'])
        for hour in [0,6,12,23]:
            q=rr[rr.hour==hour];counts=q[q.diverged].groupby('group').size().sort_values(ascending=False)
            pathsummary.append(dict(model=q.model.iloc[0],pair=q.pair.iloc[0],hour=hour,denominator=600,no_divergence=int((~q.diverged).sum()),counts=counts.to_dict()))
    with (H/'summary_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(pairs=output,paths=pathsummary),f,ensure_ascii=False,indent=2,allow_nan=False)
    print(e.to_string(index=False))
    for pair in e.pair.unique():
        print('\n',pair);print(s[(s.pair==pair)&s.model.str.startswith('original')].sort_values('smooth_phi',ascending=False)[['group','smooth_phi','high_phi']].to_string(index=False))
if __name__=='__main__':main()
