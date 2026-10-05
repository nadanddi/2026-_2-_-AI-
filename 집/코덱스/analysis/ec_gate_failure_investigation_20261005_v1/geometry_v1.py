"""Exact attainable-output bounds, not a deployable oracle or new candidate."""
from pathlib import Path
import sys,math,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local'/H.name
def main():
    verified=json.loads((H/'verification_v1.json').read_text(encoding='utf-8'));assert verified['status']=='PASS_DIAGNOSIS_AND_META270';fs=sorted(OUT.glob('outer_actual_LR_DIAG10_*.csv'));assert len(fs)==30;allrows=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in fs],ignore_index=True);records=[];daily=[]
    for seed in [7,101,2024]:
        d=allrows[allrows.row_id.notna()].copy();d['seed_file_dummy']=0
        # file names identify seed; the spine deliberately contains no redundant seed column.
        rows=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in fs if p.stem.endswith('_'+str(seed))],ignore_index=True);assert len(rows)==8640 and rows.row_id.is_unique
        for name,mask in [('all',np.ones(len(rows),bool)),('high',rows.high.to_numpy()),('ordinary',~rows.high.to_numpy()),('pass2',(rows.day>=179).to_numpy())]:
            z=rows[mask];a,b,y=z.A.to_numpy(),z.B.to_numpy(),z.y.to_numpy();lo=np.minimum(a,b);hi=np.maximum(a,b);best=np.clip(y,lo,hi);positive_cap=a+np.clip(y-a,0,.3)
            scalarbest=[min(max(float(yy),min(float(aa),float(bb))),max(float(aa),float(bb))) for aa,bb,yy in zip(a,b,y)];assert np.max(abs(best-scalarbest))<1e-12;sa=math.fsum((float(aa)-float(yy))**2 for aa,yy in zip(a,y));sb=math.fsum((float(pp)-float(yy))**2 for pp,yy in zip(scalarbest,y));assert abs(sb-np.sum((best-y)**2))<1e-9
            records.append(dict(seed=seed,segment=name,n=len(z),days=len(z[['farm','day']].drop_duplicates()),baseline_rmse=math.sqrt(sa/len(z)),best_mix_rmse=math.sqrt(sb/len(z)),best_mix_change_pct=100*(math.sqrt(sb/sa)-1),unavoidable_sse_fraction=sb/sa,both_below_y=int((y>hi).sum()),both_above_y=int((y<lo).sum()),inside_hull=int(((y>=lo)&(y<=hi)).sum()),mean_residual=float(np.mean(y-a)),mean_delta=float(np.mean(b-a)),larger_than_positive_cap=int((y>a+.3).sum()),positive_cap_oracle_rmse=float(np.sqrt(np.mean((positive_cap-y)**2)))))
        for (farm,day),q in rows.groupby(['farm','day']):
            a,b,y=q.A.to_numpy(),q.B.to_numpy(),q.y.to_numpy();daily.append(dict(seed=seed,farm=farm,day=int(day),mean_y=float(y.mean()),mean_A=float(a.mean()),mean_B=float(b.mean()),mean_max_possible=float(np.maximum(a,b).mean()),mean_min_possible=float(np.minimum(a,b).mean()),unreachable_above_day_level=bool(y.mean()>np.maximum(a,b).mean()),high=bool(y.mean()>=1)))
    pd.DataFrame(daily).to_csv(H/'geometry_days_v1.csv',index=False)
    with (H/'geometry_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS_EXACT_CONVEX_BOUNDS_SCALAR',label='y-used attainable-output bounds only; no deployed rule or adoption',records=records),f,ensure_ascii=False,indent=2)
    print('CONVEX_GEOMETRY_PASS',flush=True)
if __name__=='__main__':main()
