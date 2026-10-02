"""Diagnostic only: exact convex interval under TGATE's preregistered bounds."""
from support import *
import math
def extremum(pred,lower,upper,reverse=False):
    w=lower.copy();remaining=1-float(w.sum())
    for j in sorted(range(3),key=lambda j:float(pred[j]),reverse=reverse):
        add=min(remaining,float(upper[j]-w[j]));w[j]+=add;remaining-=add
    assert abs(remaining)<1e-12
    return math.fsum(float(a)*float(b) for a,b in zip(pred,w))
def main():
    lab,_,_,_,_,outer=loadtemp();records=[]
    for seed in TSEEDS:
        for context in ['1-8','17-24']:
            d=outer[(outer.validator=='DIAG10')&(outer.base_seed==seed)&(outer.context==context)];ref=d[d.member=='W30G'].set_index('row_id');meta=lab.set_index('row_id').loc[ref.index];p=np.column_stack([d[d.member==m].set_index('row_id').prediction.reindex(ref.index) for m in ['BASE','CODEX','PFN']]);w=weights(meta);g=gate(meta);lower=np.maximum(0,w-.1*g[:,None]);upper=np.minimum(1,w+.1*g[:,None]);mins=np.array([extremum(a,b,c) for a,b,c in zip(p,lower,upper)]);maxs=np.array([extremum(a,b,c,True) for a,b,c in zip(p,lower,upper)]);y=ref.sub_temp.to_numpy();distance=np.maximum(np.maximum(mins-y,y-maxs),0);sse=(ref.prediction.to_numpy()-y)**2
            for seg,mask in [('all',np.ones(len(y),bool)),('late',ref.day.to_numpy()>=179),('F47_late',(ref.farm.to_numpy()=='F47')&(ref.day.to_numpy()>=179))]:
                floor=math.fsum(float(v)**2 for v in distance[mask]);actual=math.fsum(float(v) for v in sse[mask]);records.append(dict(seed=seed,context=context,segment=seg,n=int(mask.sum()),minimum_remaining_sse_pct=100*floor/actual,oracle_rmse=math.sqrt(floor/int(mask.sum()))))
    savej(HERE/'oracle_limits.json',dict(scope='diagnostic target-using oracle, not an achievable candidate',bounds='old weights +/- .1*gate, simplex, cold protection',records=records,source_hash=sha(__file__)));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
