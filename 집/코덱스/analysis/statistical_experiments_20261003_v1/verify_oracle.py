from support import *
import itertools,math
def main():
    original=json.loads((HERE/'oracle_limits.json').read_text(encoding='utf-8'));lab,_,_,_,_,outer=loadtemp();records=[]
    for seed in TSEEDS:
        for context in ['1-8','17-24']:
            d=outer[(outer.validator=='DIAG10')&(outer.base_seed==seed)&(outer.context==context)];ref=d[d.member=='W30G'].set_index('row_id');meta=lab.set_index('row_id').loc[ref.index];pred=np.column_stack([d[d.member==m].set_index('row_id').prediction.reindex(ref.index) for m in ['BASE','CODEX','PFN']]);w=weights(meta);g=gate(meta);lower=np.maximum(0,w-.1*g[:,None]);upper=np.minimum(1,w+.1*g[:,None]);errors=[]
            for p,lo,hi,truth in zip(pred,lower,upper,ref.sub_temp):
                vertices=[]
                for pair in itertools.combinations(range(3),2):
                    other=next(j for j in range(3) if j not in pair)
                    for side in itertools.product([0,1],repeat=2):
                        v=np.zeros(3)
                        for j,up in zip(pair,side):v[j]=hi[j] if up else lo[j]
                        v[other]=1-v.sum()
                        if np.all(v>=lo-1e-12) and np.all(v<=hi+1e-12):vertices.append(math.fsum(float(a)*float(b) for a,b in zip(p,v)))
                assert vertices;distance=max(min(vertices)-float(truth),float(truth)-max(vertices),0.);errors.append(distance*distance)
            for seg,mask in [('all',np.ones(len(ref),bool)),('late',ref.day.to_numpy()>=179),('F47_late',(ref.farm.to_numpy()=='F47')&(ref.day.to_numpy()>=179))]:
                floor=math.fsum(e for e,m in zip(errors,mask) if m);den=math.fsum((float(p)-float(y))**2 for p,y,m in zip(ref.prediction,ref.sub_temp,mask) if m);fraction=100*floor/den;saved=next(r for r in original['records'] if r['seed']==seed and r['context']==context and r['segment']==seg);assert abs(fraction-saved['minimum_remaining_sse_pct'])<1e-10;records.append(dict(seed=seed,context=context,segment=seg,minimum_remaining_sse_pct=fraction))
    savej(HERE/'oracle_verification.json',dict(status='PASS',method='enumerate bounded-simplex vertices vs greedy mass allocation',cells=len(records),records=records));print('ORACLE_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
