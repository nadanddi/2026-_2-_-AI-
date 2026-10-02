from pathlib import Path
import csv,json,math,collections,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=ROOT/'집/코덱스/local/temp_conditional_gate_20261003_v1'
def score(e):return math.sqrt(math.fsum(float(x)**2 for x in e)/len(e))
def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'));groups=collections.defaultdict(list)
    with (OUT/'oof.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):groups[(r['scope'],r['member'],r['validator'],int(r['seed']),r['context'])].append(r)
    with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:labels={r['row_id']:float(r['sub_temp']) for r in csv.DictReader(f)}
    z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True));zi={str(r):i for i,r in enumerate(z['row_id'])};rng=np.random.default_rng(20261003);cells=[];segments=[];cache={}
    for key,rows in sorted(groups.items()):
        scope,tag,name,seed,context=key;assert len({r['row_id'] for r in rows})==len(rows);assert all(float(r['sub_temp'])==labels[r['row_id']] for r in rows)
        ea=[float(r['base'])-float(r['sub_temp']) for r in rows];eb=[float(r['candidate'])-float(r['sub_temp']) for r in rows];a,b=score(ea),score(eb)
        stored=next(s for s in result['summary'] if (s['scope'],s['member'],s['validator'],s['seed'],s['context'])==key);assert abs(a-stored['baseline_rmse'])<1e-12 and abs(b-stored['candidate_rmse'])<1e-12
        item=dict(scope=scope,member=tag,validator=name,seed=seed,context=context,n=len(rows),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1));maxdiff=0.
        if scope=='W30G':pfn=np.load(Path(env.LOCAL)/f'web_tabpfn_{"v2" if context=="1-8" else "v6"}_temp_{name}.npy')[:8].mean(0)
        for r in rows:
            k=int(r['fold']);ck=(name,k)
            if ck not in cache:
                c=dict(np.load(OUT/f'{name}_{k}.npz',allow_pickle=True));cache[ck]=(c,{str(rr):j for j,rr in enumerate(c['row_id'])})
            c,ci=cache[ck];j=ci[r['row_id']];new=float(c[f'{tag}_{seed}'][j])
            if scope=='member_only':ref=float(c[f'CODEX_{seed}'][j]);cand=new
            else:
                i=zi[r['row_id']];bs={726:7,727:101}[seed];base=float(z[f'{name}__MASK__{bs}'][i]);cx=float(z[f'{name}__CODEX__{seed}'][i]);t=float(r['in_temp']) if r['in_temp'] else math.nan;g=1. if math.isnan(t) else max(0.,min(1.,(t-8)/2));ref=math.fsum([(.4+.1*g)*base,(.6-.4*g)*cx,.3*g*float(pfn[i])]);cand=math.fsum([(.4+.1*g)*base,(.6-.4*g)*new,.3*g*float(pfn[i])])
            maxdiff=max(maxdiff,abs(ref-float(r['base'])),abs(cand-float(r['candidate'])))
        assert maxdiff<1e-12;item['blend_maxdiff']=maxdiff
        if name in ['DIAG10','GUARD']:
            blocks=collections.defaultdict(list)
            for r,x,y in zip(rows,ea,eb):blocks[r['farm']+'_'+str(int(r['day'])//5)].append(y*y-x*x)
            sums=np.array([math.fsum(blocks[k]) for k in sorted(blocks)]);counts=np.array([len(blocks[k]) for k in sorted(blocks)]);ix=rng.integers(0,len(sums),size=(20000,len(sums)));values=sums[ix].sum(1)/counts[ix].sum(1);pw=float(np.mean(values>=0));ci=np.quantile(values,[.025/16,1-.025/16]).tolist();assert pw==stored['p_worse'] and np.allclose(ci,stored['ci'],rtol=0,atol=1e-12);item.update(p_worse=pw,ci=ci)
            predicates={'late':lambda r:int(r['day'])>=179,'early':lambda r:int(r['day'])<179,'F13':lambda r:r['farm']=='F13','F47':lambda r:r['farm']=='F47','hour00_05':lambda r:int(r['hour'])<6,'hour06_17':lambda r:6<=int(r['hour'])<18,'hour18_23':lambda r:int(r['hour'])>=18}
            for segment,fn in predicates.items():
                ids=[i for i,r in enumerate(rows) if fn(r)];sa,sb=score([ea[i] for i in ids]),score([eb[i] for i in ids]);segments.append(dict(scope=scope,member=tag,seed=seed,context=context,segment=segment,n=len(ids),baseline_rmse=sa,candidate_rmse=sb,delta_pct=100*(sb/sa-1)))
            days=collections.defaultdict(list)
            for r,x,y in zip(rows,ea,eb):days[(r['farm'],int(r['day']))].append((x,y))
            l0=math.fsum(len(v)*(math.fsum(x for x,y in v)/len(v))**2 for v in days.values());l1=math.fsum(len(v)*(math.fsum(y for x,y in v)/len(v))**2 for v in days.values());s0=math.fsum(x*x for x in ea);s1=math.fsum(y*y for y in eb)
            segments.append(dict(scope=scope,member=tag,seed=seed,context=context,segment='level_shape',baseline_level_rmse=math.sqrt(l0/len(rows)),candidate_level_rmse=math.sqrt(l1/len(rows)),baseline_shape_rmse=math.sqrt((s0-l0)/len(rows)),candidate_shape_rmse=math.sqrt((s1-l1)/len(rows))))
        cells.append(item)
    verdicts={tag:('PASS_REQUIRES_FULL_GUARD' if all(s['delta_pct']<0 for s in cells if s['scope']=='W30G' and s['member']==tag) and all(s['p_worse']<.025/16 and s['ci'][1]<0 for s in cells if s['scope']=='W30G' and s['member']==tag and s['validator']=='DIAG10') else 'REJECT') for tag in ['G1']};assert verdicts==result['verdicts']

    answer=dict(status='PASS',verdicts=verdicts,cells=cells,segments=segments)
    (HERE/'verification.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2),encoding='utf-8');(HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','verdicts':verdicts}))
if __name__=='__main__':main()
