from pathlib import Path
import csv,json,math,collections,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np
OUT=ROOT/'집/코덱스/local/temp_season_base_20261003_v1'
result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
groups=collections.defaultdict(list)
with (OUT/'oof.csv').open(encoding='utf-8',newline='') as f:
    for row in csv.DictReader(f):groups[(row['validator'],int(row['seed']),row['context'])].append(row)
checks=[];rng=np.random.default_rng(20261003)
for key,rows in sorted(groups.items()):
    assert len({r['row_id'] for r in rows})==len(rows)
    ea=[float(r['base'])-float(r['sub_temp']) for r in rows]
    eb=[float(r['candidate'])-float(r['sub_temp']) for r in rows]
    a=math.sqrt(math.fsum(x*x for x in ea)/len(rows));b=math.sqrt(math.fsum(x*x for x in eb)/len(rows))
    stored=next(s for s in result['summary'] if (s['validator'],s['seed'],s['context'])==key)
    assert abs(a-stored['baseline_rmse'])<1e-12 and abs(b-stored['candidate_rmse'])<1e-12
    check=dict(validator=key[0],seed=key[1],context=key[2],n=len(rows),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1))
    # Verify every blend against saved component predictions and OOF row identity.
    z=dict(np.load(Path(env.LOCAL)/'temp_mask_v1_oof.npz',allow_pickle=True))
    zi={str(r):i for i,r in enumerate(z['row_id'])}
    name=key[0];bs={726:7,727:101}[key[1]]
    p=np.load(Path(env.LOCAL)/f'web_tabpfn_{"v2" if key[2]=="1-8" else "v6"}_temp_{name}.npy')[:8].mean(0)
    foldcache={}
    maxblend=0.
    for r in rows:
        k=int(r['fold']);i=zi[r['row_id']]
        if k not in foldcache:
            c=dict(np.load(OUT/f'{name}_{k}.npz',allow_pickle=True));foldcache[k]=(c,{str(rr):j for j,rr in enumerate(c['row_id'])})
        c,ci=foldcache[k];j=ci[r['row_id']]
        t=float(r['in_temp']) if r['in_temp'] else math.nan;g=1. if math.isnan(t) else max(0.,min(1.,(t-8)/2))
        for tag,col in [('base','base'),('season','candidate')]:
            pred=math.fsum([(.4+.1*g)*float(c[f'{tag}_{key[1]}'][j]),(.6-.4*g)*float(z[f'{name}__CODEX__{key[1]}'][i]),.3*g*float(p[i])])
            maxblend=max(maxblend,abs(pred-float(r[col])))
    assert maxblend<1e-12;check['blend_maxdiff']=maxblend
    if name=='DIAG10':
        blocks=collections.defaultdict(list)
        for row,x,y in zip(rows,ea,eb):blocks[row['farm']+'_'+str(int(row['day'])//5)].append(y*y-x*x)
        sums=np.array([math.fsum(blocks[k]) for k in sorted(blocks)]);counts=np.array([len(blocks[k]) for k in sorted(blocks)])
        ix=rng.integers(0,len(sums),size=(20000,len(sums)));means=sums[ix].sum(1)/counts[ix].sum(1)
        pw=float(np.mean(means>=0));ci=np.quantile(means,[.0125,.9875]).tolist()
        assert pw==stored['p_worse'] and np.allclose(ci,stored['ci95'],atol=1e-12,rtol=0)
        check.update(p_worse=pw,ci95=ci)
    checks.append(check)
segments=[]
for key,rows in sorted(groups.items()):
    if key[0]!='DIAG10':continue
    for label,fn in [('late',lambda r:int(r['day'])>=179),('early',lambda r:int(r['day'])<179),('cold<=8',lambda r:bool(r['in_temp']) and float(r['in_temp'])<=8),('warm>=10',lambda r:bool(r['in_temp']) and float(r['in_temp'])>=10)]:
        sub=[r for r in rows if fn(r)]
        a=math.sqrt(math.fsum((float(r['base'])-float(r['sub_temp']))**2 for r in sub)/len(sub))
        b=math.sqrt(math.fsum((float(r['candidate'])-float(r['sub_temp']))**2 for r in sub)/len(sub))
        segments.append(dict(seed=key[1],context=key[2],segment=label,n=len(sub),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1)))
passrule=all(c['delta_pct']<0 for c in checks) and all(c['p_worse']<.0125 and c['ci95'][1]<0 for c in checks if c['validator']=='DIAG10')
assert ('PASS' if passrule else 'REJECT')==result['verdict']
out={'independent_arithmetic':'PASS','cells':checks,'segments':segments,'verdict':result['verdict'],'cache_maxdiff':result['baseline_cache_maxdiff']}
(HERE/'verification_base.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))


