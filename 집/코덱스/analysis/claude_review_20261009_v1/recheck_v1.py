from pathlib import Path
import csv,math,json,random
ROOT=Path.cwd();out=Path(__file__).resolve().parent
res={}
for name,seeds,arms in [('ct1',(2121,4343,6565),('REF','THS')),('ns1',(1717,3939,5757),('BASE','NS'))]:
 rows=[];files=list((ROOT/'집/클로드/research/local'/f'{name}_ckpt').glob('*.csv'))
 for p in files:
  with p.open(encoding='utf8',newline='') as f:rows.extend(csv.DictReader(f))
 scores=[]
 for v in ['DIAG10','A','B','EL1','P2LOO']:
  rr=[x for x in rows if x['validator']==v and (v not in ['EL1','P2LOO'] or int(x['day'])>=179)];errs={};seedrm={}
  for a in arms:
   ee=[];ss=[[] for s in seeds]
   for x in rr:
    ps=[min(float(x['hi']),max(float(x['lo']),math.fsum(w*float(x[f'{a}_{kind}_{s}']) for kind,w in [('et',.6),('lgb',.3),('mlp',.1)]))) for s in seeds]
    ee.append(math.fsum(ps)/len(ps)-float(x['sub_ec']))
    for j,p in enumerate(ps):ss[j].append(p-float(x['sub_ec']))
   errs[a]=ee;seedrm[a]=[math.sqrt(math.fsum(e*e for e in es)/len(es)) for es in ss]
  rm={a:math.sqrt(math.fsum(e*e for e in errs[a])/len(rr)) for a in arms}
  nr={a:math.sqrt(sum(e*e for e in errs[a])/len(rr)) for a in arms};gap=max(abs(rm[a]-nr[a]) for a in arms);assert gap<1e-12
  z=dict(validator=v,rows=len(rr),days=len({(x['farm'],x['day']) for x in rr}),rmse=rm,pct=100*(rm[arms[1]]/rm[arms[0]]-1),seeds_better=sum(b<a for a,b in zip(seedrm[arms[0]],seedrm[arms[1]])),independent_maxdiff=gap);scores.append(z)
  if v=='DIAG10':
   keys=sorted({(x['farm'],int(x['day'])//5) for x in rr});groups={k:[[],[]] for k in keys}
   for i,x in enumerate(rr):
    g=groups[(x['farm'],int(x['day'])//5)]
    for j,a in enumerate(arms):g[j].append(errs[a][i]**2)
   delta=[math.fsum(groups[k][1])-math.fsum(groups[k][0]) for k in keys];rng=random.Random(20261009);worse=0
   for _ in range(20000):worse+=math.fsum(delta[rng.randrange(len(keys))] for j in keys)>=0
   z['independent_p_worse']=worse/20000;z['bootstrap_note']='Independent stdlib RNG, 20000 farm x relative-day//5 resamples; not identical RNG to author.'
 res[name]=dict(files=len(files),scores=scores)
res['ct2_saved_folds_at_review']=[p.stem for p in (ROOT/'집/클로드/research/local/ct2_ckpt').glob('*.csv')]
(out/'independent_metrics_v1.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(res,ensure_ascii=True))
