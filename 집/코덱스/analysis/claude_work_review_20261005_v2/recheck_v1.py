"""Readonly public calculations; EL1 excluded before numeric parsing."""
import sys,csv,json,math,hashlib,collections
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/claude_work_review_20261005_v1'))
from recheck_public_v2 import rows,rmse,np,PUB,L,sha
import pandas as pd
from sklearn.metrics import roc_auc_score
pub={r['row_id']:float(r['sub_ec']) for r in rows(PUB) if r['validator']=='DIAG10' and r['seed']=='7'}
assert len(pub)==8640
out={'public_labels_sha':sha(PUB),'adoption':False,'EL1_recalculated':False}; frames={}
for name in ['AF0','AF0b']:
 p=L/f'ec3_{name}_all.csv'; rr=[]
 for r in rows(p):
  if r['validator']=='EL1':continue
  assert r['validator']=='DIAG10' and r['row_id'] in pub
  q={k:r[k] for k in ['row_id','farm','day','hour']};q['y']=float(r['sub_ec'])
  assert abs(q['y']-pub[r['row_id']])<1e-12
  for s in [7,101,2024]:
   for a in ['sg','af']+(['afb'] if name=='AF0b' else []):q[f'{a}_{s}']=float(r[f'{a}_{s}'])
  rr.append(q)
 assert len(rr)==1104 and len({r['row_id'] for r in rr})==1104
 df=pd.DataFrame(rr); df['dm']=df.groupby(['farm','day']).y.transform('mean')
 assert len(df.groupby(['farm','day']))==46
 sc=[]; col='af' if name=='AF0' else 'afb'
 for s in [7,101,2024]:
  for segment,mask in [('all',df.dm>=0),('ordinary',df.dm<1),('high',df.dm>=1)]:
   g=df[mask]; a=rmse(g[f'sg_{s}'].tolist(),g.y.tolist());b=rmse(g[f'{col}_{s}'].tolist(),g.y.tolist())
   sc.append(dict(seed=s,segment=segment,rows=len(g),baseline=a,candidate=b,change_pct=100*(b/a-1)))
 frames[name]=df;out[name]={'csv_sha':sha(p),'scores':sc}
a=frames['AF0'].set_index('row_id');b=frames['AF0b'].set_index('row_id')
for c in ['y']+[f'{p}_{s}' for p in ['sg','af'] for s in [7,101,2024]]:assert np.max(abs(a[c]-b.loc[a.index,c]))<1e-12
out['AF0b']['cached_columns_match']=True
df=frames['AF0b'].copy();df['baseline']=df[[f'sg_{s}' for s in [7,101,2024]]].mean(axis=1);df['candidate']=df[[f'afb_{s}' for s in [7,101,2024]]].mean(axis=1)
df['delta']=(df.candidate-df.y)**2-(df.baseline-df.y)**2
days=[]
for (farm,day),g in df[df.dm<1].groupby(['farm','day']):
 delta=math.fsum((float(c)-float(y))**2-(float(a)-float(y))**2 for a,c,y in zip(g.baseline,g.candidate,g.y))
 assert abs(delta-g.delta.sum())<1e-12
 days.append(dict(farm=farm,day=int(day),y_mean=float(g.y.mean()),baseline_mean=float(g.baseline.mean()),candidate_mean=float(g.candidate.mean()),delta_sse=delta))
days.sort(key=lambda r:r['delta_sse'],reverse=True); net=math.fsum(r['delta_sse'] for r in days);pos=math.fsum(max(0,r['delta_sse']) for r in days);top=math.fsum(r['delta_sse'] for r in days[:3])
assert {(r['farm'],r['day']) for r in days[:3]}=={('F13',231),('F13',233),('F47',216)}
out['AF0b']['ordinary_concentration']={'days':len(days),'top3':days[:3],'net_delta_sse':net,'positive_delta_sse':pos,'top3_signed_net_share':top/net,'top3_positive_harm_share':top/pos,'all_days':days}
p=L/'hk1_rows_v1.csv';rr=[]
for r in rows(p):
 assert r['validator']=='DIAG10' and r['row_id'] in pub
 q={k:r[k] for k in ['row_id','farm','day']};q['y']=float(r['sub_ec']);assert abs(q['y']-pub[r['row_id']])<1e-12
 for c in ['a1','a2','pm','d1','d2','dcal','hi5','m5','dgap','hour']:q[c]=float(r[c]) if r[c] else math.nan
 rr.append(q)
assert len(rr)==8640 and len({r['row_id'] for r in rr})==8640
d=pd.DataFrame(rr);d['dm']=d.groupby(['farm','day']).y.transform('mean');g=d[(d.pm>=.8)&(d.a1>=1)].copy()
g['d21']=g.d2-g.d1;g['a12']=(g.a1-g.a2).abs();g['apm']=g.a1-g.pm
features=['d1','d2','d21','dcal','a12','pm','apm','hi5','m5','dgap','hour']
D=g.groupby(['farm','day']).agg({**{c:'mean' for c in features},'dm':'first'}).reset_index();D['first_h']=g.groupby(['farm','day']).hour.min().values; features+=['first_h']; lbl=(D.dm<1).astype(int).values
aucs={};rankvectors=[]
for c in features:
 v=D[c].fillna(D[c].median()); auc=float(roc_auc_score(lbl,v)); ranks=v.rank(method='average').values
 manual=(math.fsum(ranks[lbl==1])-sum(lbl)*(sum(lbl)+1)/2)/(sum(lbl)*sum(lbl==0));assert abs(auc-manual)<1e-12
 aucs[c]=auc;rankvectors.append(ranks)
rankvectors=np.array(rankvectors);n1=sum(lbl);n0=len(lbl)-n1;obs=max(abs(v-.5) for v in aucs.values());rng=np.random.default_rng(20261005);count=0
for _ in range(10000):
 z=lbl.copy()
 for f in ['F13','F47']:
  mm=(D.farm==f).values;z[mm]=rng.permutation(z[mm])
 aa=(rankvectors[:,z==1].sum(axis=1)-n1*(n1+1)/2)/(n1*n0);count+=bool(np.max(abs(aa-.5))>=obs-1e-15)
out['HK1']={'csv_sha':sha(p),'gated_days':len(D),'true_high':int(n0),'false':int(n1),'aucs':aucs,'permutation_count':count,'p_familywise':(count+1)/10001,'false_days':D[lbl==1][['farm','day','dm','d1','d2']].to_dict('records')}
(H/'public_recheck_v1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='public_labels_sha'},ensure_ascii=False))
