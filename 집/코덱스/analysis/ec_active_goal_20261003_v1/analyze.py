"""Public-only full-run verifier for precommitted ET replacement experiments."""
from pathlib import Path
import sys,json,math,hashlib,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'))
import support as S
import pandas as pd,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True);parser.add_argument('--family',type=int,required=True);arg=parser.parse_args()
EH=ROOT/'집/코덱스/analysis'/arg.experiment;OUT=ROOT/'집/코덱스/local'/arg.experiment
assert arg.family>=4
paths=list(OUT.glob('*.csv'));assert len(paths)==66,('incomplete',len(paths))
o=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)
assert len(o)==83160 and not o.duplicated(['validator','fold','seed','row_id']).any()
checks=0;score=[]
for (v,s),g in o.groupby(['validator','seed']):
    r=[]
    for c in ['baseline','candidate']:
        value=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g[c]))/len(g))
        assert abs(value-np.sqrt(np.mean((g.y-g[c])**2)))<1e-12;checks+=1;r.append(value)
    score.append(dict(validator=v,seed=int(s),n=len(g),baseline=r[0],candidate=r[1],change_pct=100*(r[1]/r[0]-1)))
sc=pd.DataFrame(score)
lab,core,wv,folds,outer=S.loadec()
bounds={};split_audit=[]
for v,k,tm,vm in folds:
    tr=lab[tm];va=lab[vm];tk=set(zip(tr.farm,tr.day));vk=set(zip(va.farm,va.day))
    assert not tk&vk
    assert all(not(f==ff and abs(d-q)<=1) for f,d in tk for ff,q in vk)
    bounds[(v,k)]=(float(tr.sub_ec.min()),float(tr.sub_ec.max()));checks+=2
    split_audit.append(dict(validator=v,fold=k,train_days=len(tk),valid_days=len(vk)))
maxdiff=0
for (v,k,s,f,d),g in o.groupby(['validator','fold','seed','farm','day']):
    lo,hi=bounds[(v,k)];g=g.sort_values('hour');prior=[]
    for r in g.itertuples():
        prior.append(r.new_et_raw);shrink=.5*r.new_et_raw+.5*math.fsum(prior)/len(prior)
        expected=min(hi,max(lo,r.baseline+.48*(shrink-r.old_et_shrunk)))
        delta=abs(expected-r.candidate);assert delta<1e-12;maxdiff=max(maxdiff,delta);checks+=1
    base=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==s)].set_index('row_id').season_v2.reindex(g.row_id).to_numpy()
    assert np.array_equal(base,g.baseline.to_numpy());checks+=1
boot={};alpha=.025/arg.family
for seed in [7,101,2024]:
    d=o[(o.validator=='DIAG10')&(o.seed==seed)];manual={}
    for r in d.itertuples():manual.setdefault((r.farm,r.day),[]).append((r.y-r.candidate)**2-(r.y-r.baseline)**2)
    rng=np.random.default_rng(20261003+seed);sm=np.zeros(20000);nc=np.zeros(20000)
    for f in ['F13','F47']:
        days=sorted(q for ff,q in manual if ff==f);a=[];n=[]
        for i in range(0,len(days),5):
            vals=[val for q in days[i:i+5] for val in manual[(f,q)]];a.append(math.fsum(vals));n.append(len(vals))
        a=np.array(a);n=np.array(n);ix=rng.integers(len(a),size=(20000,len(a)));sm+=a[ix].sum(1);nc+=n[ix].sum(1)
    sample=sm/nc
    boot[seed]=dict(p_worse=float(np.mean(sample>=0)),ci=np.quantile(sample,[alpha,1-alpha]).tolist())
    # Pandas grouping independently, same block definition and random stream.
    grouped=d.assign(diff=(d.y-d.candidate)**2-(d.y-d.baseline)**2).groupby(['farm','day'])['diff'].agg(['sum','count'])
    rng=np.random.default_rng(20261003+seed);ss=np.zeros(20000);nn=np.zeros(20000)
    for f in ['F13','F47']:
        g=grouped.loc[f].sort_index();a=np.array([g['sum'].iloc[i:i+5].sum() for i in range(0,len(g),5)]);n=np.array([g['count'].iloc[i:i+5].sum() for i in range(0,len(g),5)])
        ix=rng.integers(len(a),size=(20000,len(a)));ss+=a[ix].sum(1);nn+=n[ix].sum(1)
    assert np.max(np.abs(ss/nn-sample))<1e-12;checks+=1
segments=[]
for s in [7,101,2024]:
    d=o[(o.validator=='DIAG10')&(o.seed==s)];hi=d.groupby(['farm','day']).y.transform('mean')>=1
    for label,m in [('high',hi),('ordinary',~hi),('late',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47')]:
        g=d[m];segments.append(dict(seed=s,segment=label,n=len(g),days=len(g[['farm','day']].drop_duplicates()),baseline=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.baseline))/len(g)),candidate=math.sqrt(math.fsum((float(y)-float(p))**2 for y,p in zip(g.y,g.candidate))/len(g)),baseline_bias=float((g.baseline-g.y).mean()),candidate_bias=float((g.candidate-g.y).mean())))
passed=bool((sc.change_pct<0).all() and all(b['p_worse']<alpha and b['ci'][1]<0 for b in boot.values()))
sc.to_csv(EH/'scores_v1.csv',index=False);pd.DataFrame(segments).to_csv(EH/'segments_v1.csv',index=False)
result=dict(status='PASS',decision='PUBLIC_PASS_PENDING_EL1' if passed else 'REJECT',public_pass=passed,family=arg.family,alpha=alpha,bootstrap=boot,checks=checks,rows=len(o),scalar_clipped_maxdiff=maxdiff,split_audit=split_audit,source_sha256=hashlib.sha256((EH/'run.py').read_bytes()).hexdigest(),analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
assert not (EH/'result_v1.json').exists()
(EH/'result_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(sc.to_string(index=False));print(json.dumps(result,ensure_ascii=False,indent=2))
