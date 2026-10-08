"""Independent scalar structure, EWMA/OLS, role AUC and bias verification. No fits of predictive models."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import csv,json,math,statistics,hashlib
from collections import defaultdict
DATA=Path(env.DATA);F=H/'first_v4';R=H/'rest_v4';C=H/'role_v2'
checks=0;maxerr=0
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):
    global checks
    checks+=1;assert a==b,(a,b)
def near(a,b):
    global checks,maxerr
    checks+=1
    if a is None or b=='':assert a is None and b in ('',None);return
    d=abs(float(a)-float(b));maxerr=max(maxerr,d)
    assert d<=2e-10*max(1,abs(float(a)),abs(float(b))),(a,b,d)
def mean(v):return math.fsum(v)/len(v)
def corr(a,b):
    if len(a)<3:return None
    am,bm=mean(a),mean(b);da=[x-am for x in a];db=[x-bm for x in b]
    va,vb=math.fsum(x*x for x in da),math.fsum(x*x for x in db)
    return math.fsum(x*y for x,y in zip(da,db))/math.sqrt(va*vb) if va and vb else None
def rank(a):
    order=sorted(range(len(a)),key=a.__getitem__);out=[None]*len(a);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and a[order[j]]==a[order[i]]:j+=1
        for k in range(i,j):out[order[k]]=(i+j+1)/2
        i=j
    return out
def rho(a,b):return corr(rank(a),rank(b))
def key(r):return (r['farm'],int(r['day']))
def finite(x):return x is not None and math.isfinite(x)
def num(x):return float(x) if x else None
def nanmean(v):return mean([x for x in v if finite(x)])

for folder in [F,R,C]:
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for n,h in manifest['output_sha'].items():eq(sha(folder/n),h)
    eq(sha(H/('role_diagnostic_v2.py' if folder==C else 'audit_v4.py')),manifest['source_sha'])

prep=json.loads((ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1/preparation_v4.json').read_text(encoding='utf-8'))
allowed={i for f in prep['records'] for i in f['query_ids']};eq(len(allowed),8640)
y={}
for row in read(DATA/'train_y.csv'):
    if row['row_id'] not in allowed:continue
    y[row['row_id']]=(float(row['sub_ec']),float(row['sub_temp']))
raw=defaultdict(dict)
for row in read(DATA/'train_X.csv'):
    f,d,h=row['row_id'].split('_')
    if f not in {'F13','F47'}:continue
    raw[(f,int(d))][int(h)]=row
ds={key(r):r for r in read(F/'day_inputs.csv') if r['group']=='train_inputs'}
pub={k for k,r in ds.items() if r['public']=='True'}
daily={}
for k in pub:
    ids=[raw[k][h]['row_id'] for h in range(24)]
    ec=[y[i][0] for i in ids];st=[y[i][1] for i in ids]
    it=[num(raw[k][h]['in_temp']) for h in range(24)]
    # Scalar pandas adjust=False / ignore_na=False EWMA recurrence.
    alpha=1-.5**.25;beta=1-alpha;weighted=None;old_weight=1.;ew=[]
    for value in it:
        if weighted is None:
            if value is not None:weighted=value
        else:
            old_weight*=beta
            if value is not None:
                weighted=(old_weight*weighted+alpha*value)/(old_weight+alpha);old_weight=1.
        ew.append(weighted)
    hh=[h for h in range(3,24) if finite(it[h]) and finite(ew[h])]
    xx=[ew[h] for h in hh];yy=[st[h] for h in hh]
    near(corr(xx,yy),ds[k]['K1'])
    if len(xx)>8 and len(set(xx))>1:
        xm,ym=mean(xx),mean(yy);slope=math.fsum((a-xm)*(b-ym) for a,b in zip(xx,yy))/math.fsum((a-xm)**2 for a in xx)
        resid=math.sqrt(mean([(b-(ym+slope*(a-xm)))**2 for a,b in zip(xx,yy)]));near(resid,ds[k]['K2'])
    daily[k]=dict(ec=mean(ec),temp=mean(st),it=nanmean(it))

weather=defaultdict(list)
for k,v in raw.items():
    eq(set(v),set(range(24)))
    sig=tuple(float(v[h][c]) for h in range(24) for c in ['out_temp','out_hum','out_rad','out_wspd'])
    assert all(map(math.isfinite,sig));weather[sig].append(k)
structure=json.loads((R/'structure.json').read_text(encoding='utf-8'))
eq(len(raw),structure['weather_complete_days']);eq(len(weather),structure['groups'])
roles={};groups={}
for r in read(R/'roles.csv'):
    k=key(r);roles[k]=r
    groups.setdefault(int(r['gid']),[]).append(k)
for gid,kk in groups.items():
    for k in kk:
        sig=tuple(float(raw[k][h][c]) for h in range(24) for c in ['out_temp','out_hum','out_rad','out_wspd'])
        mates=sorted(z for z in weather[sig] if z[0]==k[0]);eq(len(mates),2);eq(mates[1][1]-mates[0][1],1)
        eq(roles[k]['role'],'A' if k==mates[0] else 'B')
pairs=read(R/'pairs.csv')
for r in pairs:
    a=(r['farm'],int(r['dayA']));b=(r['farm'],int(r['dayB']))
    eq(b[1]-a[1],1)
    for col,val in [('ecA',daily[a]['ec']),('ecB',daily[b]['ec']),('dt',daily[b]['it']-daily[a]['it']),('dst',daily[b]['temp']-daily[a]['temp'])]:near(val,r[col])
eq(len(pairs),structure['public_pairs'])
xx=[float(r['dt']) for r in pairs];yy=[float(r['dst']) for r in pairs]
near(corr(xx,yy)**2,structure['pair_temp_relation']['ols_r2'])

cross=read(R/'cross_farm.csv')
expected=sorted((mean([daily[k]['ec'] for k in kk if k in pub and k[0]=='F13']),mean([daily[k]['ec'] for k in kk if k in pub and k[0]=='F47'])) for kk in weather.values() if any(k in pub and k[0]=='F13' for k in kk) and any(k in pub and k[0]=='F47' for k in kk))
actual=sorted((float(r['F13']),float(r['F47'])) for r in cross)
eq(len(expected),len(actual))
for a,b in zip(expected,actual):near(a[0],b[0]);near(a[1],b[1])
near(corr([v[0] for v in expected],[v[1] for v in expected]),structure['cross_farm_pearson'])
near(rho([v[0] for v in expected],[v[1] for v in expected]),structure['cross_farm_spearman'])

for r in read(R/'persistence.csv'):
    g=sorted(k for k in roles if k in pub and k[0]==r['farm'] and (k[1]>=179)==(r['pass2']=='True'))
    xx=[];yy=[]
    for k in g:
        prev=[z for z in g if k[1]-10<=z[1]<k[1] and roles[z]['gid']!=roles[k]['gid'] and (roles[z]['role']==roles[k]['role'])==(r['same_role']=='True')]
        if prev:xx.append(daily[prev[-1]]['ec']);yy.append(daily[k]['ec'])
    eq(len(xx),int(r['n']));near(rho(xx,yy),r['rho'])

mid=read(R/'midnight.csv')
for r in mid:
    k=key(r);prev=(k[0],k[1]-1)
    eq(r['same_role']=='True',roles[k]['role']==roles[prev]['role'])
    a=f'{k[0]}_{k[1]:03d}_00';b=f'{prev[0]}_{prev[1]:03d}_23'
    near(abs(y[a][1]-y[b][1]),r['jump'])
eq(sum(r['same_role']=='True' for r in mid),0);eq(len(mid),140)

role=read(C/'role_oof.csv')
for r in role:eq(r['role'],roles[key(r)]['role'])
for filename,withfold in [('role_auc.csv',False),('role_fold_auc.csv',True)]:
    for r in read(C/filename):
        rr=[v for v in role if all(v[c]==r[c] for c in ['farm','scope','layout']) and (not withfold or v['fold']==r['fold'])]
        pos=[float(v['p']) for v in rr if v['role']=='B'];neg=[float(v['p']) for v in rr if v['role']=='A']
        auc=math.fsum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
        near(auc,r['auc'])
        groupfold=defaultdict(set)
        for v in rr:groupfold[v['gid']].add(v['fold'])
        assert all(len(v)==1 for v in groupfold.values())
ecd=read(F/'ec_days.csv')
for r in read(C/'role_bias.csv'):
    rr=[d for d in ecd if key(d) in roles and all(d[c]==r[c] for c in ['seed','farm','sealed']) and roles[key(d)]['role']==r['role']]
    eq(len(rr),int(r['n']));near(mean([float(v['bias']) for v in rr]),r['bias'])
    near(math.sqrt(math.fsum(float(v['sse']) for v in rr)/sum(int(v['n']) for v in rr)),r['rmse'])

out=dict(status='PASS_SCALAR_K1_K2_WEATHER_ROLES_STRUCTURE_PERSISTENCE_MIDNIGHT_ROLE_AUC_BIAS',checks=checks,max_absolute_error=maxerr,public_days=len(pub),public_pairs=len(pairs),cross_groups=len(cross),midnight_boundaries=len(mid),role_oof_rows=len(role),new_predictive_fits=0,source_sha=sha(__file__),limitations=['No predictive models retrained','Real greenhouse identity and calendar chronology remain unobserved','Role-model OOF arithmetic and source reviewed; estimator fitting not independently repeated'])
with (H/'critic_crosscheck_v2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False))
