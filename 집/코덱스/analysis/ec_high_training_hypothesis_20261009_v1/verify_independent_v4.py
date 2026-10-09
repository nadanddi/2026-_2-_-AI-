"""Independent verification; does not import the experiment producer or recipe."""
from pathlib import Path
import csv, json, hashlib, math, sys
from collections import defaultdict, Counter
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
L=ROOT/'집/코덱스/local'/H.name
ARMS=('BASE','ALL_HIGH','DISCORD_HIGH','CONTROL')
SEEDS=('7','101','2024','ensemble')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=2e-12):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,(a,b)
def metric(rows,col):
    e=[float(r[col])-float(r['sub_ec']) for r in rows]
    return math.sqrt(math.fsum(x*x for x in e)/len(e))
def normal(r,ys): return ys[(r['farm'],int(r['day']))]<1

def main(label):
    ks=[0] if label=='midpoint' else list(range(10))
    reg=json.loads((H/'registration_v4.json').read_text(encoding='utf8'))
    for p,v in dict(reg['pins'],**reg['cachepins']).items(): assert sha(p)==v,p
    replay=json.loads((H/'baseline_replay_v4.json').read_text(encoding='utf8'))
    assert replay['registration']==sha(H/'registration_v4.json') and replay['maxdiff']<1e-10
    rows=[];metas=[];sources={str(H/'registration_v4.json'):sha(H/'registration_v4.json'),str(H/'baseline_replay_v4.json'):sha(H/'baseline_replay_v4.json')}
    truths={};seen=set()
    for k in ks:
        p=L/'folds'/f'fold{k}.csv';mp=p.with_suffix('.json');m=json.loads(mp.read_text(encoding='utf8'))
        assert m['sha']==sha(p) and m['registration']==sha(H/'registration_v4.json') and m['baseline_final_maxdiff']<1e-10
        with p.open(encoding='utf8',newline='') as f: rr=list(csv.DictReader(f))
        assert len(rr)==m['rows']
        qs=reg['folds'][k]['query_ids'];assert len(rr)==len(qs)*16
        actual=Counter((r['arm'],r['seed']) for r in rr)
        assert set(actual)=={(a,s) for a in ARMS for s in SEEDS} and set(actual.values())=={len(qs)}
        for r in rr:
            assert int(r['k'])==k and r['row_id'] in set(qs)
            key=(r['row_id'],r['arm'],r['seed']);assert key not in seen;seen.add(key)
            yy=float(r['sub_ec']);assert math.isfinite(yy)
            if r['row_id'] in truths: assert truths[r['row_id']]==yy
            truths[r['row_id']]=yy
        assert all(len(v)==24 for v in [set(r['row_id'] for r in rr if r['farm']==f and int(r['day'])==d) for f,d in set((r['farm'],int(r['day'])) for r in rr)])
        for fi in m['fitinfo']:
            sel={tuple(x) for x in reg['folds'][k]['selections'][fi['arm']]}
            assert fi['removed_days']==len(sel) and fi['removed_rows']==24*len(sel)
            orig=reg['folds'][k]['train_ids'];expect=[x for x in orig if (x[:3],int(x.split('_')[1])) not in sel]
            assert fi['train_ids']==expect
        rows.extend(rr);metas.append(m);sources[str(p)]=sha(p);sources[str(mp)]=sha(mp)
    # Labels read once per unique row, independent of duplicated arm/seed rows.
    days=defaultdict(dict)
    for r in rows: days[(r['farm'],int(r['day']))][r['row_id']]=float(r['sub_ec'])
    ys={d:math.fsum(v.values())/len(v) for d,v in days.items()}
    assert len(truths)==len(days)*24
    if label=='final': assert len(truths)==8640 and len(days)==360
    # Registration selection consistency, not full feature-distance recomputation.
    selection_checks=[]
    for k in ks:
        rec=reg['folds'][k];dd={(x['farm'],x['day']):x for x in rec['selection_detail']}
        sels={a:{tuple(x) for x in rec['selections'][a]} for a in ARMS}
        assert sels['BASE']==set()
        assert sels['ALL_HIGH']=={d for d,r in dd.items() if r['y']>=1}
        assert sels['DISCORD_HIGH']=={d for d,r in dd.items() if r['y']>=1 and r['y']-r['neighbor_y']>.5}
        assert len(sels['CONTROL'])==len(sels['DISCORD_HIGH']) and sels['CONTROL'].isdisjoint(sels['ALL_HIGH'])
        for r in dd.values():
            ns=r['neighbors'];assert len(ns)==5 and len({(x['farm'],x['day']) for x in ns})==5
            assert all(x['farm']==r['farm'] and abs(x['day']-r['day'])>1 and (x['farm'],x['day']) in dd for x in ns)
            close(r['neighbor_y'],sorted(x['y'] for x in ns)[2]);close(r['gap'],r['y']-r['neighbor_y'])
            assert all(math.isfinite(x['distance']) and x['distance']>=0 for x in ns)
        for f in ('F13','F47'):
            for p2 in (False,True):
                ns=sum(ff==f and (d>=179)==p2 for ff,d in sels['DISCORD_HIGH'])
                nc=sum(ff==f and (d>=179)==p2 for ff,d in sels['CONTROL'])
                assert ns==nc
                assert all(dd[d]['y']<1 for d in sels['CONTROL'] if d[0]==f and (d[1]>=179)==p2)
        selection_checks.append({'k':k,'removed':{a:len(s) for a,s in sels.items()}})
    def eligible(scope,r):
        d=int(r['day']); n=normal(r,ys);f=r['farm']
        if scope=='all':return True
        if scope=='normal':return n
        if scope=='high':return not n
        if scope=='pass2':return d>=179
        if scope=='pass2_normal':return d>=179 and n
        if scope=='pass2_high':return d>=179 and not n
        if scope=='normal_without161':return n and not(f=='F47' and d==161)
        if scope in ('F13_normal','F47_normal'):return n and f==scope[:3]
        farm,ps,kind=scope.split(':')
        return f==farm and (d>=179)==(ps=='pass2') and (kind=='all' or n==(kind=='normal'))
    scopes=['all','normal','high','pass2','pass2_normal','pass2_high','normal_without161','F13_normal','F47_normal']
    scopes += [f'{f}:{ps}:{kind}' for f in ('F13','F47') for ps in ('pass1','pass2') for kind in ('all','normal','high')]
    metrics=[]
    for scope in scopes:
        groups=defaultdict(list)
        for r in rows:
            if eligible(scope,r):groups[(r['arm'],r['seed'])].append(r)
        for (a,s),rr in sorted(groups.items()):
            metrics.append(dict(scope=scope,arm=a,seed=s,rows=len(rr),days=len({(r['farm'],r['day']) for r in rr}),rmse=metric(rr,'prediction'),et_rmse=metric(rr,'et_prediction'),raw_et_rmse=metric(rr,'raw_et'),constant_rmse=metric(rr,'constant_prediction'),bias=math.fsum(float(r['prediction'])-float(r['sub_ec']) for r in rr)/len(rr)))
    produced=json.loads((H/f'{label}_score_v1.json').read_text(encoding='utf8'));sources[str(H/f'{label}_score_v1.json')]=sha(H/f'{label}_score_v1.json')
    ours={(r['scope'],r['arm'],r['seed']):r for r in metrics}
    for r in produced['groups']:
        q=ours[(r['scope'],r['arm'],r['seed'])];assert r['rows']==q['rows'] and r['days']==q['days']
        for col in ('rmse','et_rmse','bias','constant_rmse'):close(r[col],q[col])
    # Independent bootstrap loss lists and fsum totals, numpy used only for common RNG draws.
    losses=defaultdict(lambda:defaultdict(list));counts=Counter()
    for r in rows:
        if r['seed']=='ensemble' and normal(r,ys):
            key=(r['farm'],int(r['day'])//5);e=float(r['prediction'])-float(r['sub_ec']);losses[r['arm']][key].append(e*e)
            if r['arm']=='BASE':counts[key]+=1
    keys=sorted(counts);ct=np.asarray([counts[k] for k in keys],float)
    vals={a:np.asarray([math.fsum(losses[a][k]) for k in keys]) for a in ARMS}
    rng=np.random.default_rng(20261009);ixs=[]
    for farm in ('F13','F47'):
        ids=np.asarray([i for i,k in enumerate(keys) if k[0]==farm],int);assert len(ids)>0
        ixs.append(ids[rng.integers(0,len(ids),size=(20000,len(ids)))])
    ix=np.concatenate(ixs,axis=1);den=ct[ix].sum(axis=1);boot=[]
    for arm,ref in [('ALL_HIGH','BASE'),('DISCORD_HIGH','BASE'),('DISCORD_HIGH','CONTROL')]:
        delta=(vals[arm]-vals[ref])[ix].sum(axis=1);dr=np.sqrt(vals[arm][ix].sum(axis=1)/den)-np.sqrt(vals[ref][ix].sum(axis=1)/den)
        p=float(np.count_nonzero(delta>=0)/20000)
        b=dict(arm=arm,reference=ref,p_worse=p,rmse_delta=math.sqrt(math.fsum(vals[arm])/math.fsum(ct))-math.sqrt(math.fsum(vals[ref])/math.fsum(ct)),ci95_delta=[float(x) for x in np.quantile(dr,[.025,.975])],alpha=.025/3,blocks=len(keys))
        old=next(x for x in produced['bootstrap'] if x['arm']==arm and x['reference']==ref)
        close(old['p_worse'],b['p_worse'],1e-15);close(old['rmse_delta'],b['rmse_delta'])
        for x,y in zip(old['ci95_delta'],b['ci95_delta']):close(x,y)
        boot.append(b)
    # Reporting extras without changing producer source or registration.
    fold_normal=[]
    for k in ks:
        for a in ARMS:
            for s in SEEDS:
                rr=[r for r in rows if int(r['k'])==k and r['arm']==a and r['seed']==s and normal(r,ys)]
                if rr:fold_normal.append(dict(k=k,arm=a,seed=s,rows=len(rr),rmse=metric(rr,'prediction'),raw_et_rmse=metric(rr,'raw_et')))
    report=dict(status='INDEPENDENT_CHECK_PASS',label=label,unique_rows=len(truths),unique_days=len(days),high_days=sum(y>=1 for y in ys.values()),checked_producer_groups=len(produced['groups']),groups=metrics,bootstrap=boot,fold_normal=fold_normal,selection_checks=selection_checks,sources=sources,verifier_sha=sha(__file__),all_source_and_cache_pins_checked=True,limitations=['5NN features/standardization are source-reviewed; full matrices unavailable for independent replay','same single random normal CONTROL does not isolate discordance from removal of high-target mass','day//5 is calendar blocks with irregular observed normal-day coverage; descriptive CI95 unadjusted','DIAG-only ET-path diagnosis, no adoption or proof of absent evaluation high days'])
    p=H/f'{label}_independent_v4.json'
    with p.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('INDEPENDENT_CHECK_PASS',label,len(truths),len(days),'groups',len(produced['groups']),'boot',boot,flush=True)
if __name__=='__main__':main(sys.argv[1])



