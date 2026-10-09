"""생산자 run/recipe import 없이 데이터·선정·산술을 검산한다."""
from pathlib import Path
import sys,csv,json,math,hashlib,statistics
from collections import defaultdict,Counter
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
L=ROOT/'집/코덱스/local'/H.name
S=('7','101','2024');A=('BASE','CLEAN')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf8'))
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,t=3e-12):assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=t,(a,b)
def day(rid):p=rid.split('_');return p[0],int(p[1])
def rmse(rr,col='prediction'):return math.sqrt(math.fsum((float(r[col])-float(r['sub_ec']))**2 for r in rr)/len(rr))
def save(name,v):
    with (H/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def data_checks(reg):
    for p,h in dict(reg['pins'],**reg['cachepins']).items():assert sha(p)==h,p
    raw={r['row_id']:r for r in csvrows(Path(env.DATA)/'train_X.csv')}
    truth={}
    for r in csvrows(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'):
        if r['validator']=='DIAG10':
            if r['row_id'] in truth:close(truth[r['row_id']],float(r['y']),1e-14)
            truth[r['row_id']]=float(r['y'])
    assert len(truth)==8640
    def one(dest,original_ids,expected_t=None,expected_q=None):
        sn=dest/'selector_snapshot_v1.npz';de=dest/'selection_detail_v1.json';m=js(dest/'cleaning_manifest_v1.json')
        for p,h in m['hashes'].items():assert sha(dest/p)==h
        with np.load(sn,allow_pickle=False) as s:
            tk=[(x.split('_')[0],int(x.split('_')[1])) for x in s['train_keys'].astype(str)]
            qk=[(x.split('_')[0],int(x.split('_')[1])) for x in s['query_keys'].astype(str)]
            z=s['train_z'].copy();qz=s['query_z'].copy();ty=s['train_y'].copy();qy=s['query_y'].copy()
            assert z.shape==(len(tk),47) and qz.shape==(len(qk),47) and np.isfinite(z).all() and np.isfinite(qz).all()
            assert all(np.isfinite(s[x]).all() for x in ('median','mean','std')) and (s['std']>0).all()
            assert np.max(abs(z.mean(0)))<1e-10
            std=z.std(0);assert all(abs(v)<1e-10 or abs(v-1)<1e-10 for v in std)
        td=defaultdict(list)
        for rid in original_ids:td[day(rid)].append(truth[rid])
        assert set(td)==set(tk) and all(len(v)==24 for v in td.values())
        for i,d in enumerate(tk):close(ty[i],math.fsum(td[d])/24)
        detail=js(de);look={(r['scope'],r['farm'],r['day']):r for r in detail};assert len(look)==len(tk)+len(qk)
        removed={};checks=0
        for scope,keys,points,target in [('train',tk,z,ty),('query_conditional_only',qk,qz,qy)]:
            bad=set()
            for i,(f,d) in enumerate(keys):
                eligible=[j for j,(ff,dd) in enumerate(tk) if ff==f and abs(dd-d)>1];assert len(eligible)>=5
                distances=np.sqrt(np.mean((z-points[i])**2,axis=1));nn=sorted(eligible,key=lambda j:(distances[j],tk[j]))[:5]
                neighbor=sorted(float(ty[j]) for j in nn)[2];isbad=target[i]>=1 and target[i]-neighbor>.5
                if isbad:bad.add((f,d))
                r=look[scope,f,d];assert r['remove']==bool(isbad);close(r['y'],target[i]);close(r['neighbor_y'],neighbor);close(r['gap'],target[i]-neighbor)
                assert [(v['farm'],v['day']) for v in r['neighbors']]==[tk[j] for j in nn]
                for v,j in zip(r['neighbors'],nn):close(v['y'],ty[j]);close(v['distance'],distances[j],2e-10)
                if scope!='train':
                    vals=[truth[rid] for rid in truth if day(rid)==(f,d)];assert len(vals)==24;close(target[i],math.fsum(vals)/24)
                checks+=1
            removed[scope]=bad
        if expected_t is not None:assert removed['train']=={tuple(v) for v in expected_t}
        if expected_q is not None:assert removed['query_conditional_only']=={tuple(v) for v in expected_q}
        cx=csvrows(dest/'train_X_clean_v1.csv');cy=csvrows(dest/'train_y_clean_v1.csv');ids=[r['row_id'] for r in cx]
        assert ids==[r['row_id'] for r in cy] and len(ids)==len(set(ids))
        assert set(ids)=={rid for rid in original_ids if day(rid) not in removed['train']}
        assert {r['row_id'] for r in csvrows(dest/'removed_row_ids_v1.csv')}==set(original_ids)-set(ids)
        assert {tuple((r['farm'],int(r['day']))) for r in csvrows(dest/'removed_days_v1.csv')}==removed['train']
        assert len(original_ids)-len(ids)==24*len(removed['train'])
        for xr,yr in zip(cx,cy):
            close(float(yr['sub_ec']),truth[yr['row_id']],1e-14)
            for c,v in xr.items():
                if c=='row_id':continue
                u=raw[xr['row_id']][c]
                if not u or not v:assert not u and not v
                else:close(float(v),float(u),1e-10)
        assert all(v==24 for v in Counter(map(day,ids)).values())
        assert m['before']['rows']==len(original_ids) and m['after']['rows']==len(ids) and m['removed_days']==len(removed['train'])
        return dict(path=str(dest),selector_day_checks=checks,removed_train_days=len(removed['train']),removed_query_days=len(removed['query_conditional_only']),retained_rows=len(ids))
    checks=[one(L/'dataset_public',list(truth),reg['public_removed_days'],[])]
    allq=[]
    for rec in reg['folds']:
        checks.append(one(L/'cv_data'/f"fold{rec['k']}",rec['original_train_ids'],rec['removed_train_days'],rec['removed_query_days']))
        retained={r['row_id'] for r in csvrows(L/'cv_data'/f"fold{rec['k']}"/'query_retained_ids_v1.csv')};removed={r['row_id'] for r in csvrows(L/'cv_data'/f"fold{rec['k']}"/'query_removed_ids_v1.csv')}
        assert retained.isdisjoint(removed) and retained|removed==set(rec['query_ids'])
        assert removed=={rid for rid in rec['query_ids'] if day(rid) in {tuple(v) for v in rec['removed_query_days']}}
        assert set(rec['original_train_ids']).isdisjoint(rec['query_ids']);allq.extend(rec['query_ids'])
    assert len(allq)==len(set(allq))==8640
    return checks,truth

def main(mode):
    reg=js(H/'registration_v1.json');checks,truth=data_checks(reg)
    if mode=='data':save('data_independent_v1.json',dict(status='DATA_SELECTOR_PASS',checks=checks,source_pins_verified=True,global_clean_used_by_cv=False,consumed40_labels_read=False,registration_sha=sha(H/'registration_v1.json')));print('DATA_SELECTOR_PASS',checks,flush=True);return
    ks=[0] if mode=='midpoint' else list(range(10));rows=[];receipts=[]
    replay=js(H/'baseline_replay_v1.json');assert replay['models']==3 and replay['registration']==sha(H/'registration_v1.json')
    assert replay['gaps']['et']<1e-10 and replay['gaps']['lgb']<1e-10 and replay['gaps']['mlp']<1e-8
    for k in ks:
        p=L/'cv_predictions'/f'fold{k}.csv';m=js(p.with_suffix('.json'));assert m['sha']==sha(p) and m['registration']==sha(H/'registration_v1.json')
        rr=csvrows(p);rec=reg['folds'][k];assert len(rr)==len(rec['query_ids'])*8
        assert len({(r['row_id'],r['arm'],r['seed']) for r in rr})==len(rr)
        assert m['candidate_models']==len(m['fitinfo'])==9 and m['train_ids']==rec['clean_train_ids']
        assert {(r['kind'],r['seed']) for r in m['fitinfo']}=={(a,int(s)) for a in ('et','lgb','mlp') for s in S}
        assert all(r['rows']==len(m['train_ids']) for r in m['fitinfo'])
        assert all(r['train_ids_sha']==hashlib.sha256('\n'.join(m['train_ids']).encode()).hexdigest() for r in m['fitinfo'])
        cache={}
        base_root=ROOT/'연구실/코덱스/local/ec_current14_influence_20261007_v1/base'
        for seed in S:
            with np.load(base_root/f'{k}_{seed}.npz',allow_pickle=False) as z:
                assert z['row_id'].tolist()==rec['query_ids'];cache[seed]=.6*z['et']+.3*z['lgb']+.1*z['mlp']
        cache['ensemble']=np.mean([cache[s] for s in S],axis=0)
        qr={tuple(v) for v in rec['removed_query_days']}
        by=defaultdict(list)
        for r in rr:
            assert r['row_id'] in set(rec['query_ids']);close(r['sub_ec'],truth[r['row_id']],1e-14)
            assert (r['query_removed']=='True')==(day(r['row_id']) in qr)
            by[r['arm'],r['seed']].append(r)
        assert set(by)=={(a,s) for a in A for s in S+('ensemble',)}
        for (a,s),group in by.items():
            assert [r['row_id'] for r in group]==rec['query_ids']
            if a=='BASE':
                for i,r in enumerate(group):close(r['raw_prediction'],cache[s][i],2e-12)
            elif s=='ensemble':
                for i,r in enumerate(group):close(r['raw_prediction'],math.fsum(float(by[a,se][i]['raw_prediction']) for se in S)/3)
            bounds=rec['clean_bounds'] if a=='CLEAN' else [min(truth[rid] for rid in rec['original_train_ids']),max(truth[rid] for rid in rec['original_train_ids'])]
            tm=math.fsum(truth[rid] for rid in (rec['clean_train_ids'] if a=='CLEAN' else rec['original_train_ids']))/len(rec['clean_train_ids'] if a=='CLEAN' else rec['original_train_ids'])
            dd=defaultdict(list)
            for r in group:dd[day(r['row_id'])].append(r)
            for g in dd.values():
                g.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24))
                pp=[]
                for r in g:
                    pp.append(float(r['raw_prediction']));pred=min(max(.5*pp[-1]+.5*math.fsum(pp)/len(pp),bounds[0]),bounds[1]);close(r['prediction'],pred);close(r['constant_prediction'],tm)
        receipts.append(dict(k=k,new_models=9,train_rmse=m['clean_train_rmse']));rows.extend(rr)
    daily=defaultdict(dict)
    for r in rows:daily[day(r['row_id'])][r['row_id']]=float(r['sub_ec'])
    high={d:math.fsum(v.values())/24>=1 for d,v in daily.items()};assert all(len(v)==24 for v in daily.values())
    if mode=='final':assert len(daily)==360
    def eligible(sc,r):
        f,d=day(r['row_id']);removed=r['query_removed']=='True'
        if sc=='all':return True
        if sc=='filtered':return not removed
        if sc=='removed':return removed
        if sc=='normal':return not high[f,d]
        if sc=='high':return high[f,d]
        if sc=='pass2_all':return d>=179
        if sc=='pass2_filtered':return d>=179 and not removed
        ff,ps,kind=sc.split('_');return f==ff and (d>=179)==(ps=='pass2') and (kind=='all' or not removed)
    prod=js(H/f'{mode}_score_v1.json');groups=[]
    for sc in {r['scope'] for r in prod['groups']}:
        g=defaultdict(list)
        for r in rows:
            if eligible(sc,r):g[r['arm'],r['seed']].append(r)
        for (a,s),rr in sorted(g.items()):groups.append(dict(scope=sc,arm=a,seed=s,rows=len(rr),days=len({day(r['row_id']) for r in rr}),rmse=rmse(rr),raw_rmse=rmse(rr,'raw_prediction'),bias=math.fsum(float(r['prediction'])-float(r['sub_ec']) for r in rr)/len(rr),constant_rmse=rmse(rr,'constant_prediction')))
    gd={(r['scope'],r['arm'],r['seed']):r for r in groups}
    for r in prod['groups']:
        q=gd[r['scope'],r['arm'],r['seed']];assert q['rows']==r['rows'] and q['days']==r['days']
        for c in ('rmse','bias','constant_rmse'):close(q[c],r[c])
    boot=[]
    for sc in ('filtered','all'):
        losses=defaultdict(lambda:defaultdict(list));counts=Counter()
        for r in rows:
            if r['seed']=='ensemble' and eligible(sc,r):
                key=(r['farm'],int(r['day'])//5);e=float(r['prediction'])-float(r['sub_ec']);losses[r['arm']][key].append(e*e)
                if r['arm']=='BASE':counts[key]+=1
        keys=sorted(counts);n=np.array([counts[k] for k in keys]);v={a:np.array([math.fsum(losses[a][k]) for k in keys]) for a in A}
        rng=np.random.default_rng(20261009);draw=[]
        for f in ('F13','F47'):
            ids=np.array([i for i,k in enumerate(keys) if k[0]==f]);assert len(ids);draw.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
        ix=np.concatenate(draw,1);delta=(v['CLEAN']-v['BASE'])[ix].sum(1);den=n[ix].sum(1);dr=np.sqrt(v['CLEAN'][ix].sum(1)/den)-np.sqrt(v['BASE'][ix].sum(1)/den)
        b=dict(scope=sc,p_worse=float(np.count_nonzero(delta>=0)/20000),rmse_delta=math.sqrt(math.fsum(v['CLEAN'])/math.fsum(n))-math.sqrt(math.fsum(v['BASE'])/math.fsum(n)),ci95_delta=[float(x) for x in np.quantile(dr,[.025,.975])],alpha=.0125,blocks=len(keys))
        old=next(r for r in prod['bootstrap'] if r['scope']==sc);close(b['p_worse'],old['p_worse'],1e-15);close(b['rmse_delta'],old['rmse_delta'])
        for x,y in zip(b['ci95_delta'],old['ci95_delta']):close(x,y)
        boot.append(b)
    gaps=[]
    for m in receipts:
        for seed,tr in m['train_rmse'].items():
            rr=[r for r in rows if int(r['k'])==m['k'] and r['arm']=='CLEAN' and r['seed']==seed]
            for sc in ('all','filtered'):
                q=[r for r in rr if eligible(sc,r)]
                if q:gaps.append(dict(k=m['k'],seed=seed,scope=sc,train_rmse=tr,validation_rmse=rmse(q),validation_minus_train=rmse(q)-tr))
    screen={sc:all(gd[sc,'CLEAN',s]['rmse']<gd[sc,'BASE',s]['rmse'] for s in S) and next(b['p_worse'] for b in boot if b['scope']==sc)<.0125 for sc in ('filtered','all')}
    guard=any(gd[sc,'CLEAN',s]['rmse']/gd[sc,'BASE',s]['rmse']>=1.02 for sc in ('all','pass2_all') if (sc,'BASE','ensemble') in gd for s in S+('ensemble',))
    out=dict(status='INDEPENDENT_NUMERIC_PASS',mode=mode,unique_rows=len(daily)*24,unique_days=len(daily),checks=checks,groups=groups,bootstrap=boot,new_cv_models=sum(m['new_models'] for m in receipts),baseline_replay_models=3,train_validation_gaps=gaps,screen=screen,utility_guard_trigger=guard,adoption=False,registration_sha=sha(H/'registration_v1.json'),verifier_sha=sha(__file__),limitations=['조건부 검증은 q 실제 일평균으로 선택되어 실전에서 선택할 수 없다','CV CLEAN 구성원 raw와모델 미저장: mixraw 이후 산술 전수검산, 개별혼합 source검토 한정','CV trainRMSE 원예측 미저장: receipt값에서 검증격차 산술만계산','day//5는 상대기록번호 구간이며 실제연속달력일 아님'])
    save(f'{mode}_independent_v1.json',out);print('INDEPENDENT_NUMERIC_PASS',mode,len(daily),'days',boot,'screen',screen,'guard',guard,flush=True)
if __name__=='__main__':main(sys.argv[1])
