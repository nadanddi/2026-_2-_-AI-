"""사후 공개 DIAG10 일수준/시간내오차 분해. 최초실행 전에 CONFIG 고정.
모델학습/예측/test/원EC/EL1 접근 없음. 동할당 파일 사용 없음.
"""
from pathlib import Path
import sys,csv,json,math,hashlib,collections,copy
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
SRC=ROOT/'집/코덱스/analysis/ec_lgb_operation_20261004_v1'
OOF=ROOT/'집/코덱스/local/ec_lgb_operation_20261004_v1/oof.csv'
CONFIG=dict(kind='POSTHOC_DESCRIPTIVE_NO_ADOPTION',validator='DIAG10',seeds=[7,101,2024],models=['baseline','candidate'],seedmean='mean predictions per row before calculating error; not mean losses',error='prediction minus public y',daykey=['farm','day'],sse_identity='sum(e^2)=n*mean(e)^2+sum((e-mean(e))^2) per farm/day',high='mean public y per farm/day >=1',ordinary='mean public y per farm/day <1',closed='mean of raw act_vent==0 across 24hours; closed if fraction>=.8',pass2='day>=179',concentration='rank daily SSE descending within segment; ties farm/day ascending; top10 and top65 capped at segment days',groups=['all','high','ordinary','farm x pass within each segment','closed>=.8 vs rest within each segment','closed x farm x pass within each segment'],inference='no hypothesis tests; no causal or information-absence claim',no_dong=True,adoption_changes=0)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def near(a,b):assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<1e-10,(a,b)
def checkdict(a,b):
    assert set(a)==set(b)
    for k in a:
        if isinstance(a[k],dict):checkdict(a[k],b[k])
        elif isinstance(a[k],list):assert a[k]==b[k],k
        elif isinstance(a[k],(int,float)):near(a[k],b[k])
        else:assert a[k]==b[k]
def summary_scalar(days):
    assert days
    n=sum(d['n'] for d in days);sse=math.fsum(d['sse'] for d in days);sd=math.fsum(d['sse_day'] for d in days);si=math.fsum(d['sse_in'] for d in days)
    near(sse,sd+si);rank=sorted(days,key=lambda d:(-d['sse'],d['farm'],d['day']))
    return dict(n=n,days=len(days),bias=math.fsum(d['n']*d['bias'] for d in days)/n,rmse=math.sqrt(sse/n),sse=sse,sse_day=sd,sse_in=si,day_share_pct=100*sd/sse,within_share_pct=100*si/sse,day_error_rmse=math.sqrt(sd/n),within_error_rmse=math.sqrt(si/n),top10_sse_share_pct=100*math.fsum(d['sse'] for d in rank[:10])/sse,top10_days=min(10,len(days)),top65_sse_share_pct=100*math.fsum(d['sse'] for d in rank[:65])/sse,top65_days=min(65,len(days)),top10_ids=[f"{d['farm']}_{d['day']:03d}" for d in rank[:10]])
def summary_pandas(days):
    n=int(days.n.sum());sse=float(days.sse.sum());sd=float(days.sse_day.sum());si=float(days.sse_in.sum())
    rank=days.sort_values(['sse','farm','day'],ascending=[False,True,True],kind='stable')
    return dict(n=n,days=len(days),bias=float((days.n*days.bias).sum()/n),rmse=float(np.sqrt(sse/n)),sse=sse,sse_day=sd,sse_in=si,day_share_pct=100*sd/sse,within_share_pct=100*si/sse,day_error_rmse=float(np.sqrt(sd/n)),within_error_rmse=float(np.sqrt(si/n)),top10_sse_share_pct=100*float(rank.head(10).sse.sum())/sse,top10_days=min(10,len(days)),top65_sse_share_pct=100*float(rank.head(65).sse.sum())/sse,top65_days=min(65,len(days)),top10_ids=[f'{r.farm}_{r.day:03d}' for r in rank.head(10).itertuples()])
def group_keys(d):
    return ['all',f"farm_pass/{d['farm']}/{d['pass']}",f"closed/{d['closed']}",f"closed_farm_pass/{d['closed']}/{d['farm']}/{d['pass']}"]
def main():
    # Preserve all products and record fixed plan/source SHA before reading analysis rows.
    targets=[H/'predeclared_v1.json',H/'results_v1.json',H/'daily_v1.csv']
    assert all(not p.exists() for p in targets)
    with targets[0].open('x',encoding='utf-8') as f:json.dump(dict(config=CONFIG,source_sha256=sha(Path(__file__)),fit=0,predict=0),f,indent=2)
    whole=readj(SRC/'full_verification_v3.json');assert whole['status']=='PASS' and sha(OOF)==whole['aggregate_sha256']
    with OOF.open(encoding='utf-8',newline='') as f:rs=[r for r in csv.DictReader(f) if r['validator']=='DIAG10']
    assert len(rs)==25920
    raw_path=ROOT/'공용/대회자료/정형데이터/train_X.csv'
    if not raw_path.exists():
        sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
        raw_path=Path(env.DATA)/'train_X.csv'
    assert sha(raw_path)==whole['inputs']['train_X']
    # DictReader sees CSV fields, but only row_id/act_vent are used and retained.
    vents={}
    with raw_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);assert 'row_id' in reader.fieldnames and 'act_vent' in reader.fieldnames
        for r in reader:
            if r['row_id'][:3] in ['F13','F47']:
                x=float(r['act_vent']);assert math.isfinite(x) and x>=0 and r['row_id'] not in vents
                vents[r['row_id']]=x
    by_seed={s:[] for s in CONFIG['seeds']}
    for r in rs:by_seed[int(r['seed'])].append(r)
    for s,g in by_seed.items():assert len(g)==8640 and len({r['row_id'] for r in g})==8640
    maps={s:{r['row_id']:r for r in g} for s,g in by_seed.items()}
    ids=[r['row_id'] for r in by_seed[7]];assert all(set(m)==set(ids) for m in maps.values())
    records=[]
    for model in CONFIG['models']:
        for seed in CONFIG['seeds']+['mean']:
            for rid in ids:
                r=maps[7][rid];y=float(r['y'])
                assert all(float(maps[s][rid]['y'])==y for s in CONFIG['seeds']) and rid in vents
                p=math.fsum(float(maps[s][rid][model]) for s in CONFIG['seeds'])/3 if seed=='mean' else float(maps[seed][rid][model])
                records.append(dict(model=model,seed=str(seed),row_id=rid,farm=r['farm'],day=int(r['day']),hour=int(r['hour']),y=y,p=p,vent=vents[rid],error=p-y))
    grouped=collections.defaultdict(list)
    for r in records:grouped[r['model'],r['seed'],r['farm'],r['day']].append(r)
    days=[]
    for (model,seed,farm,day),g in grouped.items():
        assert len(g)==24 and {r['hour'] for r in g}==set(range(24))
        bias=math.fsum(r['error'] for r in g)/24;ymean=math.fsum(r['y'] for r in g)/24;sealed=sum(r['vent']==0 for r in g)/24
        sse=math.fsum(r['error']**2 for r in g);sd=24*bias*bias;si=math.fsum((r['error']-bias)**2 for r in g);near(sse,sd+si)
        days.append(dict(model=model,seed=seed,farm=farm,day=day,n=24,ymean=ymean,bias=bias,sse=sse,sse_day=sd,sse_in=si,sealed_fraction=sealed,closed='closed' if sealed>=.8 else 'rest',pass_='p2' if day>=179 else 'p1',segment='high' if ymean>=1 else 'ordinary'))
    # Fully separate row->day pandas path using transform rather than scalar day summaries.
    df=pd.DataFrame(records);keys=['model','seed','farm','day'];df['e']=df.p-df.y;df['ed']=df.groupby(keys).e.transform('mean');df['ei']=df.e-df.ed
    df['sse']=df.e**2;df['sse_day']=df.ed**2;df['sse_in']=df.ei**2;df['sealed']=(df.vent==0).astype(int)
    dp=df.groupby(keys,sort=False).agg(n=('e','size'),ymean=('y','mean'),bias=('e','mean'),sse=('sse','sum'),sse_day=('sse_day','sum'),sse_in=('sse_in','sum'),sealed_fraction=('sealed','mean')).reset_index()
    dp['closed']=np.where(dp.sealed_fraction>=.8,'closed','rest');dp['pass']=np.where(dp.day>=179,'p2','p1');dp['segment']=np.where(dp.ymean>=1,'high','ordinary')
    pm={(r.model,r.seed,r.farm,r.day):r for r in dp.itertuples()}
    for d in days:
        d['pass']=d.pop('pass_');r=pm[d['model'],d['seed'],d['farm'],d['day']]
        for c in ['n','ymean','bias','sse','sse_day','sse_in','sealed_fraction']:near(d[c],getattr(r,c))
        assert d['closed']==r.closed and d['pass']==getattr(r,'_14',d['pass']) if False else d['segment']==r.segment
    summaries={};comparison_count=0
    for model in CONFIG['models']:
        for seed in [str(s) for s in CONFIG['seeds']]+['mean']:
            ds=[d for d in days if d['model']==model and d['seed']==seed]
            assert len(ds)==360 and sum(d['segment']=='high' for d in ds)==31 and sum(d['segment']=='ordinary' for d in ds)==329
            for segment in ['all','high','ordinary']:
                selected=ds if segment=='all' else [d for d in ds if d['segment']==segment]
                buckets=collections.defaultdict(list)
                for d in selected:
                    for key in group_keys(d):buckets[key].append(d)
                for key,g in buckets.items():
                    a=summary_scalar(g);subset=dp[(dp.model==model)&(dp.seed==seed)]
                    if segment!='all':subset=subset[subset.segment==segment]
                    parts=key.split('/')
                    if parts[0]=='farm_pass':subset=subset[(subset.farm==parts[1])&(subset['pass']==parts[2])]
                    elif parts[0]=='closed':subset=subset[subset.closed==parts[1]]
                    elif parts[0]=='closed_farm_pass':subset=subset[(subset.closed==parts[1])&(subset.farm==parts[2])&(subset['pass']==parts[3])]
                    b=summary_pandas(subset);checkdict(a,b);comparison_count+=1
                    summaries[f'{model}/{seed}/{segment}/{key}']=a
    # ND0 source is R3S, not actualseasonv2: do not import/read dong assignment or separate Claude OOF.
    shared_oof=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
    with shared_oof.open(encoding='utf-8',newline='') as f:header=next(csv.reader(f))
    r3s_columns=[c for c in header if c.lower().startswith('r3s')]
    assert not r3s_columns,'New source model needs separate explicit treatment'
    result=dict(status='PASS_DESCRIPTIVE_TWO_PATHS',config=CONFIG,source_sha256=sha(Path(__file__)),source_inputs=dict(aggregate=sha(OOF),train_X=sha(raw_path),whole=sha(SRC/'full_verification_v3.json')),rows_per_model_seed=8640,models=CONFIG['models'],seeds=CONFIG['seeds']+['mean'],days_per_model_seed=360,high_days=31,ordinary_days=329,summary_comparisons=comparison_count,summaries=summaries,r3s='Existing public S.outer CSV has no r3s column; Claude ND0 R3S numbers remain log-only provenance',fit=0,predict=0,test_reads=0,raw_ec_reads=0,EL1_rescore=0,dong_reads=0,adoption_changes=0,limitations=['posthoc public repeated validation','y-based high/ordinary and top-error ranking are diagnostic only','daily vent==0 grouping is association; severity/season/farm confounded','raw daily control summary uses complete day for analysis; not an inference feature','no untouched holdout/no information-absence proof'])
    with targets[1].open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    pd.DataFrame(days).to_csv(targets[2],index=False,mode='x')
    print('PASS_DESCRIPTIVE_TWO_PATHS',comparison_count)
    for model in CONFIG['models']:
        for seed in [str(s) for s in CONFIG['seeds']]+['mean']:
            a=summaries[f'{model}/{seed}/ordinary/all'];closed=summaries[f'{model}/{seed}/ordinary/closed/closed'];rest=summaries[f'{model}/{seed}/ordinary/closed/rest']
            print(model,seed,'normal day%',a['day_share_pct'],'RMSE',a['rmse'],'top10%',a['top10_sse_share_pct'],'top65%',a['top65_sse_share_pct'],'closed',closed['days'],closed['bias'],'rest',rest['days'],rest['bias'])
if __name__=='__main__':main()
