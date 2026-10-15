"""Independent fsum/ID/purge/bootstrap verification of saved component outputs."""
from pathlib import Path
import sys,json,csv,math,hashlib
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np
import pandas as pd
reg=json.loads((H/'registration_v1.json').read_text(encoding='utf-8'))
res=json.loads((H/'results_v1.json').read_text(encoding='utf-8'))
p=json.loads((R/'연구실/코덱스/analysis/temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
O=R/'연구실/코덱스/local/temp_repair_20261009_v1'
for name,expected in p['data_sha'].items():
    assert reg['data_sha'][name]==expected
    assert hashlib.sha256((Path(p['temp_code']).parent/'data'/name).read_bytes()).hexdigest()==expected
labels=pd.read_csv(Path(p['temp_code']).parent/'data/train_y.csv',usecols=['row_id','sub_temp'])
truth=dict(zip(labels.row_id,labels.sub_temp))
checks=0;maxerr=0;allstats={}
def cmp(a,b):
    global checks,maxerr
    assert math.isfinite(a) and math.isfinite(b);gap=abs(a-b);maxerr=max(maxerr,gap);assert gap<5e-12,(a,b);checks+=1
for name,folds in reg['folds'].items():
    ids=[];pred={n:[] for n in ['BASELINE','DROP5','MED3']}
    for k,fd in enumerate(folds):
        file=O/(name+'_'+str(k)+'_v1.npz');m=json.loads((O/(name+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'))
        assert hashlib.sha256(file.read_bytes()).hexdigest()==m['npz_sha'];z=np.load(file,allow_pickle=True)
        rows=z['row_id'].tolist();assert rows==m['query_ids']
        assert not set(rows)&set(m['train_ids']);checks+=3
        qkeys={(r[:3],int(r[4:7])) for r in rows};trkeys={(r[:3],int(r[4:7])) for r in m['train_ids']}
        for farm,day in trkeys:assert all(abs(day-q)>1 for q in fd[farm]);checks+=1
        for farm,day in qkeys:assert day in fd[farm];checks+=1
        for farm,note in m['season'].items():
            if farm in ['F13','F47']:
                assert all((farm,int(day)) in trkeys for day in note['anchor_day']);checks+=len(note['anchor_day'])
        ids.extend(rows)
        for n in pred:pred[n].extend(map(float,z[n]))
    assert len(ids)==len(set(ids));checks+=1
    y=[truth[i] for i in ids];n=len(y)
    ea=[(a-t)**2 for a,t in zip(pred['BASELINE'],y)];ra=math.sqrt(math.fsum(ea)/n)
    for c in ['DROP5','MED3']:
        eb=[(b-t)**2 for b,t in zip(pred[c],y)];rb=math.sqrt(math.fsum(eb)/n);r=res['results'][name][c]
        cmp(ra,r['base_rmse']);cmp(rb,r['candidate_rmse']);cmp(100*(rb/ra-1),r['pct']);assert r['rows']==n
        if name=='DIAG10':
            blocks={}
            for i,row in enumerate(ids):blocks.setdefault((row[:3],int(row[4:7])//5),[]).append(i)
            keys=sorted(blocks,key=lambda x:x[0]+'_'+str(x[1]));sa=[];sb=[];sz=[]
            for key in keys:
                ix=blocks[key];sa.append(math.fsum(ea[i] for i in ix));sb.append(math.fsum(eb[i] for i in ix));sz.append(len(ix))
            sa=np.array(sa);sb=np.array(sb);sz=np.array(sz);rng=np.random.default_rng(20261009);ds=[]
            for j in range(20):
                draw=rng.integers(len(keys),size=(1000,len(keys)));den=sz[draw].sum(1)
                ds.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
            cmp(float(np.mean(np.array(ds)>=0)),r['p_worse'])
            for value,expected in zip(np.quantile(ds,[.0125,.9875]),r['ci']):cmp(float(value),expected)
        for g in r['groups']:
            take=[i for i,row in enumerate(ids) if row[:3]==g['farm'] and (int(row[4:7])>=179)==g['late']]
            assert len(take)==g['n'];cmp(math.sqrt(math.fsum(ea[i] for i in take)/len(take)),g['base']);cmp(math.sqrt(math.fsum(eb[i] for i in take)/len(take)),g['candidate'])
        allstats[(name,c)]=rb<ra
for c in ['DROP5','MED3']:
    test=all(allstats[(s,c)] for s in reg['folds']) and res['results']['DIAG10'][c]['p_worse']<.0125 and res['results']['DIAG10'][c]['ci'][1]<0
    assert test==res['screen_pass'][c];checks+=1
receipt={'status':'PASS','checks':checks,'max_abs_numeric_gap':maxerr,'sources':'saved fold NPZ + official sub_temp + metadata + registration','ec_numeric_labels_read':False,'new_fit':0}
with (H/'independent_verify_v3.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
