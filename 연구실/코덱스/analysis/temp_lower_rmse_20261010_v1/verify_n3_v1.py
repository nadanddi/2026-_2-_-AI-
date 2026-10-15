from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
BASE_HOME=H.parent/'temp_repair_20261010_v1'
reg=json.loads((H/'n3_registration_v1.json').read_text(encoding='utf-8'));parent=json.loads((BASE_HOME/'registration_v1.json').read_text(encoding='utf-8'));res=json.loads((H/'n3_results_v1.json').read_text(encoding='utf-8'))
p=json.loads((R/'연구실/코덱스/analysis/temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
O=R/'연구실/코덱스/local/temp_repair_20261010_v1';I=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n3_target_v1'
labels=pd.read_csv(Path(p['temp_code']).parent/'data/train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
assert reg['data_sha']==parent['data_sha']==p['data_sha']
assert reg['source_sha']==hashlib.sha256((H/'n3_run_v3.py').read_bytes()).hexdigest()
checks=0;maxerr=0;groups={}
def cmp(a,b):
    global checks,maxerr
    diff=abs(a-b);maxerr=max(maxerr,diff);assert diff<5e-12,(a,b);checks+=1
for name,folds in parent['folds'].items():
    ids=[];aa=[];bb=[]
    for k,fd in enumerate(folds):
        file=I/(name+'_'+str(k)+'_v1.npz');m=json.loads((I/(name+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'))
        pf=O/(name+'_'+str(k)+'_v1.npz');pm=json.loads((O/(name+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'))
        assert hashlib.sha256(file.read_bytes()).hexdigest()==m['npz_sha'];assert hashlib.sha256(pf.read_bytes()).hexdigest()==m['parent_npz_sha']==pm['npz_sha']
        z=np.load(file,allow_pickle=True);zp=np.load(pf,allow_pickle=True)
        assert z['row_id'].tolist()==m['query_ids']==pm['query_ids'];assert np.array_equal(z['BASELINE'],zp['BASELINE'])
        assert m['weights']==pm['weights'] and m['season']==pm['season'];checks+=6
        if name=='DIAG10' and k==0:assert m['baseline_exact_probe_max']==0;checks+=1
        ids.extend(z['row_id'].tolist());aa.extend(map(float,z['BASELINE']));bb.extend(map(float,z['N3']))
    assert len(ids)==len(set(ids));y=labels.loc[ids].to_numpy();ea=[(a-t)**2 for a,t in zip(aa,y)];eb=[(b-t)**2 for b,t in zip(bb,y)]
    ra=math.sqrt(math.fsum(ea)/len(ids));rb=math.sqrt(math.fsum(eb)/len(ids));m=res['results'][name]
    cmp(ra,m['base_rmse']);cmp(rb,m['candidate_rmse']);cmp(100*(rb/ra-1),m['pct'])
    if name=='DIAG10':
        blk={}
        for i,row in enumerate(ids):blk.setdefault(row[:3]+'_'+str(int(row[4:7])//5),[]).append(i)
        keys=sorted(blk);sa=np.array([math.fsum(ea[i] for i in blk[key]) for key in keys]);sb=np.array([math.fsum(eb[i] for i in blk[key]) for key in keys]);sz=np.array([len(blk[key]) for key in keys]);rng=np.random.default_rng(20261010);ds=[]
        for j in range(20):
            draw=rng.integers(len(keys),size=(1000,len(keys)));den=sz[draw].sum(1);ds.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
        cmp(float(np.mean(np.array(ds)>=0)),m['p_worse'])
        for a,b in zip(np.quantile(ds,[reg['alpha'],1-reg['alpha']]),m['ci']):cmp(float(a),b)
    rows=[]
    for farm in ['F13','F47']:
        for late in [False,True]:
            ix=[i for i,row in enumerate(ids) if row[:3]==farm and (int(row[4:7])>=179)==late]
            if ix:rows.append({'farm':farm,'late':late,'n':len(ix),'base':math.sqrt(math.fsum(ea[i] for i in ix)/len(ix)),'candidate':math.sqrt(math.fsum(eb[i] for i in ix)/len(ix))})
    groups[name]=rows
passed=all(m['better'] for m in res['results'].values()) and res['results']['DIAG10']['p_worse']<reg['alpha'] and res['results']['DIAG10']['ci'][1]<0
assert passed==res['screen_pass'];checks+=1
receipt={'status':'PASS','checks':checks,'max_abs_numeric_gap':maxerr,'groups':groups,'new_fit':0,'ec_numeric_labels_read':False,'bootstrap_independent_recomputed':True}
with (H/'n3_independent_verify_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt),flush=True)
