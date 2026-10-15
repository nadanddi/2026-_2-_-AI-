"""Independent scalar-prefix, fsum metrics, parent identity and bootstrap checks."""
from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((H/'n5_registration_v2.json').read_text(encoding='utf-8'));rs=hashlib.sha256(json.dumps(reg,sort_keys=True).encode()).hexdigest()
assert reg['source_sha']==sha(H/'n5_run_v2.py')
prep=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'));D=Path(prep['temp_code']).parent/'data'
for n,v in reg['data_sha'].items():assert sha(D/n)==v
raw=pd.concat([pd.read_csv(D/n,usecols=['row_id','in_temp']) for n in ['train_X.csv','test_X.csv']])
rows=sorted((str(r),float(t)) for r,t in raw.itertuples(index=False,name=None) if str(r)[:3] in ['F13','F47'])
seen={};prefix=None;last=None
for rid,temperature in rows:
    key=rid[:7]
    if key!=last:prefix=None;last=key
    if not math.isnan(temperature):prefix=temperature if prefix is None else min(prefix,temperature)
    q=1. if prefix is None else max(0.,min(1.,(prefix-8.)/2.))
    g=1. if math.isnan(temperature) else max(0.,min(1.,(temperature-8.)/2.))
    seen[rid]=(q,g,np.nan if prefix is None else prefix)
y=pd.read_csv(D/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
P=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1';O=P.parent/'n5_v1'
results=json.loads((H/'n5_results_v2.json').read_text(encoding='utf-8'))
stats={};checks=0;gaps=[];detail={}
for s in ['DIAG10','EXT10','EXT12']:
    allids=[];ea=[];eb=[];qzero=0;reduced=0
    for fd in [f for f in reg['folds'] if f['set']==s]:
        name=s+'_'+str(fd['fold']);fn=O/(name+'.npz');m=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
        assert m['recipe_sha']==rs and m['npz_sha']==sha(fn) and m['parent_npz_sha']==reg['parents'][name]['npz_sha']==sha(P/fn.name)
        z=np.load(fn,allow_pickle=True);p=np.load(P/fn.name,allow_pickle=True);ids=z['row_id'].tolist()
        assert ids==fd['query_ids']==m['query_ids']==p['row_id'].tolist();checks+=4
        q=np.array([seen[r][0] for r in ids]);g=np.array([seen[r][1] for r in ids]);mn=np.array([seen[r][2] for r in ids])
        assert np.array_equal(q,z['q']) and np.array_equal(g,z['g']) and np.allclose(mn,z['prefix_min'],rtol=0,atol=0,equal_nan=True);checks+=3
        assert np.array_equal(z['BASELINE'],p['BASELINE']) and np.array_equal(z['N5'][q==g],p['N4'][q==g])
        assert np.array_equal(z['N5'][q==0],p['BASELINE'][q==0]) and np.array_equal(z['N5'][g==0],p['BASELINE'][g==0]);checks+=4
        scalar=np.array([math.fsum([.4*float(a),(.2+.4*(1-float(h)))*(float(c)+float(v)*(float(i)-float(c))),.4*float(h)*float(t)]) for a,c,i,t,h,v in zip(p['BASE'],p['CODEX'],p['IRR'],p['PFN'],g,q)])
        gap=float(np.max(np.abs(scalar-z['N5'])));assert gap<1e-13;gaps.append(gap);checks+=1
        actual=y.loc[ids].to_numpy();ea.extend(z['BASELINE']-actual);eb.extend(z['N5']-actual);allids.extend(ids)
        qzero+=int((q==0).sum());reduced+=int((q<g).sum())
    a=np.array(ea);b=np.array(eb);n=len(a);ra=math.sqrt(math.fsum(float(t)**2 for t in a)/n);rb=math.sqrt(math.fsum(float(t)**2 for t in b)/n)
    m=results['results'][s];assert abs(ra-m['base_rmse'])<1e-12 and abs(rb-m['candidate_rmse'])<1e-12;checks+=2
    stats[s]={'base_rmse':ra,'candidate_rmse':rb,'q0_rows':qzero,'q_reduced_rows':reduced,'rows':n}
    delta=b-a;cross=2*math.fsum(float(x)*float(d) for x,d in zip(a,delta))/n;d2=math.fsum(float(d)**2 for d in delta)/n
    assert abs((rb*rb-ra*ra)-(cross+d2))<1e-12;checks+=1
    stats[s].update(error_delta_cross_2E=cross,correction_mse=d2)
    if s=='DIAG10':
        _,inv=np.unique([rid[:3]+'_'+str(int(rid[4:7])//5) for rid in allids],return_inverse=True)
        ng=inv.max()+1;sa=np.bincount(inv,weights=a*a);sb=np.bincount(inv,weights=b*b);sizes=np.bincount(inv)
        rng=np.random.default_rng(20261010);deltas=[]
        for chunk in range(20):
            draw=rng.integers(ng,size=(1000,ng));den=sizes[draw].sum(1);deltas.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
        pw=float(np.mean(np.array(deltas)>=0));ci=np.quantile(deltas,[reg['alpha'],1-reg['alpha']])
        assert pw==m['p_worse'] and np.allclose(ci,m['ci'],rtol=0,atol=1e-12);checks+=2
        stats[s].update(p_worse=pw,ci=ci.tolist())
assert not results['pilot_pass'] and not results['adoption']
payload={'status':'PASS_NUMERICS_PILOT_FAIL','checks':checks,'max_formula_gap':max(gaps),'results':stats,'new_fit':0,'recipe_sha':rs,'full_prediction_causality_test':False}
with (H/'n5_verify_all_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print('N5_VERIFY',json.dumps(payload))
