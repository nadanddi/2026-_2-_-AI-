"""Saved-array arithmetic only. No model/helper imports, fitting or tree routing."""
from pathlib import Path
import sys, csv, json, math, hashlib, collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
import numpy as np
L=ROOT/'연구실/코덱스/local'/H.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
maxerr=collections.defaultdict(float)
def near(tag,a,b,tol=1e-10):
    e=abs(float(a)-float(b));maxerr[tag]=max(maxerr[tag],e);assert math.isfinite(e) and e<=tol,(tag,a,b)
def phi(v,n):
    out=[]
    for i in range(n):
        terms=[]
        for mask in range(2**n):
            if (mask>>i)&1:continue
            terms.append((float(v[mask+2**i])-float(v[mask]))/(n*math.comb(n-1,mask.bit_count())))
        out.append(math.fsum(terms))
    near('phi_efficiency',math.fsum(out),float(v[-1])-float(v[0]));return out
def sumdot(w,y):return math.fsum(float(a)*float(b) for a,b in zip(w,y))
completion=json.loads((H/'completion_v1.json').read_text(encoding='utf-8'))
assert completion['status']=='COMPLETE_FIXED_ET_DIAGNOSTIC' and completion['new_ET_fits']==5 and completion['other_fits']==0
prep=json.loads((H/'preparation_v1.json').read_text(encoding='utf-8'))
assert sha(H/'preparation_v1.json')==completion['prep_sha']
for path,h in prep['sources'].items():assert sha(Path(path))==h
for name,h in completion['files'].items():assert sha(H/name)==h
endpoints=read(H/'endpoints_v1.csv');shapleys=read(H/'shapley_v1.csv');interactions=read(H/'interactions_v1.csv');profiles=read(H/'profiles_v1.csv')
cols=[c for c in profiles[0] if c not in {'model','pair','endpoint','row_id','hour'}]
assert len(cols)==47 and len(set(cols))==47
flat=sum(prep['groups'].values(),[]);assert len(flat)==47 and set(flat)==set(cols)
model_labels=['original_1_7','original_4_7','original_8_7','common_7','common_101','common_2024']
expected={(m,p['tag']) for m in model_labels for p in prep['pairs'] if not m.startswith('original') or int(m.split('_')[1])==p['fold']}
assert len(expected)==20
checks=[];files={};mask_total=0
for label in model_labels:
    path=L/f'{label}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert sha(path)==meta['sha'] and meta['prep_sha']==completion['prep_sha'];files[path.name]=meta['sha']
    if label.startswith('original'):assert meta['baseline_cache_error']<=1e-10
    with np.load(path,allow_pickle=False) as z:
        ids=z['train_row_id'].copy();y=z['train_y'].copy();median=z['imputer_median'].copy()
    byday=collections.defaultdict(list)
    for rid,val in zip(ids,y):byday[(str(rid)[:3],int(str(rid)[4:7]))].append(float(val))
    assert all(len(v)==24 for v in byday.values())
    high_days={k for k,v in byday.items() if math.fsum(v)/24>=1}
    high=np.array([(str(rid)[:3],int(str(rid)[4:7])) in high_days for rid in ids])
    for p in prep['pairs']:
        tag=p['tag']
        if (label,tag) not in expected:continue
        path=L/f'coalition_{label}_{tag}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert sha(path)==meta['sha'] and meta['forest_sha']==files[f'{label}.npz'];files[path.name]=meta['sha']
        with np.load(path,allow_pickle=False) as z:
            assert z['train_row_id'].tolist()==ids.tolist()
            g=z['groups'].tolist();n=len(g);count=2**n;assert meta['groups']==g and meta['coalitions']==count
            assert z['X'].shape==(24*count,47) and z['raw_curve'].shape==(count,24)
            X=z['X'].reshape(count,24,47)
            for end,index in [('control',0),('failure',-1)]:
                rows=sorted([r for r in profiles if r['model']==label and r['pair']==tag and r['endpoint']==end],key=lambda r:int(r['hour']))
                assert [int(r['hour']) for r in rows]==list(range(24))
                raw=np.array([[float(r[c]) if r[c]!='' else float('nan') for c in cols] for r in rows]);im=np.where(np.isnan(raw),median,raw).astype(np.float32)
                assert np.array_equal(im,X[index],equal_nan=True)
            active=[k for k,cc in prep['groups'].items() if not np.array_equal(X[0][:,[cols.index(c) for c in cc]],X[-1][:,[cols.index(c) for c in cc]],equal_nan=True)]
            assert active==g and 'clock' not in g
            for mask in range(count):
                for i,name in enumerate(g):
                    ix=[cols.index(c) for c in prep['groups'][name]]
                    assert np.array_equal(X[mask][:,ix],X[-1 if (mask>>i)&1 else 0][:,ix],equal_nan=True)
                curve=z['raw_curve'][mask];rawvalue=math.fsum(map(float,curve))/24
                smoothvalue=math.fsum((float(curve[h])+math.fsum(map(float,curve[:h+1]))/(h+1))/2 for h in range(24))/24
                for name,val in [('raw',rawvalue),('smooth',smoothvalue)]:
                    w=z['weights_raw' if name=='raw' else 'weights_smooth'][mask]
                    assert len(w)==len(y) and np.isfinite(w).all() and (w>=0).all()
                    near('weight_sum',math.fsum(map(float,w)),1)
                    near(name+'_curve',z[name][mask],val);near(name+'_support',z[name][mask],sumdot(w,y))
                near('high_day_support',z['high_support'][mask],math.fsum(float(w) for w,h in zip(z['weights_smooth'][mask],high) if h))
            values={k:z[k].copy() for k in ['raw','smooth','high_support']}
            phis={k:phi(v,n) for k,v in values.items()}
            sr=[r for r in shapleys if r['model']==label and r['pair']==tag];assert len(sr)==n and {r['group'] for r in sr}==set(g)
            for r in sr:
                i=g.index(r['group'])
                for metric,key in [('raw','raw_phi'),('smooth','smooth_phi'),('high_support','high_phi')]:near('shapley_'+metric,r[key],phis[metric][i])
            ir=[r for r in interactions if r['model']==label and r['pair']==tag];assert len(ir)==math.comb(n,2)*6
            seen=set()
            for r in ir:
                i,j=g.index(r['g1']),g.index(r['g2']);assert i<j
                key=(i,j,r['context'],r['metric']);assert key not in seen;seen.add(key)
                v=values[r['metric']];start=0 if r['context']=='control' else count-1
                both=start^(2**i)^(2**j);one=start^(2**i);two=start^(2**j)
                delta=(float(v[both])-float(v[start]))-(float(v[one])-float(v[start]))-(float(v[two])-float(v[start]))
                near('nonadditivity',r['nonadditivity'],delta)
            for end,index in [('control',0),('failure',-1)]:
                rr=[r for r in endpoints if r['model']==label and r['pair']==tag and r['endpoint']==end];assert len(rr)==1;r=rr[0]
                for k in values:near('endpoint_'+k,r[k],values[k][index])
                w=z['weights_smooth'][index];near('samefarm_support',r['samefarm_support'],math.fsum(float(a) for a,rid in zip(w,ids) if str(rid)[:3]==p['farm']))
                support=read(H/f'support_{label}_{tag}_{end}.csv');expected_support=collections.defaultdict(list)
                for rid,a,b in zip(ids,w,y):expected_support[(str(rid)[:3],int(str(rid)[4:7]))].append((float(a),float(a)*float(b)))
                expected_support={k:(math.fsum(a for a,b in v),math.fsum(b for a,b in v)) for k,v in expected_support.items() if math.fsum(a for a,b in v)>0}
                assert len(support)==len(expected_support) and {(r['farm'],int(r['day'])) for r in support}==set(expected_support)
                for r in support:
                    key=(r['farm'],int(r['day']));weight,contribution=expected_support[key]
                    near('support_day_weight',r['weight'],weight);near('support_day_y',r['weighted_y'],contribution);near('support_day_selected_y',r['selected_y'],contribution/weight);near('support_day_truth',r['day_truth'],math.fsum(byday[key])/24)
            mask_total+=count;checks.append({'model':label,'pair':tag,'groups':n,'masks':count,'smooth_difference':float(values['smooth'][-1]-values['smooth'][0]),'high_day_support_difference':float(values['high_support'][-1]-values['high_support'][0]),'smooth_shapley':dict(zip(g,phis['smooth'])),'high_shapley':dict(zip(g,phis['high_support']))})
        print('CRITIC_PAIR_PASS',label,tag,flush=True)
assert len(checks)==20 and len(endpoints)==40
artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in [H/'preparation_v1.json',H/'preflight_v1.json',H/'completion_v1.json',H/'run_v1.py',H/'core_v1.py',*H.glob('*_v1.csv'),*H.glob('support_*.csv'),*H.glob('paths_*.json'),*L.glob('*.npz'),*L.glob('*.json')]}
out={'status':'PASS_FIXED_ET_SAVED_ARRAY_ARITHMETIC','new_fit':0,'tree_routing':0,'pairs':len(checks),'masks':mask_total,'max_errors':dict(maxerr),'files_sha':files,'checks':checks,'scope':'Independent saved-array sums, prefix formulas, high-DAY definition, all coalition endpoints/masks, Shapley combinatorial formula and two-context nonadditivity. Forest routing belongs to root audit; no new raw labels read.'}
out['artifact_hashes']=artifact_hashes
with (H/'critic_arithmetic_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('CRITIC_ALL_PASS',len(checks),mask_total,flush=True)
