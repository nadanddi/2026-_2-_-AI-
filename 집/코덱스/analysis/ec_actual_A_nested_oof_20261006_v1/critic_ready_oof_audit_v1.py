"""Read-only independent audit of fixed ready folds; never fits or scores models."""
from pathlib import Path
import sys,csv,json,math,hashlib,collections
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
OUT=ROOT/'집/코덱스/local'/H.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ar(x):return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
p=load(H/'preparation_v4.json')
for name,digest in load(H/'registration_v4.json')['hashes'].items():assert sha(H/name)==digest
for name,digest in p['dependencies'].items():assert sha(ROOT/name)==digest
publicpath=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
assert sha(publicpath)==p['inputs']['public_oof']
y={r['row_id']:float(r['sub_ec']) for r in rows(publicpath) if r['validator']=='DIAG10' and int(r['seed'])==7}
assert len(y)==8640
seeds=[7,101,2024];contexts={};files=[];counts=[];firstpfn=[]
gaps={key:0. for key in ['r3_mix','pfn_bag','raw_r3','raw_A','A','prefix_A','y_day']}
for e in p['records']:
    if e['v']!='DIAG10' or e['k'] not in [0,1,2,3]:continue
    k,j=e['k'],e['j'];t=e['train_ids'];q=e['query_ids'];outer=set(e['outer_query_ids'])
    assert not(set(t)&set(q) or (set(t)|set(q))&outer)
    banned={(rid[:3],int(rid[4:7])+off) for rid in q for off in [-1,0,1]}
    assert not {(rid[:3],int(rid[4:7])) for rid in t}&banned
    assert ar(np.array([y[rid] for rid in t]))==e['train_target_sha']
    assert ar(np.array([y[rid] for rid in q]))==e['query_target_sha']
    assert [min(y[rid] for rid in t),max(y[rid] for rid in t)]==e['bounds']
    rs={};ps=[]
    for kind,ss in [('r3',seeds),('pfn',[1,2,3,4])]:
        for s in ss:
            name=f'DIAG10_{k}_{j}_{kind}_{s}'
            fp=OUT/'components'/name/(name+'.npz');m=load(fp.with_suffix('.json'))
            assert sha(fp)==m['sha']
            sig=m['signature']
            assert sig['record_sha']==hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
            assert sig['prepared_sha']==sha(H/'preparation_v4.json') and sig['seed']==s
            assert sig['runtime']==p['runtime'] and sig['dependencies']==p['dependencies']
            assert sig['kind']==('R3 .6ET+.3LGB+.1MLP' if kind=='r3' else 'TabPFN V2 CPU float32 context2000 estimator4')
            with np.load(fp,allow_pickle=False) as z:d={key:z[key] for key in z.files}
            assert d['row_id'].tolist()==q and d['train_row_id'].tolist()==t
            assert d['raw'].shape==(len(q),) and np.isfinite(d['raw']).all()
            if kind=='pfn':
                ix=np.random.default_rng(s).choice(len(t),min(2000,len(t)),replace=False)
                assert d['context_row_id'].tolist()==[t[int(i)] for i in ix]
                cs=set(d['context_row_id']);assert len(cs)==min(2000,len(t)) and cs<=set(t) and not cs&(set(q)|outer)
                assert m['details']['first8_batch_invariance'] is True
                ps.append(d['raw'])
                if k==0 and j==0:
                    firstpfn.append(dict(seed=s,context_rows=len(cs),query_rows=len(q),repeat_maxdiff=m['details']['repeat_maxdiff'],first8_batch_invariance=m['details']['first8_batch_invariance']))
            else:
                for key in ['et','lgb','mlp']:assert d[key].shape==(len(q),) and np.isfinite(d[key]).all()
                assert d['train_raw'].shape==(len(t),) and np.isfinite(d['train_raw']).all()
                mixed=np.array([math.fsum([.6*float(a),.3*float(b),.1*float(c)]) for a,b,c in zip(d['et'],d['lgb'],d['mlp'])])
                gaps['r3_mix']=max(gaps['r3_mix'],float(np.max(abs(mixed-d['raw']))));rs[s]=d['raw']
            files.append(dict(path=str(fp.relative_to(OUT)),sha=m['sha']))
    bag=[math.fsum(float(z[i]) for z in ps)/4 for i in range(len(q))]
    contexts[k,j]=(e,rs,bag)
assert len(contexts)==16 and len(files)==112
for k in [0,1,2,3]:
    recs=[contexts[k,j][0] for j in range(4)]
    tr=recs[0]['outer_train_ids'];assert collections.Counter(rid for e in recs for rid in e['query_ids'])==collections.Counter(tr)
    for s in seeds:
        fp=OUT/f'OOF_DIAG10_{k}_{s}.csv';data=rows(fp)
        assert [r['row_id'] for r in data]==tr
        idx={r['row_id']:r for r in data};assert len(idx)==len(data)
        groups=collections.defaultdict(list)
        for j in range(4):
            e,rs,bag=contexts[k,j]
            for i,rid in enumerate(e['query_ids']):
                r=idx[rid];assert r['farm']==rid[:3] and int(r['day'])==int(rid[4:7]) and int(r['hour'])==int(rid[8:10])
                assert r['v']=='DIAG10' and int(r['k'])==k and int(r['j'])==j and int(r['s'])==s
                assert all(math.isfinite(float(r[key])) for key in ['y','y_day','A','raw_r3','raw_pfn','raw_A','clip_lo','clip_hi','prefix_A'])
                assert float(r['y'])==y[rid] and [float(r['clip_lo']),float(r['clip_hi'])]==e['bounds']
                raw=.8*float(rs[s][i])+.2*bag[i]
                gaps['pfn_bag']=max(gaps['pfn_bag'],abs(float(r['raw_pfn'])-bag[i]))
                gaps['raw_r3']=max(gaps['raw_r3'],abs(float(r['raw_r3'])-float(rs[s][i])))
                gaps['raw_A']=max(gaps['raw_A'],abs(float(r['raw_A'])-raw))
                groups[r['farm'],int(r['day'])].append((r,raw))
        for key,g in groups.items():
            g.sort(key=lambda z:int(z[0]['hour']));assert [int(r['hour']) for r,raw in g]==list(range(24))
            assert len({r['j'] for r,raw in g})==1
            yy=math.fsum(float(r['y']) for r,raw in g)/24;hist=[];phist=[]
            for r,raw in g:
                hist.append(raw);pred=min(float(r['clip_hi']),max(float(r['clip_lo']),.5*raw+.5*math.fsum(hist)/len(hist)))
                phist.append(pred);prefix=math.fsum(phist)/len(phist)
                gaps['A']=max(gaps['A'],abs(float(r['A'])-pred));gaps['prefix_A']=max(gaps['prefix_A'],abs(float(r['prefix_A'])-prefix))
                gaps['y_day']=max(gaps['y_day'],abs(float(r['y_day'])-yy))
                expected={'high':int(yy>=1),'hard_high':int(yy>=1 and prefix<1.2),'missed_high':int(yy>=1 and prefix<.9),'hard_low':int(yy<1 and prefix>=.9)}
                for flag,value in expected.items():assert int(r[flag])==value
        counts.append(dict(validator='DIAG10',outer_fold=k,seed=s,rows=len(data),days=len(groups),cases=sum(int(r['hour']) in [0,6,12,23] for r in data)))
        files.append(dict(path=fp.name,sha=sha(fp)))
assert max(gaps.values())<1e-12
assert firstpfn[0]['repeat_maxdiff']==0.0 and all(z['repeat_maxdiff'] is None for z in firstpfn[1:])
proof=dict(status='PASS_READY_SUBSET_ONLY_NOT_FULL_MODEL',completed_outer_folds=[0,1,2,3],validator='DIAG10',inner_contexts=16,components=112,oof_files=12,hourly_occurrences=sum(z['rows'] for z in counts),case_occurrences=sum(z['cases'] for z in counts),gaps=gaps,first_pfn=firstpfn,coverage=counts,files=files,model_fit=False,performance_scoring=False,worker_modified=False,limitations=['Only fixed completed DIAG10 folds 0-3; not all 80 contexts','Repeat and batch evidence are recorded worker checks, not independent refits','No RMSE or classifier improvement assessed'])
with (H/'critic_ready_oof_audit_v1.json').open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2)
print(json.dumps({k:proof[k] for k in ['status','inner_contexts','components','oof_files','hourly_occurrences','case_occurrences','gaps','first_pfn']}))
