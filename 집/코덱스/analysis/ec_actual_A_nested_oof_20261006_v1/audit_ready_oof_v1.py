"""Read-only independent audit of complete outer-training OOF groups, no score."""
from pathlib import Path
import sys,json,csv,hashlib,math,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env;import env_extra
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ar(x):
    x=np.asarray(x,dtype=float);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def component(name):
    p=OUT/'components'/Path(name).stem/name;m=load(p.with_suffix('.json'));assert sha(p)==m['sha']
    with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    return d
def main():
    p=load(H/'preparation_v4.json');registration=load(H/'registration_v4.json')
    for name,h in registration['hashes'].items():assert sha(H/name)==h,name
    source=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv';assert sha(source)==p['inputs']['public_oof']
    public={r['row_id']:float(r['sub_ec']) for r in read(source) if r['validator']=='DIAG10' and int(r['seed'])==7};assert len(public)==8640
    records=collections.defaultdict(list)
    for e in p['records']:records[e['v'],e['k']].append(e)
    ready=[key for key in records if all((OUT/f'OOF_{key[0]}_{key[1]}_{s}.csv').is_file() for s in [7,101,2024])];assert ready
    gaps=dict(R3=0.,PFN=0.,raw_A=0.,A=0.,prefix_A=0.,y_day=0.,public_y=0.);files=[];coverage=[];counts=[]
    for v,k in ready:
        es=records[v,k];assert len(es)==4;outertr=es[0]['outer_train_ids'];oq=set(es[0]['outer_query_ids'])
        assert sorted(rid for e in es for rid in e['query_ids'])==sorted(outertr)
        for s in [7,101,2024]:
            path=OUT/f'OOF_{v}_{k}_{s}.csv';data=read(path);files.append(dict(path=path.name,sha=sha(path)));assert [r['row_id'] for r in data]==outertr
            index={r['row_id']:r for r in data};assert len(index)==len(data) and not set(index)&oq
            for r in data:
                rid=r['row_id'];assert r['farm']==rid[:3] and int(r['day'])==int(rid[4:7]) and int(r['hour'])==int(rid[8:10]);assert r['v']==v and int(r['k'])==k and int(r['s'])==s
                assert all(math.isfinite(float(r[n])) for n in ['y','y_day','A','raw_r3','raw_pfn','raw_A','prefix_A','clip_lo','clip_hi'])
                gaps['public_y']=max(gaps['public_y'],abs(float(r['y'])-public[rid]))
            for e in es:
                j=e['j'];assert ar([float(index[rid]['y']) for rid in e['query_ids']])==e['query_target_sha']
                r3=component(f'{v}_{k}_{j}_r3_{s}.npz');pfns=[component(f'{v}_{k}_{j}_pfn_{c}.npz') for c in [1,2,3,4]]
                assert r3['row_id'].tolist()==e['query_ids'];assert all(z['row_id'].tolist()==e['query_ids'] for z in pfns)
                for i,rid in enumerate(e['query_ids']):
                    r=index[rid];assert int(r['j'])==j and [float(r['clip_lo']),float(r['clip_hi'])]==e['bounds']
                    rr=math.fsum([.6*float(r3['et'][i]),.3*float(r3['lgb'][i]),.1*float(r3['mlp'][i])]);pp=math.fsum(float(z['raw'][i]) for z in pfns)/4;raw=.8*rr+.2*pp
                    for key,a,b in [('R3',rr,float(r['raw_r3'])),('PFN',pp,float(r['raw_pfn'])),('raw_A',raw,float(r['raw_A']))]:gaps[key]=max(gaps[key],abs(a-b))
            daily=collections.defaultdict(list)
            for r in data:daily[r['farm'],int(r['day'])].append(r)
            for key,g in daily.items():
                g.sort(key=lambda r:int(r['hour']));assert [int(r['hour']) for r in g]==list(range(24));assert len({r['j'] for r in g})==1
                mean=math.fsum(float(r['y']) for r in g)/24;hist=[];ah=[]
                for r in g:
                    hist.append(float(r['raw_A']));a=min(float(r['clip_hi']),max(float(r['clip_lo']),.5*hist[-1]+.5*math.fsum(hist)/len(hist)));ah.append(float(r['A']));prefix=math.fsum(ah)/len(ah)
                    for name,a0,b0 in [('A',a,float(r['A'])),('prefix_A',prefix,float(r['prefix_A'])),('y_day',mean,float(r['y_day']))]:gaps[name]=max(gaps[name],abs(a0-b0))
                    flags=dict(high=int(mean>=1),hard_high=int(mean>=1 and prefix<1.2),missed_high=int(mean>=1 and prefix<.9),hard_low=int(mean<1 and prefix>=.9))
                    assert all(int(r[name])==val for name,val in flags.items())
            coverage.append(dict(v=v,k=k,s=s,rows=len(data),days=len(daily)))
            for h in [0,6,12,23]:
                hh=[r for r in data if int(r['hour'])==h];counts.append(dict(v=v,k=k,s=s,hour=h,n=len(hh),high=sum(int(r['high']) for r in hh),hard_high=sum(int(r['hard_high']) for r in hh),missed_high=sum(int(r['missed_high']) for r in hh),hard_low=sum(int(r['hard_low']) for r in hh)))
    assert max(gaps.values())<1e-12,gaps
    result=dict(status='PASS_READY_OUTER_OOF_ONLY',outer_groups=len(ready),oof_csv=len(files),total_outer_groups=20,total_oof_csv=60,hourly_occurrences=sum(z['rows'] for z in coverage),maxdiff=gaps,coverage=coverage,files=files,case_counts=counts,partial=True,performance_scored=False,limitations=['Completion-order sample; no extrapolation to pending folds','Repeated day occurrences across outer folds/seeds are not independent samples','Case counts are training-sample diagnostics, not classifier validation performance'])
    with (H/'ready_oof_audit_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({n:result[n] for n in ['status','outer_groups','oof_csv','hourly_occurrences','maxdiff']},ensure_ascii=False))
if __name__=='__main__':main()
