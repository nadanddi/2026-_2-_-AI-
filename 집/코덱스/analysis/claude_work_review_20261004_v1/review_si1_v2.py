"""Readonly Claude SI1 refresh. EL1 filtered as strings before numeric use.
Uses only pre-existing public EC labels to authorize included observations.
"""
from pathlib import Path
import sys,csv,json,math,hashlib,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
SOURCE=ROOT/'집/클로드/research/ec3_SI1_sealed_active_interaction_v1.py'
CSV=ROOT/'집/클로드/research/local/ec3_SI1_all.csv'
PUBLIC=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    pub={}
    with PUBLIC.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['validator']=='DIAG10' and r['seed']=='7':assert r['row_id'] not in pub;pub[r['row_id']]=float(r['sub_ec'])
    assert len(pub)==8640
    selected=[]
    with CSV.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['validator']=='EL1':continue
            assert row['validator'] in ('DIAG10','DIAG10w') and row['row_id'] in pub
            r={k:row[k] for k in ['row_id','farm','day','hour','validator','validation_fold']}
            r.update({k:float(row[k]) for k in ['sub_ec']+[f'{p}_{s}' for p in ['base','dp'] for s in [13,505,6060]]})
            assert all(math.isfinite(r[k]) for k in ['sub_ec']+[f'{p}_{s}' for p in ['base','dp'] for s in [13,505,6060]])
            assert abs(r['sub_ec']-pub[r['row_id']])<=1e-12
            assert int(r['day'])>=179;selected.append(r)
    assert len(selected)==2208 and len({(r['validator'],r['row_id']) for r in selected})==2208
    scores=[]
    for v in ('DIAG10','DIAG10w'):
        rs=[r for r in selected if r['validator']==v];assert len(rs)==1104 and len({(r['farm'],r['day']) for r in rs})==46
        for s in [13,505,6060]:
            a=math.sqrt(math.fsum((r['base_'+str(s)]-r['sub_ec'])**2 for r in rs)/len(rs))
            b=math.sqrt(math.fsum((r['dp_'+str(s)]-r['sub_ec'])**2 for r in rs)/len(rs))
            na=float(np.sqrt(np.mean([(r['base_'+str(s)]-r['sub_ec'])**2 for r in rs])))
            nb=float(np.sqrt(np.mean([(r['dp_'+str(s)]-r['sub_ec'])**2 for r in rs])))
            assert max(abs(a-na),abs(b-nb))<=1e-12 and (b<a)==(nb<na)
            scores.append(dict(validator=v,seed=s,baseline=a,candidate=b,change_pct=100*(b/a-1)))
    rs=[r for r in selected if r['validator']=='DIAG10w'];groups=collections.defaultdict(list)
    basem=[];candm=[];y=[]
    for r in rs:
        b=math.fsum(r['base_'+str(s)] for s in [13,505,6060])/3
        c=math.fsum(r['dp_'+str(s)] for s in [13,505,6060])/3;t=r['sub_ec']
        groups[(r['farm'],int(r['day'])//5)].append((c-t)**2-(b-t)**2)
        basem.append(b);candm.append(c);y.append(t)
    keys=sorted(groups);sums=np.array([math.fsum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys])
    draw=np.random.default_rng(20261004).integers(len(keys),size=(20000,len(keys)))
    samples=sums[draw].sum(1)/counts[draw].sum(1)
    independent=[math.fsum(sums[j] for j in ix)/sum(counts[j] for j in ix) for ix in draw]
    assert np.max(abs(samples-independent))<=1e-12
    p=float(np.mean(samples>=0));assert p==sum(x>=0 for x in independent)/20000 and abs(p-.8862)<.000051
    assert any(r['candidate']>=r['baseline'] for r in scores)
    report=dict(status='PASS_PUBLIC_SI1_READONLY_RECHECK',rows=2208,score_cells=6,scores=scores,p_worse=p,seedmean_baseline=math.sqrt(math.fsum((a-b)**2 for a,b in zip(basem,y))/len(y)),seedmean_candidate=math.sqrt(math.fsum((a-b)**2 for a,b in zip(candm,y))/len(y)),public_fail=True,source_sha256=sha(SOURCE),input_sha256=sha(CSV),public_labels_sha256=sha(PUBLIC),verifier_sha256=sha(Path(__file__)),fit=0,predict=0,raw_EC_reads=0,test_reads=0,EL1_rescore=0,limitations=['R3S baseline differs from actual .8R3+.2PFN season_v2','same46days in two layouts are not independent labels','Author resampling pools calendar day//5 blocks across farms, distinct from our farm-stratified5observedday protocol','Only fixed3interactions tested; rejection does not establish no information in controls','EL1 numeric labels/predictions skipped; no new full training/source causal replay'])
    with (H/'si1_readonly_recheck_v2.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('PASS_PUBLIC_SI1_READONLY_RECHECK',p,report['seedmean_baseline'],report['seedmean_candidate'])
if __name__=='__main__':main()
