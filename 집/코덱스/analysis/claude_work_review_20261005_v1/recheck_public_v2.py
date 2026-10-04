"""Readonly public SG1/SG2/HK0 arithmetic. Skip EL1 before numeric parsing."""
from pathlib import Path
import sys,csv,json,math,hashlib,collections
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np
L=ROOT/'집/클로드/research/local'
PUB=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def rmse(a,y):
    r=math.sqrt(math.fsum((x-t)**2 for x,t in zip(a,y))/len(y))
    other=float(np.sqrt(np.mean((np.asarray(a)-y)**2)))
    assert abs(r-other)<=1e-12
    return r
def boot(rs,b,c,rng):
    groups=collections.defaultdict(list)
    for r,x,z in zip(rs,b,c):groups[r['farm']+'_'+str(int(r['day'])//5)].append((z-r['y'])**2-(x-r['y'])**2)
    kk=sorted(groups);ss=np.array([math.fsum(groups[k]) for k in kk]);nn=np.array([len(groups[k]) for k in kk])
    ix=rng.integers(0,len(kk),(20000,len(kk)))
    v=ss[ix].sum(1)/nn[ix].sum(1)
    manual=[math.fsum(ss[i] for i in ids)/sum(nn[i] for i in ids) for ids in ix]
    assert np.max(abs(v-manual))<=1e-12
    p=float(np.mean(v>=0));assert p==sum(x>=0 for x in manual)/20000
    return dict(p_worse=p,draws=20000,blocks=len(kk),resampling='author pooled farm_calendar_day//5',seedmean_baseline=rmse(b,[r['y'] for r in rs]),seedmean_candidate=rmse(c,[r['y'] for r in rs]))
def main():
    pub={r['row_id']:float(r['sub_ec']) for r in rows(PUB) if r['validator']=='DIAG10' and r['seed']=='7'}
    assert len(pub)==8640
    reports={};sourcehash={}
    for name,seeds,alt,expected in [('SG1',[17,606,7070],'DIAG10x',.1602),('SG2',[23,808,9090],'DIAG10y',0.)]:
        p=L/f'ec3_{name}_all.csv';rs=[]
        for r in rows(p):
            if r['validator']=='EL1':continue
            assert r['validator'] in ['DIAG10',alt] and r['row_id'] in pub
            q={k:r[k] for k in ['row_id','farm','day','hour','validator','validation_fold']};q['y']=float(r['sub_ec'])
            assert abs(q['y']-pub[r['row_id']])<=1e-12 and int(q['day'])>=179
            for k in [f'{a}_{s}' for a in ['base','sg'] for s in seeds]:q[k]=float(r[k]);assert math.isfinite(q[k])
            rs.append(q)
        assert len(rs)==2208 and len({(r['validator'],r['row_id']) for r in rs})==2208
        sc=[]
        for v in ['DIAG10',alt]:
            rr=[r for r in rs if r['validator']==v];assert len(rr)==1104 and len({(r['farm'],r['day']) for r in rr})==46
            for s in seeds:
                a=rmse([r[f'base_{s}'] for r in rr],[r['y'] for r in rr]);b=rmse([r[f'sg_{s}'] for r in rr],[r['y'] for r in rr])
                sc.append(dict(validator=v,seed=s,baseline=a,candidate=b,change_pct=100*(b/a-1)))
        rr=[r for r in rs if r['validator']==alt]
        bm=[math.fsum(r[f'base_{s}'] for s in seeds)/3 for r in rr];cm=[math.fsum(r[f'sg_{s}'] for s in seeds)/3 for r in rr]
        bb=boot(rr,bm,cm,np.random.default_rng(20261004));assert abs(bb['p_worse']-expected)<.000051
        reports[name]=dict(rows=len(rs),scores=sc,bootstrap=bb,all_public_six_improve=all(x['candidate']<x['baseline'] for x in sc),whole_adoption=False)
        sourcehash[name]=dict(csv=sha(p),source=sha(ROOT/(f'집/클로드/research/ec3_{name}_'+('signature_knn_level_v1.py' if name=='SG1' else 'reference_knn_level_v1.py'))))
    p=L/'hk0_rows_v1.csv';rs=[]
    for r in rows(p):
        assert r['validator']=='DIAG10' and r['row_id'] in pub
        q={k:r[k] for k in ['row_id','farm','day','hour']};q['y']=float(r['sub_ec']);assert abs(q['y']-pub[r['row_id']])<=1e-12
        for k in ['p','pm','a1','a2']:q[k]=float(r[k]) if r[k] else math.nan
        assert math.isfinite(q['p']) and math.isfinite(q['pm'])
        a=q['a1'];a2=q['a2'];delta=a-q['pm'];sg=q['p']+.5*delta if math.isfinite(a) and abs(delta)<=.3 else q['p']
        ga=math.isfinite(a) and q['pm']>=.9 and a>=1
        gb=ga and math.isfinite(a2) and a2>=1
        q.update(sg=sg,hga=q['p']+.5*delta if ga else sg,hgb=q['p']+.5*((a+a2)/2-q['pm']) if gb else sg)
        rs.append(q)
    assert len(rs)==8640 and len({r['row_id'] for r in rs})==8640
    days=collections.defaultdict(list)
    for r in rs:days[(r['farm'],r['day'])].append(r)
    for dd in days.values():
        dd.sort(key=lambda r:int(r['hour']));assert len(dd)==24
        dm=math.fsum(r['y'] for r in dd)/24
        for i,r in enumerate(dd):r['dm']=dm;assert abs(r['pm']-math.fsum(x['p'] for x in dd[:i+1])/(i+1))<=1e-12
    seg=[]
    for name,sel in [('all',rs),('high',[r for r in rs if r['dm']>=1]),('ordinary',[r for r in rs if r['dm']<1]),('late',[r for r in rs if int(r['day'])>=179])]:
        seg.append(dict(segment=name,rows=len(sel),days=len({(r['farm'],r['day']) for r in sel}),**{k:rmse([r[k] for r in sel],[r['y'] for r in sel]) for k in ['p','sg','hga','hgb']}))
    rng=np.random.default_rng(20261005);boots={k:boot(rs,[r['sg'] for r in rs],[r[k] for r in rs],rng) for k in ['hga','hgb']}
    assert abs(boots['hga']['p_worse']-.2006)<.000051 and abs(boots['hgb']['p_worse']-.0951)<.000051
    reports['HK0']=dict(rows=len(rs),segments=seg,bootstrap=boots,adoption=False)
    sourcehash['HK0']=dict(csv=sha(p),source=sha(ROOT/'집/클로드/research/hk0_high_gate_anchor_v1.py'))
    result=dict(status='PASS_PUBLIC_ARITHMETIC_ONLY',reports=reports,sources=sourcehash,public_sha256=sha(PUB),verifier_sha256=sha(Path(__file__)),fit=0,predict=0,raw_EC_reads=0,test_reads=0,EL1_rescore=0,limitations=['source/runtime leakage audit separate; arithmetic PASS is not model adoption','R3S differs from actual EC season_v2','same46days reused in different layouts','author bootstrap differs from Codex protocol','p=0 in20000 draws is not true probability0','SG2C scores log only; no candidate saved CSV replay'])
    with (H/'public_recheck_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('PASS_PUBLIC_ARITHMETIC_ONLY', {k:v['bootstrap'] for k,v in reports.items() if k!='HK0'})
if __name__=='__main__':main()
