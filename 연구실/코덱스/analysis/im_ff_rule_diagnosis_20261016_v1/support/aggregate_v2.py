"""Frozen E1/E2 midpoint and E3 final descriptive table; no adoption."""
import sys,json,csv,tarfile,io,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent.parent;ROOT=HERE.parents[3]
LOCAL=ROOT/'연구실/코덱스/local/im_ff_rule_diagnosis_20261016_v1';mode=sys.argv[1]
cfg=json.loads((HERE/'support/config_v2.json').read_text(encoding='utf-8'));T=cfg['finefp']
D={}
with tarfile.open(LOCAL/'raw_v2.tar') as tf:
    for member in tf.getmembers():
        if member.name.endswith('.npz'):
            with np.load(io.BytesIO(tf.extractfile(member).read())) as z:D[Path(member.name).stem]={k:z[k] for k in z.files}
core=('im_set_r2','im_set_r3','im_set_r4','im_set_r5')
out=[]
def row(section,dataset,group,item,n=None,den=None,value=None,**kwargs):
    out.append(dict(구분=section,자료=dataset,집단=group,항목=item,수=n,분모=den,값=value,**kwargs))
def criteria(z):
    dg,gap,lg,le,ge=z['ff'].T
    return dict(ff유효=z['valid'],dg1=dg<=T['m'],gap=gap>=T['gap'],가짜우선분기아님=~z['fake_branch'],lg=lg<=T['r_g'],le=le<=T['r_e'],ge=abs(ge)<=T['r_glob'],ppre=z['ppre']<.8)
block=Counter();sole=Counter();first=Counter();comb=Counter();detail=[]
for sn in core:
    z=D[sn];high=(z['y']==0)&(z['p9']>=.6);C=criteria(z);names=list(C)
    for i in np.flatnonzero(high):
        fails=[k for k in names if not C[k][i]]
        if z['rc'][i]:assert not fails;comb['기존rc발동·후단전문가오판']+=1
        else:
            assert fails
            block.update(fails);first[fails[0]]+=1;comb['|'.join(fails)]+=1
            if len(fails)==1:sole[fails[0]]+=1
        row('E1 개별40',sn,'고확신진짜','진짜확정차단',1,1,파일=str(z['files'][i]),kind=str(z['kind'][i]),p9=float(z['p9'][i]),ppre=float(z['ppre'][i]),rc=int(z['rc'][i]),차단='|'.join(fails) or '기존rc발동·후단전문가오판',**{k:float(v) for k,v in zip(('dg1','gap','lg','le','ge'),z['ff'][i])})
assert sum(1 for r in out if r['구분']=='E1 개별40')==40
for k in criteria(D[core[0]]):row('E1 조건집계','R2~R5','고확신진짜',k,block[k],40,단독차단수=sole[k],최초차단수=first[k])
for k,v in sorted(comb.items()):row('E1 차단조합','R2~R5','고확신진짜',k,v,40)
e2={}
def metrics(y,b,c):
    bb=b>=.5;cc=c>=.5
    ba=lambda q:float(.5*((~q[y==0]).mean()+q[y==1].mean())) if (y==0).any() and (y==1).any() else None
    return dict(고친진짜=int(((y==0)&bb&~cc).sum()),새가짜놓침=int(((y==1)&bb&~cc).sum()),고친가짜=int(((y==1)&~bb&cc).sum()),새진짜오판=int(((y==0)&~bb&cc).sum()),기존BA=ba(bb),반사실BA=ba(cc),기존정답=int((bb==y).sum()),반사실정답=int((cc==y).sum()))
for sn,z in D.items():
    assert np.array_equal(z['p9_no_pcap'][~z['gate']],z['p6_no_pcap'][~z['gate']])
    newrc=z['rc_no_pcap']&~z['rc'];assert not (z['rc']&~z['rc_no_pcap']).any()
    e2[sn]=metrics(z['y'],z['p9'],z['p9_no_pcap']);e2[sn]['신규rc']=int(newrc.sum())
    row('E2 최종판정',sn,'전체','ppre제한만제거',len(z['y']),len(z['y']),**e2[sn])
    for yy in (0,1):
        for kind in sorted(set(z['kind'][z['y']==yy])):
            ix=(z['y']==yy)&(z['kind']==kind)
            row('E2 유형별',sn,'진짜' if yy==0 else '가짜','ppre제한만제거',int(newrc[ix].sum()),int(ix.sum()),kind=str(kind),p6가짜에서진짜=int((ix&(z['p6']>=.5)&(z['p6_no_pcap']<.5)).sum()),최종가짜에서진짜=int((ix&(z['p9']>=.5)&(z['p9_no_pcap']<.5)).sum()),**metrics(z['y'][ix],z['p9'][ix],z['p9_no_pcap'][ix]))
    for i in np.flatnonzero(newrc):
        row('E2 신규rc개별',sn,'진짜' if z['y'][i]==0 else '가짜','ppre제한만제거',1,1,파일=str(z['files'][i]),kind=str(z['kind'][i]),ppre=float(z['ppre'][i]),p9=float(z['p9'][i]),반사실p9=float(z['p9_no_pcap'][i]),**{k:float(v) for k,v in zip(('dg1','gap','lg','le','ge'),z['ff'][i])})
mid=dict(high_real=40,blocked=dict(block),sole=dict(sole),first=dict(first),combination=dict(comb),E2=e2,E4_access=False)
if mode=='mid':
    with (HERE/'support/midpoint_v1.json').open('x',encoding='utf-8') as f:json.dump(mid,f,ensure_ascii=False,indent=2)
    print(json.dumps(mid,ensure_ascii=False));sys.exit(0)
assert mode=='final'
def values(z):
    dg,gap,lg,le,ge=z['ff'].T
    a=np.full(len(ge),np.nan);b=a.copy();nz=(ge!=0)&np.isfinite(ge)
    a[nz]=lg[nz]/ge[nz];b[nz]=le[nz]/ge[nz]
    return dict(dg1=dg,gap=gap,lg=lg,le=le,ge=ge,abs_ge=abs(ge),lg_div_ge=a,le_div_ge=b,lg_div_absge_floor1=lg/np.maximum(abs(ge),1),le_div_absge_floor1=le/np.maximum(abs(ge),1))
def joined(parts):return {k:np.concatenate([z[k] for z in parts]) for k in ('y','ff','p9','kind','valid')}
allcore=joined([D[s] for s in core]);part=(allcore['y']==1)&(allcore['p9']<.5)&np.isin(allcore['kind'],('partial_style','partial_gen_v7','partial_gen_v7s'));assert int(part.sum())==147
groups=[('R2~R5','놓친부분편집147',allcore,part),('R2~R5','정답진짜',allcore,(allcore['y']==0)&(allcore['p9']<.5)),('R2~R5','고확신오판진짜',allcore,(allcore['y']==0)&(allcore['p9']>=.6))]
for sn,z in D.items():
    if (z['y']==0).any():groups.append((sn,'전체진짜',z,z['y']==0))
for sn,group,z,ix in groups:
    for name,v in values(z).items():
        valid=ix&np.isfinite(v);vv=v[valid];q=np.quantile(vv,[0,.25,.5,.75,.95,.99,1]) if len(vv) else [None]*7
        row('E3 분포',sn,group,name,int(valid.sum()),int(ix.sum()),결측수=int((ix&~np.isfinite(v)).sum()),ge영분모수=int((ix&(z['ff'][:,4]==0)).sum()),**dict(zip(('q0','q25','q50','q75','q95','q99','q100'),[float(a) if a is not None else None for a in q])))
    if group=='놓친부분편집147':
        for kind in sorted(set(z['kind'][ix])):row('E3 부분편집분모',sn,group,str(kind),int((ix&(z['kind']==kind)).sum()),147)
cal=joined([D[s] for s in ('im_set_newreal','im_set_realg','im_set_realn')]);assert len(cal['y'])==1395 and not cal['y'].any()
scores=('dg1','gap','lg','le','abs_ge','lg_div_absge_floor1','le_div_absge_floor1')
E3={};calvals=values(cal)
for scope in ('ff유효전체','ff유효_gap기존이상'):
    def elig(z):return z['valid']&((z['ff'][:,1]>=T['gap']) if scope!='ff유효전체' else True)
    for name in scores:
        v=calvals[name];ix=elig(cal)&np.isfinite(v);n=len(v);allowed=int(np.floor(.0025*n));sv=np.sort(v[ix])[::-1]
        threshold=float(np.nextafter(sv[allowed],np.inf)) if len(sv)>allowed else (float(sv.min()) if len(sv) else None)
        assert threshold is not None and np.isfinite(threshold)
        alerts=ix&(v>=threshold);assert int(alerts.sum())<=allowed
        tag=scope+'|'+name;pc=values(allcore)[name];caught=part&elig(allcore)&(pc>=threshold);E3[tag]=dict(threshold=threshold,calibration_denominator=n,calibration_eligible=int(ix.sum()),calibration_alerts=int(alerts.sum()),allowed=allowed,caught147=int(caught.sum()),caught_kind={str(k):int((caught&(allcore['kind']==k)).sum()) for k in sorted(set(allcore['kind'][part]))})
        row('E3 .25%문턱',scope,'고정보정진짜1395',name,int(alerts.sum()),n,threshold,적격수=int(ix.sum()),허용수=allowed,부분편집탐지=int(caught.sum()),부분편집분모=147,주의='단독규칙; 합집합/최적선택/채택 아님; JPEG게이트무관')
        for sn,z in D.items():
            a=elig(z)&(values(z)[name]>=threshold);pred=(z['p9']>=.5)|a;mm=metrics(z['y'],z['p9'],pred.astype(float));real=z['y']==0
            row('E3 자료별경보',sn,scope,name,int(a.sum()),len(a),threshold,진짜경보=int((a&real).sum()),진짜분모=int(real.sum()),진짜경보율=float((a&real).sum()/real.sum()) if real.any() else None,부분편집탐지=int((a&(z['y']==1)&(z['p9']<.5)&np.isin(z['kind'],('partial_style','partial_gen_v7','partial_gen_v7s'))).sum()),**mm)
cols=[]
for r in out:
    for k in r:
        if k not in cols:cols.append(k)
with (HERE/'진단표_v1.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(out)
summary=dict(**mid,E3=E3,table_rows=len(out),no_adoption=True,no_package=True)
with (HERE/'support/result_v1.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
print('TABLE_DONE',len(out));print(json.dumps(dict(E1=mid['sole'],E2_core={s:e2[s] for s in core},E3=E3),ensure_ascii=False))
