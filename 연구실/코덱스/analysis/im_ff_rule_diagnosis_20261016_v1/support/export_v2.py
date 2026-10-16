"""CPU-only cached-data export; no test521 paths or model modifications."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'im_submission_ops_20261016_v1/a'))
from eval_support_v1 import H,metadata,gate,L,B,read
SETS=('im_set_r2','im_set_r3','im_set_r4','im_set_r5','jig500','im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444','sealed_real','sealed_brown','sealed_bright','im_set_newreal','im_set_realg','im_set_realn','pub21')
cfg_path=B/'submit_img09/팜모니_나단9/model/config.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8'));T=cfg['finefp'];TH=cfg['h']
M={}
with np.load(L/'preds/h_model_h2.npz') as z:
    for k in z.files:a,c=k.split('__');M.setdefault(a,{})[c]=z[k]
with np.load(L/'preds/ip_model_v1.npz') as z:gr={k.split('__')[1]:z[k] for k in z.files}
OUT=HERE/'raw_v2';OUT.mkdir(exist_ok=False)
audit={};keys=('dg1','gap','lg','le','ge')
for sn in SETS:
    F=H.load(sn);rr,ps=metadata(sn);assert F['ok'].all() and F['files'].tolist()==[r['file'] for r in rr]
    gg=np.array([gate(p) for p in ps]);ff=F['ff'];dg,gap_,lg,le,ge=ff.T;valid=np.isfinite(ff).all(1)
    match=valid&(dg<=T['m'])&(gap_>=T['gap']);fake=match&((lg>T['f_g'])|(le>T['f_e']))
    safe=match&~fake&(lg<=T['r_g'])&(le<=T['r_e'])&(abs(ge)<=T['r_glob'])
    rc=safe&(F['ppre']<.8)
    pc=np.where(fake,np.maximum(F['ppre'],.95),np.where(rc,np.minimum(F['ppre'],.3),F['ppre']))
    assert np.array_equal(rc,F['rc']),(sn,'rc mismatch')
    assert np.allclose(pc,F['p6'],rtol=0,atol=1e-12),(sn,'p6 mismatch',np.max(abs(pc-F['p6'])))
    sc={**H.score(M,F),'E_gr':H.lrp(gr,np.c_[F['low'],F['emb']])}
    p9=np.where(gg,H.apply(F['p6'],sc,TH['theta'],TH['theta_rc'],rc),F['p6'])
    ref=BASE/'im_meta_probe_20261016_v1'/f'pred_1601_{sn}.csv'
    if sn=='im_set_r5':
        with np.load(BASE/'im_m2_20261016_v1/open/eval_im_set_r5.npz') as z:
            assert np.array_equal([int(r['label']) for r in rr],z['y']) and np.array_equal([r['group'] for r in rr],z['group'])
            assert np.allclose(p9,z['p9'],rtol=0,atol=1e-12)
    else:
        assert ref.is_file();br=read(ref)
        assert [r['file'] for r in rr]==[r['file'] for r in br] and [int(r['label']) for r in rr]==[int(r['label']) for r in br]
        assert np.allclose(p9,[float(r['p9']) for r in br],rtol=0,atol=1e-12)
    p6new=np.where(fake,np.maximum(F['ppre'],.95),np.where(safe,np.minimum(F['ppre'],.3),F['ppre']))
    pnew=np.where(gg,H.apply(p6new,sc,TH['theta'],TH['theta_rc'],safe),p6new)
    np.savez(OUT/f'{sn}.npz',files=F['files'],source=np.array([r.get('source','') for r in rr]),group=np.array([r.get('group','') for r in rr]),kind=np.array([r.get('kind','unknown') for r in rr]),y=np.array([int(r['label']) for r in rr]),ff=ff,valid=valid,ppre=F['ppre'],p6=F['p6'],p9=p9,rc=rc,gate=gg,fake_branch=fake,safe=safe,p6_no_pcap=p6new,p9_no_pcap=pnew,rc_no_pcap=safe,**sc)
    audit[sn]=dict(n=len(rr),ff_valid=int(valid.sum()),rc_baseline_equal=True,p6_baseline_equal=True,p9_baseline_equal=True,features_sha256=hashlib.sha256((L/'feats_h'/f'{sn}.npz').read_bytes()).hexdigest())
    print('EXPORT',sn,len(rr),flush=True)
with (HERE/'config_v2.json').open('x',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
with (HERE/'audit_export_v2.json').open('x',encoding='utf-8') as f:json.dump(dict(sets=audit,config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),E4_access=False,gpu_used=False),f,ensure_ascii=False,indent=2)
print('FF_EXPORT_DONE',flush=True)
