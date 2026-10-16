"""Recompute actual ff through package code; reuse completed actual N9 probabilities."""
import sys,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;team=HERE/'build_upper_v1/n10/팜모니_나단10'
sys.path.insert(0,str(team));sp=importlib.util.spec_from_file_location('upper_package',team/'inference.py');I=importlib.util.module_from_spec(sp);sp.loader.exec_module(I)
m=I.load_model(str(team/'model'))
# Only the unchanged ff/DB path is needed: prior full replay already produced actual N9.
m['visual']=None;m['b16']=None
sys.path.insert(0,str(BASE/'im_submission_ops_20261016_v1/a'));from eval_support_v1 import metadata
R={}
for sn in ('im_set_r2','im_set_r3','im_set_r4','im_set_r5'):
    rr,ps=metadata(sn);order=sorted(range(len(ps)),key=lambda i:str(ps[i]));ff=np.full((len(ps),5),np.nan)
    for b in range(0,len(order),I.BATCH):
        ix=order[b:b+I.BATCH];imgs=[Image.open(ps[i]).convert('RGB') for i in ix];infos=I._batch_gpu(m,imgs)
        for j,info in zip(ix,infos):ff[j]=info.get('ff',np.full(5,np.nan))
        if b%800==0:print('ACTUAL_FF',sn,b,'/',len(ps),flush=True)
    old=HERE/f'build_v2/n11/actual_{sn}.npz';new=HERE/f'build_upper_v1/n10/expected_{sn}.npz'
    with np.load(old) as a,np.load(new) as z:
        assert a['files'].tolist()==z['files'].tolist()==[r['file'] for r in rr]
        base=a['baseline_actual'];y=z['y'];exp=z['expected']
        alarm=np.isfinite(ff).all(1)&(ff[:,1]>=3.4739584922790527)&(ff[:,0]>1.0911458730697632)&(ff[:,0]<2.3890950679779053)
        actual=np.maximum(base,np.where(alarm,.6,0));assert np.isfinite(actual).all() and ((0<=actual)&(actual<=1)).all()
        diff=int(((actual>=.5)!=(exp>=.5)).sum());assert diff==0,(sn,diff)
        ba=lambda p:float(.5*((p[y==0]<.5).mean()+(p[y==1]>=.5).mean()))
        R[sn]=dict(n=len(y),decision_mismatches=diff,package_ba=ba(actual),cached_ba=ba(exp),max_abs=float(abs(actual-exp).max()),ff_valid=int(np.isfinite(ff).all(1).sum()),backbone_reused=True,actual_ff_recomputed=True)
        np.savez(HERE/f'build_upper_v1/n10/actual_upper_{sn}.npz',files=z['files'],y=y,ff=ff,actual=actual,expected=exp,baseline_actual=base)
    print('UPPER_REPLAY_OK',sn,R[sn],flush=True)
with (HERE/'validation_r2r5_upper_v1.json').open('x',encoding='utf-8') as f:json.dump(R,f,ensure_ascii=False,indent=2)
