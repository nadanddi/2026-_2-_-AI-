"""Reuse the identical N9 backbone once; run each package's actual final function."""
import sys,json,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;B=Path.home()/'storage/nadan/farmai_img';L=B/'data/local'
def load(name,path):
    sys.path.insert(0,str(path.parent));sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
I={n:load(f'candidate{n}',HERE/f'n{n}/팜모니_나단{n}/inference.py') for n in (10,11)}
orig=load('original9',B/'submit_img09/팜모니_나단9/inference.py');m=I[10].load_model(str(HERE/'n10/팜모니_나단10/model'))
sys.path.insert(0,str(BASE/'im_submission_ops_20261016_v1/a'));from eval_support_v1 import metadata
report={};first=sys.argv[1]=='pub'
sets=('pub21',) if first else ('im_set_r2','im_set_r3','im_set_r4','im_set_r5')
for sn in sets:
    rr,ps=metadata(sn);paths=[str(p) for p in ps];order=sorted(range(len(paths)),key=lambda i:paths[i]);pred={n:np.full(len(ps),np.nan) for n in (9,10,11)}
    with ThreadPoolExecutor(8) as ex:
        lows={i:ex.submit(I[10]._safe_low,paths[i]) for i in order}
        for b in range(0,len(order),I[10].BATCH):
            ix=order[b:b+I[10].BATCH];imgs=[Image.open(paths[i]).convert('RGB') for i in ix];infos=I[10]._batch_gpu(m,imgs)
            for j,info in zip(ix,infos):
                info['std']=I[10]._std_format(paths[j]);low=lows[j].result()
                for n,mod in ((9,orig),(10,I[10]),(11,I[11])):pred[n][j]=mod._final(m,info,low)
            if b%400==0:print('PACKAGE_REPLAY',sn,b,'/',len(paths),flush=True)
    report[sn]={}
    for n in (10,11):
        with np.load(HERE/f'n{n}/expected_{sn}.npz') as z:
            assert z['files'].tolist()==[r['file'] for r in rr];y=z['y'];expected=z['expected'];base=z['base']
            diff=int(((pred[n]>=.5)!=(expected>=.5)).sum());bdiff=int(((pred[9]>=.5)!=(base>=.5)).sum())
            ba=lambda a:float(.5*((a[y==0]<.5).mean()+(a[y==1]>=.5).mean())) if (y==0).any() and (y==1).any() else None
            report[sn][str(n)]=dict(n=len(y),decision_mismatches=diff,baseline_decision_mismatches=bdiff,max_abs=float(abs(pred[n]-expected).max()),base_max_abs=float(abs(pred[9]-base).max()),cached_ba=ba(expected),package_ba=ba(pred[n]),correct=int(((pred[n]>=.5)==y).sum()))
            np.savez(HERE/f'n{n}/actual_{sn}.npz',files=z['files'],y=y,actual=pred[n],expected=expected,baseline_actual=pred[9],baseline_expected=base)
            assert diff==0 and bdiff==0,(sn,n,report[sn][str(n)])
            if sn=='pub21':assert report[sn][str(n)]['correct']==21
    print('REPLAY_OK',sn,report[sn],flush=True)
with (HERE/f'validation_{"pub" if first else "r2r5"}_v1.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print('VALIDATION_DONE',flush=True)
