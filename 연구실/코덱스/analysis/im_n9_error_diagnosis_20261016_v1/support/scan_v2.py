"""Read fixed N9 caches/models and image diagnostics only; no fitting or adoption."""
import sys,os,csv,json,io,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,cv2
from PIL import Image,JpegImagePlugin
HERE=Path(__file__).resolve().parent
cv2.setNumThreads(1)
B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
sys.path.insert(0,str(B/'data/image')); sys.argv=[sys.argv[0]]
import im44_h_train_eval as H
M={}
with np.load(L/'preds/h_model_h2.npz') as z:
    for k in z.files:
        a,c=k.split('__'); M.setdefault(a,{})[c]=z[k]
with np.load(L/'preds/ip_model_v1.npz') as z: M['E_gr']={k.split('__')[1]:z[k] for k in z.files}
T=json.loads((L/'preds/ip_theta_v1.json').read_text())
buf=io.BytesIO(); Image.new('RGB',(64,64),(120,130,140)).save(buf,'JPEG',quality=92)
REF=Image.open(io.BytesIO(buf.getvalue())).quantization
def picture(p):
    with Image.open(p) as im:
        gate=im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==REF
        a=np.asarray(im.convert('RGB')); h,w=a.shape[:2]
    rgb=a.astype(np.float32); rr,gg,bb=rgb[...,0],rgb[...,1],rgb[...,2]; lum=(rr+gg+bb)/3
    sat=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)[...,1]
    return dict(gate=int(gate),width=w,height=h,brightness=float(lum.mean()),saturation=float(sat.mean()/255),
        brown=float(((rr>gg)&(gg-bb>8)&(lum>50)&(lum<200)).mean()),image_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def paths(sn):
    with (L/sn/'manifest.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    return rows,[L/sn/'img'/r['file'] for r in rows]
def nullable(v): return float(v) if np.isfinite(v) else ''
fields=['dataset','file','source','group','farm_code','manifest_farm_id','tar','label','kind','info','p6','p9','rc','gate','dg1','gap','mask_area','mask_mean_alpha','mask_width','mask_height','width','height','brightness','saturation','brown','Ef','Ef_theta','Ef_fire','expert_fires','image_sha256']
audit={}; selected=[]
with (HERE/'rows_v1.csv').open('x',newline='',encoding='utf-8-sig') as fo:
    wr=csv.DictWriter(fo,fieldnames=fields); wr.writeheader()
    for sn in ('im_set_realg','im_set_r2','im_set_r3','im_set_r4','im_set_r5'):
        rr,ps=paths(sn); F=H.load(sn)
        assert len(rr)==len(F['files']) and all(r['file']==f for r,f in zip(rr,F['files']))
        assert F['ok'].all()
        sc={**H.score(M,F),'E_gr':H.lrp(M['E_gr'],np.c_[F['low'],F['emb']])}
        # gate is deterministic image header metadata, independent of class/score.
        gs=[]
        for p in ps:
            with Image.open(p) as im: gs.append(im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==REF)
        gs=np.array(gs); p9=np.where(gs,H.apply(F['p6'],sc,T['theta'],T['theta_rc'],F['rc']),F['p6'])
        assert np.isfinite(p9).all() and ((0<=p9)&(p9<=1)).all()
        if sn=='im_set_r5':
            z=np.load(HERE.parent/'im_m2_20261016_v1/open/eval_im_set_r5.npz')
            assert np.array_equal(z['y'],np.array([int(r['label']) for r in rr]))
            assert np.array_equal(z['group'],np.array([r['group'] for r in rr]))
            ref=z['p9']
        else:
            with (HERE.parent/'im_meta_probe_20261016_v1'/f'pred_1601_{sn}.csv').open() as f: old=list(csv.DictReader(f))
            assert len(old)==len(rr) and all(r['file']==v['file'] and r['source']==v['source'] for r,v in zip(rr,old))
            ref=np.array([float(r['p9']) for r in old])
        assert np.allclose(p9,ref,rtol=0,atol=1e-12) and np.array_equal(p9>=.5,ref>=.5)
        ef=sc['E_f']>=np.where(F['rc'],T['theta_rc']['E_f'],T['theta']['E_f'])
        choices=[i for i,r in enumerate(rr) if sn=='im_set_realg' and int(r['label'])==0 and gs[i] and ef[i]]
        if sn=='im_set_realg':
            selected=[dict(rr[i],p9=float(p9[i]),p6=float(F['p6'][i]),rc=bool(F['rc'][i]),Ef=float(sc['E_f'][i]),theta=float(T['theta_rc']['E_f'] if F['rc'][i] else T['theta']['E_f'])) for i in choices]
            with (HERE/'realg_selected_v1.json').open('x',encoding='utf-8') as f: json.dump(selected,f,ensure_ascii=False,indent=2)
            print('SELECTED_REALG',len(selected),[r['file'] for r in selected],flush=True)
        todo=[i for i,r in enumerate(rr) if int(r['label'])==0 and (sn!='im_set_realg' or i in choices)]
        with ThreadPoolExecutor(4) as ex: pix=dict(zip(todo,ex.map(picture,[ps[i] for i in todo])))
        for i,r in enumerate(rr):
            v=dict(dataset=sn,file=r['file'],source=r['source'],group=r['group'],farm_code=r.get('farm_code',''),manifest_farm_id=r.get('farm_id',''),tar=r.get('tar',''),label=int(r['label']),kind=r['kind'],info=r.get('info',''),p6=float(F['p6'][i]),p9=float(p9[i]),rc=int(F['rc'][i]),gate=int(gs[i]),dg1=nullable(F['ff'][i][0]),gap=nullable(F['ff'][i][1]),
                mask_area='',mask_mean_alpha='',mask_width='',mask_height='',Ef=float(sc['E_f'][i]),Ef_theta=float(T['theta_rc']['E_f'] if F['rc'][i] else T['theta']['E_f']),Ef_fire=int(gs[i] and ef[i]),
                expert_fires=','.join(k for k in ('E_f','E_ps2','E_g7','E_gr') if gs[i] and sc[k][i]>= (T['theta_rc'][k] if F['rc'][i] else T['theta'][k])))
            if i in pix: v.update(pix[i])
            if sn=='im_set_r5' and r['kind'].startswith('partial'):
                mp=L/sn/'mask'/Path(r['file']).with_suffix('.png').name
                assert mp.is_file(),mp
                with Image.open(mp) as im: mask=np.asarray(im.convert('L'))
                v.update(mask_area=float((mask>127.5).mean()),mask_mean_alpha=float(mask.mean()/255),mask_width=mask.shape[1],mask_height=mask.shape[0])
            wr.writerow(v)
        fo.flush()
        yy=np.array([int(r['label']) for r in rr])
        audit[sn]=dict(n=len(rr),fp=int(((yy==0)&(p9>=.5)).sum()),fn=int(((yy==1)&(p9<.5)).sum()),high_fp=int(((yy==0)&(p9>=.6)).sum()),correct_real=int(((yy==0)&(p9<.5)).sum()),reference_max_abs=float(abs(ref-p9).max()),feature_sha256=hashlib.sha256((L/'feats_h'/f'{sn}.npz').read_bytes()).hexdigest())
        print('SET_DONE',sn,audit[sn],flush=True)
with (HERE/'scan_audit_v1.json').open('x',encoding='utf-8') as f: json.dump(audit,f,indent=2)
print('SCAN_DONE',flush=True)
