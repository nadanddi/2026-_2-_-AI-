"""C1 one-shot E_f refit; first fold checkpoint before continuation/evaluation."""
import sys,csv,json,io,warnings,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,JpegImagePlugin
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error',category=ConvergenceWarning)
HERE=Path(__file__).resolve().parent; B=Path.home()/'storage/nadan/farmai_img'; L=B/'data/local'
sys.path.insert(0,str(B/'data/image'));mode=sys.argv[1];sys.argv=[sys.argv[0]]
import im44_h_train_eval as H
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def build():
    parts=[]; rows=[]
    for sn in ('im_set_h3','im_set_h3j','im_set_c1g'):
        own=sn=='im_set_c1g'; rr=read((HERE if own else L)/sn/'manifest.csv')
        with np.load((HERE/'feats' if own else L/'feats_h')/f'{sn}.npz') as z:
            assert z['ok'].all() and z['files'].tolist()==[r['file'] for r in rr]
            parts.append({k:z[k] for k in ('emb','low','rc')})
        rows.extend(dict(r,kind='real' if sn!='im_set_h3' else r['kind'],dataset=sn) for r in rr)
    F={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    X=np.c_[F['low'],F['emb']]; y=np.array([int(r['label']) for r in rows]); kind=np.array([r['kind'] for r in rows]); groups=np.array([r['group'] for r in rows]); source=np.array([r['source'] for r in rows])
    selected=(kind=='real')|np.isin(kind,list(H.FILT)); folds=np.full(len(y),-1)
    for i,(_,te) in enumerate(GroupKFold(5).split(X,groups=groups)):folds[te]=i
    assert all(len(set(folds[source==s]))==1 for s in set(source))
    return F,rows,X,y,kind,selected,folds
def main():
    F,rows,X,y,kind,sel,fold=build(); n=len(rows)
    if mode=='first':
        te=fold==0;tr=(fold!=0)&sel;M=H.lrfit(X[tr],y[tr],C=.5);oof=np.full(n,np.nan);oof[te]=H.lrp(M,X[te])
        np.savez(HERE/'first_fold_v1.npz',oof=oof,fold=fold,**M)
        dump(HERE/'first_fold_v1.json',dict(n=n,train=int(tr.sum()),held=int(te.sum()),held_real=int(((y==0)&te).sum()),source_group_safe=True,finite=bool(np.isfinite(oof[te]).all())))
        print('C1_FIRST_FOLD_DONE',flush=True);return
    assert mode=='rest'
    with np.load(HERE/'first_fold_v1.npz') as z:oof=z['oof'];assert np.array_equal(fold,z['fold'])
    for i in range(1,5):
        te=fold==i;tr=(fold!=i)&sel;M=H.lrfit(X[tr],y[tr],C=.5);oof[te]=H.lrp(M,X[te]);print('C1_FOLD_DONE',i,flush=True)
    assert np.isfinite(oof).all()
    M=H.lrfit(X[sel],y[sel],C=.5);np.savez(HERE/'ef_model_v1.npz',**M)
    real=y==0; pool=[oof[real]]; rc=[F['rc'][real]]
    for sn in ('im_set_newreal','im_set_realg','im_set_realn'):
        G=H.load(sn);pool.append(H.lrp(M,np.c_[G['low'],G['emb']]));rc.append(G['rc'])
    P,R=np.concatenate(pool),np.concatenate(rc);th=float(np.quantile(P,.9975));thr=float(np.quantile(P[R],.999))
    assert 0<th<1 and 0<thr<1
    dump(HERE/'ef_theta_v1.json',dict(theta=th,theta_rc=thr,real_pool=len(P),rc_pool=int(R.sum())))
    np.savez(HERE/'calibration_v1.npz',P=P,rc=R,oof=oof,fold=fold,y=y,kind=kind,source=np.array([r['source'] for r in rows]))
    old={}
    with np.load(L/'preds/h_model_h2.npz') as z:
        for k in z.files:a,c=k.split('__');old.setdefault(a,{})[c]=z[k]
    with np.load(L/'preds/ip_model_v1.npz') as z:gr={k.split('__')[1]:z[k] for k in z.files}
    T=json.loads((L/'preds/ip_theta_v1.json').read_text());b=io.BytesIO();Image.new('RGB',(64,64)).save(b,'JPEG',quality=92);q=Image.open(io.BytesIO(b.getvalue())).quantization
    def std(p):
        with Image.open(p) as im:return im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==q
    def meta(sn):
        if sn=='pub21':
            p=Path.home()/'storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플';rr=read(p/'sample_labels.csv');return rr,[p/'images'/r['file'] for r in rr]
        if sn=='jig500':
            ps=sorted((L/'preds/_jig_hold').glob('j*.jpg'));return [dict(file=p.name,label='0',kind='jig',group='unknown') for p in ps],ps
        d=L/'sealed_c3'/sn[7:] if sn.startswith('sealed_') else L/sn;rr=read(d/'manifest.csv');return rr,[d/'img'/r['file'] for r in rr]
    E={}; sets=('im_set_newreal','im_set_realg','im_set_realn','im_set_r2','im_set_r3','im_set_r4','im_set_r5','sealed_real','sealed_brown','sealed_bright','jig500','pub21','im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp','im_set_r3p_c444')
    for sn in sets:
        G=H.load(sn);rr,ps=meta(sn);assert G['files'].tolist()==[r['file'] for r in rr];gate=np.array([std(p) for p in ps]);yy=np.array([int(r['label']) for r in rr]);kd=np.array([r.get('kind','unknown') for r in rr])
        sc={**H.score(old,G),'E_gr':H.lrp(gr,np.c_[G['low'],G['emb']])};base=np.where(gate,H.apply(G['p6'],sc,T['theta'],T['theta_rc'],G['rc']),G['p6'])
        sc['E_f']=H.lrp(M,np.c_[G['low'],G['emb']]);TC={**T['theta'],'E_f':th};TR={**T['theta_rc'],'E_f':thr}
        cand=np.where(gate,H.apply(G['p6'],sc,TC,TR,G['rc']),G['p6']);assert np.array_equal(cand[~gate],base[~gate]);assert np.isfinite(cand).all()
        ref=HERE.parents[1]/'im_meta_probe_20261016_v1'/f'pred_1601_{sn}.csv'
        if sn!='im_set_r5':
            assert ref.is_file(),ref
            br=read(ref);assert [r['file'] for r in br]==[r['file'] for r in rr]
            assert np.array_equal(yy,[int(r['label']) for r in br])
            assert np.allclose(base,np.array([float(r['p9']) for r in br]),rtol=0,atol=1e-12)
        if sn=='im_set_r5':
            with np.load(HERE.parents[1]/'im_m2_20261016_v1/open/eval_im_set_r5.npz') as z:
                assert np.array_equal(yy,z['y']) and np.array_equal(kd,z['kind'])
                assert np.array_equal([r['group'] for r in rr],z['group'])
                assert np.allclose(base,z['p9'],rtol=0,atol=1e-12)
        np.savez(HERE/f'eval_{sn}.npz',y=yy,kind=kd,base=base,candidate=cand,gate=gate,rc=G['rc'])
        fp=lambda p:int(((yy==0)&(p>=.5)).sum());ba=lambda p:float(.5*((p[yy==0]<.5).mean()+(p[yy==1]>=.5).mean())) if (yy==0).any() and (yy==1).any() else None
        E[sn]=dict(n=len(yy),base_fp=fp(base),candidate_fp=fp(cand),new_fp=int(((yy==0)&(base<.5)&(cand>=.5)).sum()),base_ba=ba(base),candidate_ba=ba(cand),filter_losses=int(((yy==1)&np.char.startswith(kd,'filter_')&(base>=.5)&(cand<.5)).sum()),correct=int(((cand>=.5)==yy).sum()))
        print('C1_EVAL',sn,E[sn],flush=True)
    U=('im_set_newreal','im_set_realg','im_set_realn'); core=('im_set_r2','im_set_r3','im_set_r4','im_set_r5')
    conditions=dict(real_fp_reduction=sum(E[s]['base_fp']-E[s]['candidate_fp'] for s in U)>=3,filter_losses=sum(E[s]['filter_losses'] for s in core)<=2,
        r345_ba=all(E[s]['candidate_ba']>=E[s]['base_ba'] for s in core[1:]),public21=E['pub21']['correct']==21,
        real_guards=all(E[s]['new_fp']==0 for s in ('jig500','sealed_real','sealed_brown','sealed_bright')))
    dump(HERE/'result_v1.json',dict(metrics=E,conditions=conditions,internal_pass=all(conditions.values()),theta=th,theta_rc=thr))
    print('C1_FINAL',conditions,'PASS',all(conditions.values()),flush=True)
if __name__=='__main__':main()
