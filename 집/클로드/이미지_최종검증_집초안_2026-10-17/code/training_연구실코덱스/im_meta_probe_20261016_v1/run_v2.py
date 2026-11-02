"""Exploratory replacement stacking; no sealed R5 access, adoption or submission."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys, csv, json, io, hashlib, warnings
from pathlib import Path
import numpy as np
from PIL import Image, JpegImagePlugin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error', category=ConvergenceWarning)
B = Path.home() / 'storage/nadan/farmai_img'
L = B / 'data/local'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(B / 'data/image'))
# Only import the pure CPU helper; im43 loads inference and is deliberately not imported.
saved_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import im44_h_train_eval as H
sys.argv = saved_argv
EXP = ('E_f', 'E_ps2', 'E_g7', 'E_gr')
SEEDS = (1601, 1602, 1603)
TRAIN = ('im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src')
EVAL = ('im_set_r2','im_set_r3','im_set_r4','sealed_real','sealed_brown','sealed_bright',
        'jig500','pub21','im_set_r3p_sharp','im_set_r3p_area','im_set_r3p_png','im_set_r3p_webp',
        'im_set_r3p_c444','im_set_gpt3','im_set_gpt1','im_set_sd2','sealed_qwen','sealed_v7',
        'im_set_gptr','im_set_gptr_src','im_set_newreal','im_set_realg','im_set_realn')
buf = io.BytesIO(); Image.new('RGB',(64,64)).save(buf,'JPEG',quality=92)
REFQ = Image.open(io.BytesIO(buf.getvalue())).quantization

def dump(name, obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def rows_paths(sn):
    if sn == 'jig500':
        paths = sorted((L/'preds/_jig_hold').glob('j*.jpg'))
        return [dict(file=p.name,label='0',kind='jig',group='unknown',source='unknown') for p in paths],paths
    if sn == 'pub21':
        p = Path.home()/'storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플'
        mf, img = p/'sample_labels.csv',p/'images'
    elif sn.startswith('sealed_'):
        p = L/'sealed_c3'/sn[7:]; mf,img=p/'manifest.csv',p/'img'
    else:
        p=L/sn; mf,img=p/'manifest.csv',p/'img'
    with mf.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    return rows,[img/r['file'] for r in rows]

def std(path):
    with Image.open(path) as im:
        return im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==REFQ

def load(sn):
    assert sn in TRAIN+EVAL, sn
    rows,paths=rows_paths(sn)
    F=H.load(sn)
    assert len(rows)==len(F['files'])
    assert all(r['file']==str(f) for r,f in zip(rows,F['files'])),sn
    F['std']=np.array([std(p) for p in paths])
    assert F['ok'].all(), (sn,'invalid cached rows')
    assert all(np.isfinite(F[k]).all() for k in ('emb','low','es','p6','ppre'))
    return F,rows

def logit(p):
    p=np.clip(p,1e-4,1-1e-4); return np.log(p/(1-p))

def score(M,F):
    return {**H.score(M,F),'E_gr':H.lrp(M['E_gr'],np.c_[F['low'],F['emb']])}

def metaX(F,sc):
    return np.column_stack([logit(F['p6']),logit(F['ppre']),F['rc'].astype(float)]+[logit(sc[k]) for k in EXP])

def models():
    M={}
    with np.load(L/'preds/h_model_h2.npz') as z:
        for k in z.files:
            a,c=k.split('__'); M.setdefault(a,{})[c]=z[k]
    with np.load(L/'preds/ip_model_v1.npz') as z: M['E_gr']={k.split('__')[1]:z[k] for k in z.files}
    T=json.loads((L/'preds/ip_theta_v1.json').read_text())
    return M,T

def build():
    parts=[]; rows=[]; src=[]
    for sn in TRAIN:
        F,rr=load(sn)
        keep=np.array([r.get('split','train')=='train' for r in rr])
        parts.append({k:F[k][keep] for k in ('emb','es','low','p6','ppre','rc','std')})
        for r,ok in zip(rr,keep):
            if ok:
                rows.append(dict(r,kind='real_jig' if sn=='im_set_h3j' else r['kind']))
                src.append(sn)
    F={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return F,rows,np.array(src)

def folds(rows,seed):
    grp=np.array([r['group'] for r in rows]); source=np.array([r['source'] for r in rows])
    gps=sorted({r['group'] for r in rows if r['kind']=='gpt_regen'})
    assert len(gps)==2, gps
    rng=np.random.default_rng(seed)
    rest=np.array(sorted(set(grp)-set(gps))); rng.shuffle(rest)
    order=list(gps)+list(rest); sizes=np.zeros(5,int); mapping={}
    for g in order:
        f=int(np.argmin(sizes)); mapping[g]=f; sizes[f]+=int((grp==g).sum())
    fold=np.array([mapping[g] for g in grp])
    assert len(set(mapping[g] for g in gps))==2
    for s in set(source): assert len(set(fold[source==s]))==1,(s,'source leakage')
    assert set(fold)==set(range(5))
    return fold

def train_exp(F,rows,src,idx,ov):
    hmask=np.isin(src,('im_set_h3','im_set_h3j'))
    rh=[dict(r,kind='real' if r['kind']=='real_jig' else r['kind']) for r in rows]
    M=H.train(F,rh,idx[hmask[idx]],ov)
    kd=np.array([r['kind'] for r in rows])
    gm=np.isin(src,('im_set_h3j','im_set_gptr','im_set_gptr_src'))|((src=='im_set_h3')&(kd=='real'))
    gi=idx[gm[idx]]; yy=(kd[gi]=='gpt_regen').astype(int)
    assert len(set(yy))==2
    M['E_gr']=H.lrfit(np.c_[F['low'][gi],F['emb'][gi]],yy,C=.5)
    return M

def metric(y,b,p):
    real=y==0; fake=y==1; bp=b>=.5; pp=p>=.5
    return dict(n=len(y),real=int(real.sum()),fake=int(fake.sum()),
        base_fp=int((bp&real).sum()),meta_fp=int((pp&real).sum()),
        base_tp=int((bp&fake).sum()),meta_tp=int((pp&fake).sum()),
        new_fp=int((~bp&pp&real).sum()),fixed_real=int((bp&~pp&real).sum()),
        new_fn=int((bp&~pp&fake).sum()),fixed_fake=int((~bp&pp&fake).sum()),
        base_correct=int((bp==fake).sum()),meta_correct=int((pp==fake).sum()),
        base_ba=float(.5*((~bp[real]).mean()+bp[fake].mean())) if real.any() and fake.any() else None,
        meta_ba=float(.5*((~pp[real]).mean()+pp[fake].mean())) if real.any() and fake.any() else None)

def main(mode):
    F,rr,src=build(); n=len(rr)
    y=np.array([int(r['label']) for r in rr]); groups={r['group'] for r in rr}; sources={r['source'] for r in rr}
    if mode=='audit':
        a=dict(n=n,gate=int(F['std'].sum()),groups=len(groups),gpt_groups=sorted({r['group'] for r in rr if r['kind']=='gpt_regen'}),
               folds={str(s):np.bincount(folds(rr,s)).tolist() for s in SEEDS},overlap={})
        for sn in EVAL:
            er,_=rows_paths(sn)
            if sn in ('im_set_gptr','im_set_gptr_src'): er=[r for r in er if r['split']=='hold']
            a['overlap'][sn]=dict(n=len(er),sources=len(sources&{r.get('source','') for r in er}),groups=len(groups&{r.get('group','') for r in er}))
        dump('audit_v1.json',a); print(json.dumps(a,ensure_ascii=False),flush=True); return
    seed=int(mode); assert seed in SEEDS
    assert not (OUT/f'metrics_{seed}.json').exists()
    fold=folds(rr,seed)
    fold_hashes={str(s):hashlib.sha256(folds(rr,s).tobytes()).hexdigest() for s in SEEDS}
    assert len(set(fold_hashes.values()))==3
    ov={i:H.ov(L/'im_set_h3/mask'/r['file'].replace('.jpg','.png')) for i,r in enumerate(rr)
        if src[i]=='im_set_h3' and r['kind'] in ('partial_style','partial_gen_v7')}
    oof={k:np.full(n,np.nan) for k in EXP}
    for f in range(5):
        tr=np.flatnonzero(fold!=f); te=np.flatnonzero(fold==f)
        assert not ({rr[i]['source'] for i in tr}&{rr[i]['source'] for i in te})
        assert not ({rr[i]['group'] for i in tr}&{rr[i]['group'] for i in te})
        M=train_exp(F,rr,src,tr,ov)
        ss=score(M,{k:F[k][te] for k in ('low','emb','es')})
        for k in EXP: oof[k][te]=ss[k]
        print(f'seed {seed} fold {f} train {len(tr)} held {len(te)} DONE',flush=True)
    assert all(np.isfinite(s).all() for s in oof.values())
    X=metaX(F,oof); gate=F['std']; MM=H.lrfit(X[gate],y[gate],C=1)
    np.savez(OUT/f'meta_{seed}.npz',**MM)
    np.savez_compressed(OUT/f'oof_{seed}.npz',X=X,y=y,fold=fold,gate=gate,source=np.array([r['source'] for r in rr]),group=np.array([r['group'] for r in rr]),dataset=src)
    M9,T=models(); results={}
    dep=score(M9,F)
    real=(y==0)&gate
    dump(f'distribution_{seed}.json',dict(fold_hashes=fold_hashes,
        quantiles={k:{str(q):dict(oof=float(np.quantile(logit(oof[k][real]),q)),
                   deployment_in_sample=float(np.quantile(logit(dep[k][real]),q)))
                   for q in (.5,.99,.9975)} for k in EXP}))
    for sn in EVAL:
        G,er=load(sn); sc=score(M9,G)
        b=np.where(G['std'],H.apply(G['p6'],sc,T['theta'],T['theta_rc'],G['rc']),G['p6'])
        p=np.where(G['std'],H.lrp(MM,metaX(G,sc)),G['p6'])
        assert np.array_equal(p[~G['std']],b[~G['std']])
        keep=np.array([r.get('split','hold')=='hold' for r in er]) if sn in ('im_set_gptr','im_set_gptr_src') else np.ones(len(er),bool)
        yy=np.array([int(r['label']) for r in er]); result=metric(yy[keep],b[keep],p[keep]); results[sn]=result
        same_source=np.array([r.get('source','') in sources for r in er])
        same_group=np.array([r.get('group','') in groups for r in er])
        result['source_overlap_rows']=int((keep&same_source).sum())
        result['group_overlap_rows']=int((keep&same_group).sum())
        result['source_disjoint']=metric(yy[keep&~same_source],b[keep&~same_source],p[keep&~same_source])
        result['group_disjoint']=metric(yy[keep&~same_group],b[keep&~same_group],p[keep&~same_group])
        with (OUT/f'pred_{seed}_{sn}.csv').open('x',newline='',encoding='utf-8') as fh:
            w=csv.writer(fh); w.writerow(['file','source','group','kind','label','gate','p9','meta','source_overlap','group_overlap'])
            for i in np.flatnonzero(keep): w.writerow([er[i]['file'],er[i].get('source',''),er[i].get('group',''),er[i].get('kind',''),yy[i],int(G['std'][i]),b[i],p[i],int(same_source[i]),int(same_group[i])])
        print(seed,sn,json.dumps(result),flush=True)
    dump(f'metrics_{seed}.json',results)
    print('SEED_DONE',seed,flush=True)

if __name__=='__main__': main(sys.argv[1])
