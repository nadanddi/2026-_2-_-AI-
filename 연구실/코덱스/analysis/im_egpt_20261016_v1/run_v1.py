"""Single frozen E_gpt hypothesis. CPU training; no submission/test input access."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,csv,json,io,hashlib,warnings
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
from PIL import Image,JpegImagePlugin
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('error',category=ConvergenceWarning)
B=Path.home()/'storage/nadan/farmai_img';L=B/'data/local';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'data/image'));sys.argv=[sys.argv[0]]
import im44_h_train_eval as H
POS=['im_set_gpt1','im_set_gpt2','im_set_gpt3','im_set_gptr'];NEG=['im_set_h3','im_set_h3j','im_set_gptr_src'];UN=['im_set_newreal','im_set_realg','im_set_realn']
b=io.BytesIO();Image.new('RGB',(64,64)).save(b,'JPEG',quality=92);Q=Image.open(io.BytesIO(b.getvalue())).quantization
cache={};hashes={}
def dump(name,obj):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def man(sn):
 if sn=='pub21':p=B.parent/'공용/대회자료/이미지데이터/참가자_배포/공개샘플';mf=p/'sample_labels.csv';d=p/'images'
 elif sn=='jig500':
  ps=sorted((L/'preds/_jig_hold').glob('j*.jpg'));return [dict(file=p.name,label='0',group='unknown',source='') for p in ps],ps
 else:d=(L/'sealed_c3'/sn[7:] if sn.startswith('sealed_') else L/sn);mf=d/'manifest.csv';d=d/'img'
 rr=list(csv.DictReader(mf.open(encoding='utf-8-sig')));return rr,[d/r['file'] for r in rr]
def load(sn):
 if sn not in cache:
  rr,ps=man(sn);p=L/'feats_h'/f'{sn}.npz';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  with np.load(p) as z:F={k:z[k] for k in z.files if k!='es'}
  assert len(rr)==len(F['files']) and [r['file'] for r in rr]==F['files'].tolist()
  gt=[]
  for p in ps:
   with Image.open(p) as im:gt.append(im.format=='JPEG' and JpegImagePlugin.get_sampling(im)==2 and im.quantization==Q)
  F['std']=np.array(gt);F['X']=np.c_[F['low'],F['emb']];assert np.isfinite(F['X'][F['ok']]).all()
  cache[sn]=(F,rr)
 return cache[sn]
def norm(r):return Path(r.get('source','').replace('\\','/')).stem
def lineage(rr):return {norm(r) for r in rr if norm(r)},{r['group'] for r in rr if r.get('group') and r['group']!='unknown'}
def disjoint(rr,held):
 ss,gg=lineage(held);return np.array([norm(r) not in ss and r.get('group') not in gg for r in rr])
def assemble(names):
 parts=[];rr=[];ds=[]
 for sn in names:
  F,r=load(sn);keep=F['ok'] & (np.array([int(x['label']) for x in r])==0 if sn in NEG else True)
  parts.append({k:F[k][keep] for k in ['X','rc','std']});rr += [dict(x,dataset=sn) for x,t in zip(r,keep) if t];ds += [sn]*int(keep.sum())
 return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]},rr,np.array(ds)
def fitcal(F,rr,held,name):
 y=np.array([int(r['label']) for r in rr]);grp=np.array([r['group'] for r in rr]);oof=np.full(len(rr),np.nan);fold=np.full(len(rr),-1)
 for j,(tr,te) in enumerate(GroupKFold(5).split(y,groups=grp)):
  assert not (set(grp[tr])&set(grp[te]));assert not (set(filter(None,[norm(rr[i]) for i in tr]))&set(filter(None,[norm(rr[i]) for i in te])))
  m=H.lrfit(F['X'][tr],y[tr],C=.5);oof[te]=H.lrp(m,F['X'][te]);fold[te]=j
 assert np.isfinite(oof).all();M=H.lrfit(F['X'],y,C=.5)
 r0=(y==0)&F['std'];pool=[oof[r0]];rc=[F['rc'][r0]]
 for sn in UN:
  U,ur=load(sn);keep=U['ok']&U['std']&disjoint(ur,held);pool.append(H.lrp(M,U['X'][keep]));rc.append(U['rc'][keep])
 P=np.concatenate(pool);R=np.concatenate(rc);assert R.any()
 th=float(np.quantile(P,.9975));trc=float(np.quantile(P[R],.999));assert 0<th<1 and 0<trc<1
 np.savez_compressed(OUT/f'{name}_oof.npz',score=oof,fold=fold,y=y,rc=F['rc'],gate=F['std'],dataset=np.array([r['dataset'] for r in rr]),files=np.array([r['file'] for r in rr]),group=grp,pool=P,pool_rc=R)
 np.savez(OUT/f'{name}_model.npz',**M);dump(f'{name}_theta.json',dict(theta=th,theta_rc=trc,pool=len(P)))
 print(name,'fit DONE',len(y),'theta',th,trc,flush=True);return M,th,trc,oof
M9={}
with np.load(L/'preds/h_model_h2.npz') as z:
 for k in z.files:a,c=k.split('__');M9.setdefault(a,{})[c]=z[k]
with np.load(L/'preds/ip_model_v1.npz') as z:M9['E_gr']={k.split('__')[1]:z[k] for k in z.files}
T=json.loads((L/'preds/ip_theta_v1.json').read_text());EXP=list(T['theta'])
def sc9(sn,F):
 with np.load(L/'feats_h'/f'{sn}.npz') as z:es=z['es']
 return {**H.score(M9,dict(low=F['low'],emb=F['emb'],es=es)),'E_gr':H.lrp(M9['E_gr'],F['X'])}
def base(sn):
 F,rr=load(sn);sc=sc9(sn,F);p=np.where(F['std'],H.apply(F['p6'],sc,T['theta'],T['theta_rc'],F['rc']),F['p6']);return p,sc
def predict(sn,M,th,trc):
 F,rr=load(sn);p,_=base(sn);s=H.lrp(M,F['X']);pnew=np.where(F['std']&F['ok'],np.maximum(p,H.g(s,np.where(F['rc'],trc,th))),p)
 assert (pnew>=p).all();return p,pnew,s
def main():
 F,rr,ds=assemble(POS+NEG);lg=[]
 for held in POS:
  _,hr=load(held);keep=(ds!=held)&disjoint(rr,hr);# keep other batches but purge overlapping original/group
  overlap=int(((ds!=held)&np.isin(ds,POS)&~disjoint(rr,hr)).sum())
  tr={k:v[keep] for k,v in F.items()};rows=[r for r,k in zip(rr,keep) if k]
  M,th,trc,_=fitcal(tr,rows,hr,'logo_'+held);p,pn,s=predict(held,M,th,trc);G,gr=load(held);y=np.array([int(r['label']) for r in gr]);fix=int(((p<.5)&(pn>=.5)&(y==1)).sum())
  rec=dict(held=held,n=len(p),base_missed=int(((p<.5)&(y==1)).sum()),fixed=fix,removed_other_positive_overlap=overlap,train=len(rows),removed_negative=int(((ds!=held)&np.isin(ds,NEG)&~keep).sum()))
  lg.append(rec);dump('logo_'+held+'_result.json',rec);np.savez(OUT/f'logo_{held}_eval.npz',p9=p,pnew=pn,score=s,y=y);print('LOGO',rec,flush=True)
  if len(lg)==1:
   dump('MIDPOINT_READY.json',rec)
   if '--unused' in []:return
 dump('logo_summary.json',lg)
 M,th,trc,oof=fitcal(F,rr,[],'final');metrics={}
 for sn in UN+['sealed_real','sealed_brown','sealed_bright','jig500','pub21','im_set_r2','im_set_r3','im_set_r4','im_set_r5']:
  p,pn,s=predict(sn,M,th,trc);G,r=load(sn);y=np.array([int(x['label']) for x in r]);re=y==0;fa=y==1
  met=dict(n=len(y),real=int(re.sum()),base_fp=int((p[re]>=.5).sum()),new_fp=int((pn[re]>=.5).sum()),fixed_fake=int(((p<.5)&(pn>=.5)&fa).sum()),correct=int(((pn>=.5)==fa).sum()))
  if re.any():met['fp_delta']=float((pn[re]>=.5).mean()-(p[re]>=.5).mean())
  if re.any() and fa.any():met.update(base_ba=float(.5*((p[re]<.5).mean()+(p[fa]>=.5).mean())),new_ba=float(.5*((pn[re]<.5).mean()+(pn[fa]>=.5).mean())))
  metrics[sn]=met;np.savez_compressed(OUT/f'eval_{sn}.npz',p9=p,pnew=pn,score=s,y=y,rc=G['rc'],gate=G['std']);print(sn,met,flush=True)
 # Reconstruct the original im66 real OOF pool in its exact row order.
 lookup={(r['dataset'],r['file']):i for i,r in enumerate(rr)};ix=[]
 for sn in ['im_set_h3','im_set_h3j','im_set_h3s','im_set_h3g','im_set_gptr','im_set_gptr_src']:
  G,r=load(sn)
  for j,row in enumerate(r):
   if row.get('split','train')=='train' and int(row['label'])==0 and G['ok'][j] and G['std'][j]:ix.append(lookup[(sn,row['file'])])
 P=[oof[ix]];R=[F['rc'][ix]];keys=[f"{rr[i]['dataset']}/{rr[i]['file']}" for i in ix]
 for sn in UN:
  G,r=load(sn);kk=G['ok']&G['std'];P.append(H.lrp(M,G['X'][kk]));R.append(G['rc'][kk]);keys += [sn+'/'+v['file'] for v,k in zip(r,kk) if k]
 P=np.concatenate(P);R=np.concatenate(R)
 cp=Path.home()/'storage/nadan/codex/im_m2_20261016_v1/preflight/calibration_v1.npz'
 with np.load(cp) as z:O9=z['O9'];oldR=z['PRC']
 assert len(P)==len(O9) and np.array_equal(R,oldR),'im66 pool mismatch'
 fire=P>=np.where(R,trc,th);union=O9|fire
 budget=dict(n=len(P),base_count=int(O9.sum()),new_count=int(union.sum()),base_rate=float(O9.mean()),new_rate=float(union.mean()),delta=float(union.mean()-O9.mean()),limit=.001)
 np.savez_compressed(OUT/'union_pool.npz',keys=np.array(keys),score=P,rc=R,O9=O9,union=union)
 conditions=dict(a=sum(v['fixed']>=1 for v in lg)>=3 and sum(v['fixed'] for v in lg)>=5,b_real=all(metrics[s]['fp_delta']<=.005+1e-12 for s in UN+['sealed_real','sealed_brown','sealed_bright','jig500']),b_union=budget['delta']<=.001+1e-12,c_pub=metrics['pub21']['correct']==21,c_ba=all(metrics[s]['new_ba']>=metrics[s]['base_ba']-.002-1e-12 for s in ['im_set_r2','im_set_r3','im_set_r4','im_set_r5']))
 now=datetime.now(timezone(timedelta(hours=9)));conditions['on_time']=now.hour<15
 dump('input_hashes.json',hashes);dump('result.json',dict(finished=now.isoformat(),logo=lg,metrics=metrics,budget=budget,conditions=conditions,passed=all(conditions.values())))
 print('DONE',conditions,'PASS',all(conditions.values()),flush=True)
if __name__=='__main__':main()
