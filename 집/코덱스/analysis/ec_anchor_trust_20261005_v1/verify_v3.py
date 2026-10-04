"""Whole saved output replay without model fit. Run only after ALL66."""
from pathlib import Path
import sys,json,csv,math,hashlib,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('gate_runner',H/'run_v4.py');R=importlib.util.module_from_spec(sp);sp.loader.exec_module(R)
np,pd=R.np,R.pd
def rmse(y,p):
 a=math.sqrt(math.fsum((float(t)-float(z))**2 for t,z in zip(y,p))/len(y));b=float(np.sqrt(np.mean((np.asarray(y)-p)**2)));R.close([a],[b]);return a
def main(mode):
 refs,prep=R.preparation();assert R.load(H/'preparation_v4.json')==prep;fit=R.load(H/f'fit_{mode}_v1.json');assert fit['status']=='PASS_FIT66_SCORE0' and fit['source']==R.sha(H/'run_v4.py') and fit['preparation']==R.sha(H/'preparation_v4.json')
 registration=R.load(H/'registration_v1.json');assert registration['run_sha']==R.sha(H/'run_v4.py') and registration['prep_sha']==R.sha(H/'preparation_v4.json');first=R.load(H/f'first_{mode}_v1.json');assert first['status']=='PASS' and first['source']==R.sha(H/'run_v4.py') and first['repeat_error']<=1e-12;assert registration['prereg_sha']==R.sha(H/'preregistration_v1.md') and registration['verify_sha']==R.sha(Path(__file__))
 dest=R.OUT/mode;assert len(fit['manifest'])==66 and [(z['validator'],z['fold'],z['seed']) for z in fit['manifest']]==[(v,k,s) for v,k in R.KEYS for s in R.SEEDS];frames=[];maxerr=0
 for rec in fit['manifest']:
  v,k,s=rec['validator'],rec['fold'],rec['seed'];ref=refs[v,k];q=ref['q'];p=dest/f'{v}_{k}_{s}.csv';assert R.sha(p)==rec['csv_sha'];fp=dest/f'{v}_{k}_{s}_fit.json';assert R.sha(fp)==rec['fit_sha'];m=R.load(fp);d=pd.read_csv(p,float_precision='round_trip');assert np.array_equal(d.row_id,q.row_id)
  assert list(d.columns)==['row_id','farm','day','hour','y','baseline','candidate','delta','confidence','anchor','d1','d2','validator','fold','seed','clip_lo','clip_hi'];assert d.row_id.is_unique;assert all(np.array_equal(d[c],q[c]) for c in ['farm','day','hour']);assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==s).all();R.close(d.clip_lo,np.full(len(q),ref['bounds'][0]));R.close(d.clip_hi,np.full(len(q),ref['bounds'][1]));R.close(d.anchor,ref['NQ'][:,0]);R.close(d.d1,ref['NQ'][:,1]);R.close(d.d2,ref['NQ'][:,2]);R.close(d.y,q.sub_ec);R.close(d.baseline,ref['baseline'][s]);x,ok,dq=R.design(q,ref['baseline'][s],ref['NQ'],ref['QQ']);xx=(x-np.asarray(m['mean']))/m['scale']
  if mode=='GATE':
   if m['model_none']:z=np.full(len(q),float(m['positive']==m['n']))
   else:
    lin=xx@np.asarray(m['coef']).reshape(-1)+float(np.asarray(m['intercept']).reshape(-1)[0]);z=np.empty_like(lin);pos=lin>=0;z[pos]=1/(1+np.exp(-lin[pos]));e=np.exp(lin[~pos]);z[~pos]=e/(1+e)
   delta=np.where(ok&(z>=.5),dq,0.)
  else:z=xx@np.asarray(m['coef'])+float(m['intercept']);delta=np.where(ok,.2*np.clip(z,-.3,.3),0.)
  maxerr=max(maxerr,R.close(z,d.confidence),R.close(delta,d.delta),R.close(np.clip(ref['baseline'][s]+delta,*ref['bounds']),d.candidate));assert np.array_equal(d.candidate.to_numpy()[~ok],d.baseline.to_numpy()[~ok]);frames.append(d)
 allrows=pd.concat(frames,ignore_index=True);agg=pd.read_csv(dest/'oof.csv',float_precision='round_trip');assert len(allrows)==len(agg)==83160 and not agg.duplicated(['validator','fold','seed','row_id']).any();assert R.sha(dest/'oof.csv')==fit['aggregate'];assert allrows.equals(agg)
 lp=H/f'learning_check_{mode}_v1.json';learning=R.load(lp);assert learning['status']=='PASS_SCALER_OBJECTIVE_NOFIT' and learning['cells']==66 and learning['source']==R.sha(H/'run_v4.py') and learning['verifier']==registration['learning_sha'];scores=[];segments=[];boots={};alpha=R.CFG['alpha']
 for v in R.CFG['validators']:
  for s in R.SEEDS:
   d=agg[(agg.validator==v)&(agg.seed==s)];a=rmse(d.y,d.baseline);b=rmse(d.y,d.candidate);scores.append(dict(validator=v,seed=s,n=len(d),baseline=a,candidate=b,change_pct=100*(b/a-1)))
 for s in R.SEEDS:
  d=agg[(agg.validator=='DIAG10')&(agg.seed==s)].copy();d['ld']=(d.candidate-d.y)**2-(d.baseline-d.y)**2;daily=d.groupby(['farm','day']).ld.agg(['sum','count']);rng=np.random.default_rng(20261003+s);tot=np.zeros(20000);nn=np.zeros(20000)
  for f in ['F13','F47']:
   g=daily.loc[f].sort_index();blocks=[g.iloc[i:i+5] for i in range(0,len(g),5)];aa=np.array([z['sum'].sum() for z in blocks]);n=np.array([z['count'].sum() for z in blocks]);ix=rng.integers(len(aa),size=(20000,len(aa)));tot+=aa[ix].sum(1);nn+=n[ix].sum(1)
  sample=tot/nn;boots[str(s)]=dict(p_worse=float(np.mean(sample>=0)),ci_adjusted=np.quantile(sample,[alpha,1-alpha]).tolist(),ci95=np.quantile(sample,[.025,.975]).tolist())
  high=d.groupby(['farm','day']).y.transform('mean')>=1
  for segment,mask in [('high',high),('ordinary',~high),('pass2',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47')]:
   z=d[mask];a=rmse(z.y,z.baseline);b=rmse(z.y,z.candidate);segments.append(dict(seed=s,segment=segment,days=len(z[['farm','day']].drop_duplicates()),baseline=a,candidate=b,change_pct=100*(b/a-1),changed_rows=int((abs(z.delta)>0).sum())))
 passed=all(x['candidate']<x['baseline'] for x in scores) and all(x['p_worse']<alpha and x['ci_adjusted'][1]<0 for x in boots.values()) and all(x['change_pct']<=2 for x in segments if x['segment']=='ordinary')
 R.save(H/f'verification_{mode}_v1.json',dict(status='PASS_ARITHMETIC_REPLAY',decision='PUBLIC_PASS_PENDING_REVIEW' if passed else 'REJECT',mode=mode,rows=len(agg),cells=66,replay_maxerr=maxerr,scores=scores,segments=segments,bootstrap=boots,alpha=alpha,fit_count=0,verifier=R.sha(Path(__file__)),aggregate=fit['aggregate']))
 print(pd.DataFrame(scores).to_string(index=False));print('DECISION',passed,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['GATE','RIDGE'],required=True);main(ap.parse_args().mode)
