from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'run_v3.py').read_text(encoding='utf-8')
s=s.replace('def neighbors(ref,q,sig):','def neighbors_slow(ref,q,sig):').replace('preparation_v3.json','preparation_v4.json')
fast='''def neighbors(ref,q,sig):
 daily=ref.groupby(['farm','day']).sub_ec.mean(); A=sig.reindex(ref.row_id).to_numpy();Q=sig.reindex(q.row_id).to_numpy();out=np.zeros((len(q),6));Qfilled=np.empty_like(Q);sources=[]
 for h in range(24):
  rr=ref[ref.hour==h];av=A[ref.hour.to_numpy()==h];med=np.nanmedian(av,axis=0);med=np.nan_to_num(med);av=np.where(np.isnan(av),med,av);sd=av.std(0);sd[sd==0]=1
  qi=np.where(q.hour.to_numpy()==h)[0];qv=Q[qi];qv=np.where(np.isnan(qv),med,qv);Qfilled[qi]=qv
  for f in ['F13','F47']:
   for late in [False,True]:
    ai=np.where(((rr.farm==f)&((rr.day>=179)==late)).to_numpy())[0];jj=np.where(((q.iloc[qi].farm==f)&((q.iloc[qi].day>=179)==late)).to_numpy())[0]
    if not len(jj):continue
    if not len(ai):sources.extend({'row_id':q.row_id.iloc[qi[j]],'anchors':[]} for j in jj);continue
    ai=ai[np.argsort(rr.day.to_numpy()[ai],kind='stable')];days=rr.day.to_numpy()[ai];eligible=days[None,:]<q.day.to_numpy()[qi[jj],None];dist=np.sqrt(np.mean(((av[ai][None,:,:]-qv[jj][:,None,:])/sd)**2,axis=2));dist[~eligible]=np.inf;order=np.argsort(dist,axis=1,kind='stable');labels=np.array([daily[f,int(d)] for d in days])
    for i,j in enumerate(jj):
     ii=qi[j];n=int(eligible[i].sum())
     if not n:sources.append({'row_id':q.row_id.iloc[ii],'anchors':[]});continue
     z=order[i,:min(3,n)];vals=labels[z];ds=dist[i,order[i]];out[ii]=[vals.mean(),ds[0],ds[min(1,n-1)],vals.std(),np.log1p(n),1];sources.append({'row_id':q.row_id.iloc[ii],'anchors':[int(d) for d in days[z]]})
 return out,Qfilled,sources
'''
s=s.replace('def design(q,base,N,Q):',fast+'\ndef design(q,base,N,Q):')
s=s.replace("NB,QB,SB=neighbors(a,b,sig);NQ,QQ,SQ=neighbors(tr,q,sig)","NB,QB,SB=neighbors(a,b,sig);NQ,QQ,SQ=neighbors(tr,q,sig)\n  if v=='DIAG10' and k==0:\n   toy=q.iloc[:24].reset_index(drop=True);nf,qf,sf=neighbors(tr,toy,sig);ns,qs,ss=neighbors_slow(tr,toy,sig);close(nf,ns);close(qf,qs);assert sorted(sf,key=lambda z:z['row_id'])==sorted(ss,key=lambda z:z['row_id'])")
with (H/'run_v4.py').open('x',encoding='utf-8') as f:f.write(s)
v=(H/'verify_v2.py').read_text(encoding='utf-8').replace('run_v3.py','run_v4.py').replace('preparation_v3.json','preparation_v4.json')
v=v.replace("assert R.load(H/f'first_{mode}_v1.json')['status']=='PASS'", "first=R.load(H/f'first_{mode}_v1.json');assert first['status']=='PASS' and first['source']==R.sha(H/'run_v4.py') and first['repeat_error']<=1e-12;assert registration['prereg_sha']==R.sha(H/'preregistration_v1.md') and registration['verify_sha']==R.sha(Path(__file__))")
v=v.replace("assert len(fit['manifest'])==66;frames=[]", "assert len(fit['manifest'])==66 and [(z['validator'],z['fold'],z['seed']) for z in fit['manifest']]==[(v,k,s) for v,k in R.KEYS for s in R.SEEDS];frames=[]")
v=v.replace("R.close(d.y,q.sub_ec);", "assert list(d.columns)==['row_id','farm','day','hour','y','baseline','candidate','delta','confidence','anchor','d1','d2','validator','fold','seed','clip_lo','clip_hi'];assert d.row_id.is_unique;assert all(np.array_equal(d[c],q[c]) for c in ['farm','day','hour']);assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==s).all();R.close(d.clip_lo,np.full(len(q),ref['bounds'][0]));R.close(d.clip_hi,np.full(len(q),ref['bounds'][1]));R.close(d.anchor,ref['NQ'][:,0]);R.close(d.d1,ref['NQ'][:,1]);R.close(d.d2,ref['NQ'][:,2]);R.close(d.y,q.sub_ec);")
v=v.replace("scores=[];segments=[];boots={};alpha=", "lp=H/f'learning_check_{mode}_v1.json';learning=R.load(lp);assert learning['status']=='PASS_SCALER_OBJECTIVE_NOFIT' and learning['cells']==66 and learning['source']==R.sha(H/'run_v4.py') and learning['verifier']==registration['learning_sha'];scores=[];segments=[];boots={};alpha=")
with (H/'verify_v3.py').open('x',encoding='utf-8') as f:f.write(v)
l=(H/'verify_learning_v1.py').read_text(encoding='utf-8').replace('run_v3.py','run_v4.py').replace('preparation_v3.json','preparation_v4.json')
with (H/'verify_learning_v2.py').open('x',encoding='utf-8') as f:f.write(l)
print('v4 vectorized neighbors; v3 exact whole schema; v2 independent learning verifier saved')
