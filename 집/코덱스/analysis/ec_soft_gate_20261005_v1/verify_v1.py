"""Whole model replay, scalar scaler/gradient and independent loss calculations."""
from pathlib import Path
import sys,math,json,csv,hashlib,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('soft_runner',H/'run_v1.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
np,pd=M.np,M.pd
def rmse(y,p):
    a=math.sqrt(math.fsum((float(t)-float(z))**2 for t,z in zip(y,p))/len(y));b=float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)));M.close([a],[b]);return a
def main():
    M.frozen();refs,receipt=M.prep();assert receipt==M.load(H/'preparation_v1.json');fit=M.load(H/'fit_receipt_v1.json');assert fit['status']=='COMPLETE66_SCORE0' and fit['registration_sha']==M.sha(H/'registration_v1.json');assert len(fit['manifest'])==66
    assert M.load(H/'first_audit_v1.json')['status']=='PASS';expected=[(v,k,s) for v,k in M.R.KEYS for s in M.R.SEEDS];assert [(r['validator'],r['fold'],r['seed']) for r in fit['manifest']]==expected
    frames=[];checks=[];maxerr=0.;maxgrad=0.
    for r in fit['manifest']:
        v,k,s=r['validator'],r['fold'],r['seed'];ref=refs[v,k];path=M.OUT/f'{v}_{k}_{s}.csv';fp=M.OUT/f'{v}_{k}_{s}_fit.json';assert M.sha(path)==r['csv_sha'] and M.sha(fp)==r['fit_sha'];d=pd.read_csv(path,float_precision='round_trip');m=M.load(fp)
        q,a,b,x,ok=M.pair(ref,s,False);assert np.array_equal(d.row_id,q.row_id) and d.row_id.is_unique;assert all(np.array_equal(d[c],q[c]) for c in ['farm','day','hour']);assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==s).all();assert np.array_equal(d.eligible,ok)
        M.close(d.y,q.sub_ec);M.close(d.A,a);M.close(d.B,b);g=M.replay(x,ok,m);M.close(g,d.g);maxerr=max(maxerr,M.close(a+g*(b-a),d.candidate));assert np.array_equal(d.candidate.to_numpy()[~ok],a[~ok]);assert np.all(d.candidate>=np.minimum(a,b)-1e-15) and np.all(d.candidate<=np.maximum(a,b)+1e-15)
        ib,ia,isp,ix,iok=M.pair(ref,s,True)
        if not m['empty']:
            xx=ix[iok];mean=np.array([math.fsum(map(float,col))/len(col) for col in xx.T]);sd=np.array([math.sqrt(math.fsum((float(t)-float(mu))**2 for t in col)/len(col)) for mu,col in zip(mean,xx.T)]);sd[sd==0]=1
            assert np.max(abs(mean-m['mean']))<1e-8 and np.max(abs(sd-m['scale']))<1e-8;assert M.R.ids(ib.row_id[iok])==m['ids']
            z=np.column_stack([np.ones(iok.sum()),(xx-np.asarray(m['mean']))/m['scale']]);t=np.asarray(m['theta']);t0=np.asarray(m['prior']);y=ib.sub_ec.to_numpy()[iok];delta=isp[iok]-ia[iok]
            # math.exp/fsum formula independently implements the trained objective.
            logits=[math.fsum(float(zz)*float(tt) for zz,tt in zip(row,t)) for row in z]
            probs=np.array([1/(1+math.exp(-val)) if val>=0 else math.exp(val)/(1+math.exp(val)) for val in logits]);res=ia[iok]+probs*delta-y
            f=math.fsum(float(rr)**2 for rr in res)/len(y)+M.CFG['penalty']*math.fsum((float(tt)-float(pr))**2 for tt,pr in zip(t,t0));M.close([f],[m['objective']])
            grad=np.array([2*math.fsum(float(zz)*float(rr)*float(dd)*float(gg)*(1-float(gg)) for zz,rr,dd,gg in zip(col,res,delta,probs))/len(y)+2*M.CFG['penalty']*(float(tt)-float(pr)) for col,tt,pr in zip(z.T,t,t0)]);assert np.max(abs(grad))<=1e-6;maxgrad=max(maxgrad,float(np.max(abs(grad))))
            M.close([np.mean(res*res)],[m['train_mse']]);M.close([np.mean((ia[iok]-y)**2)],[m['train_a_mse']]);assert m['objective']<=M.objective(t0,z,ia[iok],delta,y,t0)[0]+1e-12
        checks.append(dict(validator=v,fold=k,seed=s,n=len(d),train_n=m['n'],empty=m['empty'],outer_a_mse=float(np.mean((d.A-d.y)**2)),outer_mse=float(np.mean((d.candidate-d.y)**2)),train_a_mse=m.get('train_a_mse'),train_mse=m.get('train_mse')));frames.append(d)
    agg=pd.concat(frames,ignore_index=True);saved=pd.read_csv(M.OUT/'oof.csv',float_precision='round_trip');assert agg.equals(saved) and len(agg)==fit['rows']==83160;assert not agg.duplicated(['validator','fold','seed','row_id']).any();assert M.sha(M.OUT/'oof.csv')==fit['aggregate_sha']
    scores=[];segments=[];boots={};manual=[]
    for v in ['DIAG10','A','B','EXT10','EXT12']:
        for s in M.R.SEEDS:
            d=agg[(agg.validator==v)&(agg.seed==s)];a=rmse(d.y,d.A);p=rmse(d.y,d.candidate);b=rmse(d.y,d.B)
            scores.append(dict(validator=v,seed=s,n=len(d),baseline=a,candidate=p,specialist=b,change_pct=100*(p/a-1),fold_rmse_sd=float(np.std([rmse(f.y,f.candidate) for _,f in d.groupby('fold')],ddof=0)),gate_mean=float(d.g.mean()),changed_rows=int((d.candidate!=d.A).sum())))
    for s in M.R.SEEDS:
        d=agg[(agg.validator=='DIAG10')&(agg.seed==s)].copy();d['ld']=(d.candidate-d.y)**2-(d.A-d.y)**2;daily=d.groupby(['farm','day']).ld.agg(['sum','count']);rng=np.random.default_rng(20261003+s);tot=np.zeros(20000);nn=np.zeros(20000);manualtot=0.;manualn=0
        for f in ['F13','F47']:
            z=daily.loc[f].sort_index();blocks=[z.iloc[i:i+5] for i in range(0,len(z),5)];ss=np.array([b['sum'].sum() for b in blocks]);ns=np.array([b['count'].sum() for b in blocks]);ix=rng.integers(len(ss),size=(20000,len(ss)));tot+=ss[ix].sum(1);nn+=ns[ix].sum(1)
            # Check first bootstrap draw against raw rows using a dictionary and fsum.
            direct={}
            for row in d[d.farm==f].itertuples():direct.setdefault(int(row.day),[]).append((float(row.candidate)-float(row.y))**2-(float(row.A)-float(row.y))**2)
            days=sorted(direct);bl=[days[i:i+5] for i in range(0,len(days),5)]
            for j in ix[0]:
                for day in bl[int(j)]:manualtot+=math.fsum(direct[day]);manualn+=len(direct[day])
        sample=tot/nn;M.close([sample[0]],[manualtot/manualn]);manual.append(dict(seed=s,first_vector=float(sample[0]),first_manual=manualtot/manualn));alpha=M.CFG['alpha'];boots[str(s)]=dict(p_worse=float(np.mean(sample>=0)),ci_adjusted=np.quantile(sample,[alpha,1-alpha]).tolist(),ci95=np.quantile(sample,[.025,.975]).tolist())
        high=d.groupby(['farm','day']).y.transform('mean')>=1
        for name,mask in [('high',high),('ordinary',~high),('pass1',d.day<179),('pass2',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47')]:
            z=d[mask];a=rmse(z.y,z.A);p=rmse(z.y,z.candidate);segments.append(dict(seed=s,segment=name,n=len(z),days=len(z[['farm','day']].drop_duplicates()),baseline=a,candidate=p,change_pct=100*(p/a-1)))
    judged=[r for r in scores if r['validator'] in M.CFG['judged']];direction=sum(r['candidate']<r['baseline'] for r in judged);stat=all(b['p_worse']<M.CFG['alpha'] and b['ci_adjusted'][1]<0 for b in boots.values());guard=any(r['change_pct']>=2 for r in segments if r['segment']=='pass2');passed=direction==9 and stat and not guard
    decision='HOLD_EL1_GUARD_UNASSESSED' if passed else 'REJECT'
    summary=dict(status='PASS_WHOLE_INDEPENDENT_ARITHMETIC',decision=decision,cells=66,rows=len(agg),direction_pass=direction,direction_total=9,alpha=M.CFG['alpha'],statistical_pass=stat,pass2_guard_triggered=guard,EL1_scored=False,replay_maxerr=maxerr,gradient_max=maxgrad,scores=scores,segments=segments,bootstrap=boots,manual_bootstrap=manual,learning_checks=checks,aggregate_sha=fit['aggregate_sha'],verifier_sha=M.sha(Path(__file__)),limitations=['existing one heldout b subset, not full inner crossfit','repeated public validation, not untouched holdout','base cache provenance verified; A not refit'])
    M.save(H/'verification_v1.json',summary);pd.DataFrame(scores).to_csv(H/'scores_v1.csv',index=False);pd.DataFrame(segments).to_csv(H/'segments_v1.csv',index=False);print(pd.DataFrame(scores).to_string(index=False));print('DECISION',decision,direction,'/9',boots,flush=True)
if __name__=='__main__':main()
