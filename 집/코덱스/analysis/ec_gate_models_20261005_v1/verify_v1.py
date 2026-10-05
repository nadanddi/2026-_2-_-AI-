"""Verify 198 saved gates and all scores independently before reporting."""
from pathlib import Path
import sys,math,json,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('models_runner',H/'run_v2.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
np,pd=M.np,M.pd
def rmse(y,p):
    a=math.sqrt(math.fsum((float(t)-float(z))**2 for t,z in zip(y,p))/len(y));b=float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)));M.close([a],[b]);return a
def sigmoid(v):return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
def scalar_forward(mode,row,m):
    if m['constant'] is not None:return m['constant']
    z=[(float(x)-float(mu))/float(sd) for x,mu,sd in zip(row,m['mean'],m['scale'])]
    if mode=='LR':return sigmoid(math.fsum(x*t for x,t in zip(z,m['coef'][0]))+m['intercept'])
    if mode=='LGB':return sigmoid(math.fsum(M.tree_value(t['tree_structure'],z) for t in m['tree']['tree_info']))
    hidden=[math.tanh(math.fsum(z[i]*m['coefs'][0][i][j] for i in range(len(z)))+m['intercepts'][0][j]) for j in range(8)]
    return sigmoid(math.fsum(hidden[j]*m['coefs'][1][j][0] for j in range(8))+m['intercepts'][1][0])
def learning(mode,ref,s,m):
    q,a,b,x,ok=M.B.pair(ref,s,True);y=q.sub_ec.to_numpy();cost=(a-y)**2-(b-y)**2;eligible=ok&(abs(cost)>M.CFG['tie_epsilon']);assert M.B.R.ids(q.row_id[eligible])==m['eligible_ids'] and M.B.R.ar(cost)==m['cost_sha'];assert eligible.sum()==m['n']
    if not m['n']:assert m['constant']==0;return 0.
    t=(cost[eligible]>0).astype(int);w=abs(cost[eligible]);w=w/w.mean();assert M.B.R.ar(t)==m['label_sha'] and M.B.R.ar(w)==m['weight_sha'];xx=x[eligible]
    means=np.array([math.fsum(map(float,col))/len(col) for col in xx.T]);sd=np.array([math.sqrt(math.fsum((float(v)-float(mu))**2 for v in col)/len(col)) for mu,col in zip(means,xx.T)]);sd[sd==0]=1
    assert np.max(abs(means-m['mean']))<1e-8 and np.max(abs(sd-m['scale']))<1e-8
    z=(xx-np.asarray(m['mean']))/m['scale'];g=M.forward(mode,xx,m);M.close([np.mean((a[eligible]+g*(b[eligible]-a[eligible])-y[eligible])**2)],[m['train_mse']]);M.close([np.mean((a[eligible]-y[eligible])**2)],[m['train_a_mse']])
    probs=np.clip(g,1e-15,1-1e-15);ll=math.fsum(float(ww)*(-float(tt)*math.log(float(gg))-(1-float(tt))*math.log(1-float(gg))) for ww,tt,gg in zip(w,t,probs))/math.fsum(map(float,w));assert abs(ll-m['train_logloss'])<1e-10
    if m['constant'] is not None:return 0.
    error=(g-t)*w/w.sum()
    if mode=='LR':
        coef=np.asarray(m['coef']).reshape(-1);grad=np.r_[z.T@error+coef/(M.CFG['lr']['C']*w.sum()),error.sum()];norm=float(np.max(abs(grad)));assert norm<1e-5;return norm
    if mode=='MLP':
        w0,w1=map(np.asarray,m['coefs']);hidden=np.tanh(z@w0+m['intercepts'][0]);dh=error[:,None]*w1.T*(1-hidden**2);alpha=M.CFG['mlp']['alpha'];grads=[z.T@dh+alpha*w0/w.sum(),hidden.T@error[:,None]+alpha*w1/w.sum(),dh.sum(0),np.array([error.sum()])];norm=max(float(np.max(abs(gv))) for gv in grads);loss=ll+alpha*(np.sum(w0*w0)+np.sum(w1*w1))/(2*w.sum());assert abs(loss-m['loss'])<1e-10;assert norm<5e-4;return norm
    return 0.
def evaluate(mode,refs):
    fit=M.load(H/f'fit_{mode}_v1.json');assert fit['status']=='COMPLETE66_SCORE0' and fit['registration_sha']==M.sha(H/'registration_v1.json');assert M.load(H/f'first_{mode}_v1.json')['status']=='PASS';expected=[(v,k,s) for v,k in M.KEYS for s in M.SEEDS];assert [(r['validator'],r['fold'],r['seed']) for r in fit['manifest']]==expected
    dest=M.OUT/mode;frames=[];checks=[];maxerr=0.;maxgrad=0.
    for r in fit['manifest']:
        v,k,s=r['validator'],r['fold'],r['seed'];ref=refs[v,k];path=dest/f'{v}_{k}_{s}.csv';fp=dest/f'{v}_{k}_{s}_fit.json';assert M.sha(path)==r['csv_sha'] and M.sha(fp)==r['fit_sha'];d=pd.read_csv(path,float_precision='round_trip');m=M.load(fp);q,a,b,x,ok=M.B.pair(ref,s,False)
        assert np.array_equal(d.row_id,q.row_id) and d.row_id.is_unique;assert all(np.array_equal(d[c],q[c]) for c in ['farm','day','hour']);assert (d.validator==v).all() and (d.fold==k).all() and (d.seed==s).all();assert np.array_equal(d.eligible,ok);M.close(d.y,q.sub_ec);M.close(d.A,a);M.close(d.B,b)
        g=np.where(ok,M.forward(mode,x,m),0);M.close(g,d.g);maxerr=max(maxerr,M.close(a+g*(b-a),d.candidate));assert np.array_equal(d.candidate.to_numpy()[~ok],a[~ok]);assert np.all(d.candidate>=np.minimum(a,b)-1e-15) and np.all(d.candidate<=np.maximum(a,b)+1e-15)
        chosen=sorted(set([0,len(x)//2,len(x)-1]));manual=[scalar_forward(mode,x[i],m) if ok[i] else 0 for i in chosen];M.close(g[chosen],manual);maxgrad=max(maxgrad,learning(mode,ref,s,m));checks.append(dict(validator=v,fold=k,seed=s,train_n=m['n'],train_mse=m.get('train_mse'),train_a_mse=m.get('train_a_mse'),outer_mse=float(np.mean((d.candidate-d.y)**2)),outer_a_mse=float(np.mean((d.A-d.y)**2))));frames.append(d)
    agg=pd.concat(frames,ignore_index=True);saved=pd.read_csv(dest/'oof.csv',float_precision='round_trip');assert agg.equals(saved) and len(agg)==fit['rows']==83160;assert not agg.duplicated(['validator','fold','seed','row_id']).any();assert M.sha(dest/'oof.csv')==fit['aggregate_sha'];scores=[];segments=[];boots={};manualboots=[]
    for v in ['DIAG10','A','B','EXT10','EXT12']:
        for s in M.SEEDS:
            d=agg[(agg.validator==v)&(agg.seed==s)];a=rmse(d.y,d.A);p=rmse(d.y,d.candidate);scores.append(dict(mode=mode,validator=v,seed=s,n=len(d),baseline=a,candidate=p,change_pct=100*(p/a-1),fold_rmse_sd=float(np.std([rmse(f.y,f.candidate) for _,f in d.groupby('fold')])),gate_mean=float(d.g.mean())))
    for s in M.SEEDS:
        d=agg[(agg.validator=='DIAG10')&(agg.seed==s)].copy();d['ld']=(d.candidate-d.y)**2-(d.A-d.y)**2;daily=d.groupby(['farm','day']).ld.agg(['sum','count']);rng=np.random.default_rng(20261003+s);tot=np.zeros(20000);nn=np.zeros(20000);manualtot=0.;manualn=0
        for f in ['F13','F47']:
            z=daily.loc[f].sort_index();blocks=[z.iloc[i:i+5] for i in range(0,len(z),5)];ss=np.array([b['sum'].sum() for b in blocks]);ns=np.array([b['count'].sum() for b in blocks]);ix=rng.integers(len(ss),size=(20000,len(ss)));tot+=ss[ix].sum(1);nn+=ns[ix].sum(1);direct={}
            for row in d[d.farm==f].itertuples():direct.setdefault(int(row.day),[]).append((float(row.candidate)-float(row.y))**2-(float(row.A)-float(row.y))**2)
            days=sorted(direct);bl=[days[i:i+5] for i in range(0,len(days),5)]
            for j in ix[0]:
                for day in bl[int(j)]:manualtot+=math.fsum(direct[day]);manualn+=len(direct[day])
        sample=tot/nn;M.close([sample[0]],[manualtot/manualn]);manualboots.append(dict(seed=s,first=float(sample[0]),independent=manualtot/manualn));alpha=M.CFG['alpha'];boots[str(s)]=dict(p_worse=float(np.mean(sample>=0)),ci_adjusted=np.quantile(sample,[alpha,1-alpha]).tolist());high=d.groupby(['farm','day']).y.transform('mean')>=1
        for name,mask in [('high',high),('ordinary',~high),('pass1',d.day<179),('pass2',d.day>=179),('F13',d.farm=='F13'),('F47',d.farm=='F47')]:
            z=d[mask];a=rmse(z.y,z.A);p=rmse(z.y,z.candidate);segments.append(dict(mode=mode,seed=s,segment=name,n=len(z),days=len(z[['farm','day']].drop_duplicates()),baseline=a,candidate=p,change_pct=100*(p/a-1)))
    judged=[r for r in scores if r['validator'] in M.CFG['judged']];direction=sum(r['candidate']<r['baseline'] for r in judged);stat=all(b['p_worse']<M.CFG['alpha'] and b['ci_adjusted'][1]<0 for b in boots.values());guard=any(r['change_pct']>=2 for r in segments if r['segment']=='pass2');decision='HOLD_EL1_UNASSESSED' if direction==9 and stat and not guard else 'REJECT'
    summary=dict(status='PASS_WHOLE_INDEPENDENT_ARITHMETIC',mode=mode,decision=decision,cells=66,rows=len(agg),direction_pass=direction,direction_total=9,alpha=M.CFG['alpha'],pass2_guard_triggered=guard,replay_maxerr=maxerr,gradient_max=maxgrad,scores=scores,segments=segments,bootstrap=boots,manual_bootstrap=manualboots,learning_checks=checks,aggregate_sha=fit['aggregate_sha'],verifier_sha=M.sha(Path(__file__)),EL1_scored=False)
    M.save(H/f'verification_{mode}_v1.json',summary);print(pd.DataFrame(scores).to_string(index=False));print(mode,'DECISION',decision,direction,'/9',boots,flush=True);return summary
def main():
    M.frozen();refs,p=M.prep();assert p==M.load(H/'preparation_v1.json');results=[evaluate(mode,refs) for mode in M.MODES];pd.DataFrame([s for r in results for s in r['scores']]).to_csv(H/'scores_v1.csv',index=False);pd.DataFrame([s for r in results for s in r['segments']]).to_csv(H/'segments_v1.csv',index=False);M.save(H/'verification_all_v1.json',dict(status='PASS_ALL198',cells=198,rows=249480,results=[dict(mode=r['mode'],decision=r['decision'],direction_pass=r['direction_pass'],verification_sha=M.sha(H/f"verification_{r['mode']}_v1.json")) for r in results]));print('ALL198_VERIFIED',flush=True)
if __name__=='__main__':main()
