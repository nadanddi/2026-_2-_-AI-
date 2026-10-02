from support import *
import math,time
def main():
    while not (HERE/'extra_result.json').exists() or not (HERE/'verification.json').exists():time.sleep(3)
    result=json.loads((HERE/'extra_result.json').read_text(encoding='utf-8'));d=pd.read_csv(OUT/'extra_oof.csv');assert sha(OUT/'extra_oof.csv')==result['oof_hash'];original=pd.read_csv(OUT/'oof.csv');labels=original.drop_duplicates(['target','row_id']).set_index(['target','row_id']).y.to_dict();checked=[]
    for s in result['scores']:
        q=d[(d.arm==s['arm'])&(d.validator==s['validator'])&(d.seed==s['seed'])&(d.context==s['context'])];assert all(abs(float(r.y)-labels[(r.target,r.row_id)])<1e-12 for r in q.itertuples());a=math.sqrt(math.fsum((float(r.baseline)-float(r.y))**2 for r in q.itertuples())/len(q));b=math.sqrt(math.fsum((float(r.candidate)-float(r.y))**2 for r in q.itertuples())/len(q));assert abs(a-s['baseline_rmse'])<1e-12 and abs(b-s['candidate_rmse'])<1e-12
        if s['validator']=='DIAG10':
            groups={}
            for r in q.itertuples():groups.setdefault((r.farm,int(r.day)//5),[]).append((float(r.candidate)-float(r.y))**2-(float(r.baseline)-float(r.y))**2)
            keys=sorted(groups);counts=[len(groups[k]) for k in keys];sums=[math.fsum(groups[k]) for k in keys];rng=np.random.default_rng(20261003);ix=[]
            for farm in ['F13','F47']:
                choices=np.array([i for i,k in enumerate(keys) if k[0]==farm]);ix.append(choices[rng.integers(0,len(choices),(20000,len(choices)))])
            ix=np.concatenate(ix,axis=1);samples=np.array([math.fsum(sums[i] for i in row)/sum(counts[i] for i in row) for row in ix]);p=float(np.mean(samples>=0));ci=np.quantile(samples,[ALPHA,1-ALPHA]);assert abs(p-s['p_worse'])<1e-10 and np.max(np.abs(ci-s['ci_mse']))<1e-11
        checked.append(dict(arm=s['arm'],validator=s['validator'],seed=s['seed'],context=s['context'],delta_pct=100*(b/a-1)))
    coef_checks=0;max_gap=0.
    for path in OUT.glob('*_extra.npz'):
        z=dict(np.load(path));x=z['cal_design'];w=z.get('cal_w',np.ones(len(x)));theta=np.linalg.solve(x.T@(w[:,None]*x)+100*np.eye(x.shape[1]),x.T@(w*z['cal_target']));gap=float(np.max(np.abs(theta-z['coef'])));assert gap<1e-9;max_gap=max(max_gap,gap);coef_checks+=1
        target,name,fold,*_=path.name.split('_');seed=int(path.name.split('_')[-3] if target=='T' else path.name.split('_')[-2]);context=path.name.split('_')[-2] if target=='T' else '1-4';arm='TBIAS_PH' if target=='T' else 'ESTATE_PAR';q=d[(d.arm==arm)&(d.validator==name)&(d.fold==int(fold))&(d.seed==seed)&(d.context==context)].set_index('row_id').reindex(z['outer_id'])
        if target=='T':pred=q.baseline.to_numpy()+z['outer_g']*np.clip(z['outer_design']@theta,-.35,.35)
        else:
            pred=np.clip(q.baseline.to_numpy()+np.clip(z['outer_design']@theta,-.15,.15),z['clip_lo'],z['clip_hi']);cpu=dict(np.load(OUT/f'E_{name}_{fold}_cpu.npz'));keys={(r[:3],int(r[4:7])) for r in cpu['outer_train_id']}
            for farm,day,source in zip(q.farm,q.day,z['outer_source']):
                allowed=[dd for f,dd in keys if f==farm and dd<int(day) and dd%2==int(day)%2 and (dd>=179)==(int(day)>=179)];expected=max(allowed) if allowed else -1;assert source==expected
        assert np.max(np.abs(pred-q.candidate.to_numpy()))<1e-11
    assert len(checked)==27 and coef_checks==114,(len(checked),coef_checks);savej(HERE/'extra_verification.json',dict(status='PASS',score_cells=checked,coef_checks=coef_checks,max_coef_gap=max_gap,methods=['raw labels/math.fsum','normal equations','independent causal parity source IDs','scalar block bootstrap'],oof_hash=sha(OUT/'extra_oof.csv')));print('EXTRA_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
