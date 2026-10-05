"""Independent accounting and meta leakage/learning checks; no adoption decision."""
from pathlib import Path
import sys,math,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('investigator',H/'run_v1.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
np,pd=D.np,D.pd
vs=importlib.util.spec_from_file_location('original_learning_verifier',H.parent/'ec_gate_models_20261005_v1/verify_v1.py');V=importlib.util.module_from_spec(vs);vs.loader.exec_module(V)
def scalar(frame,g,mask):
    vals={name:[] for name in ['sse_A','sse_B','sse_P','sse_oracle','linear','quadratic','net','positive_loss','wrong_positive_loss','overshoot_positive_loss','partial_oracle_gain','total_oracle_gain','gate_sum','residual_sum','delta_sum','cost_positive','cost_negative','cost_gate_positive','cost_gate_negative']};counts={name:0 for name in ['wrong_rows','wrong_changed','overshoot_rows','B_better','A_better','partial_help_rows','up_rows','down_rows']}
    for a,b,y,gate in zip(frame.A.to_numpy()[mask],frame.B.to_numpy()[mask],frame.y.to_numpy()[mask],np.asarray(g)[mask]):
        a,b,y,gate=map(float,[a,b,y,gate]);r=y-a;delta=b-a;step=gate*delta;p=a+step;aa=(a-y)**2;bb=(b-y)**2;pp=(p-y)**2;ld=pp-aa;lin=-2*r*step;quad=step**2;oracle=min(1,max(0,r/delta)) if delta else 0;os=(a+oracle*delta-y)**2;cost=aa-bb;wrong=r*delta<=0 and delta!=0;over=r*delta>0 and abs(step)>2*abs(r);partial=0<oracle<.5 and cost<0
        for name,value in [('sse_A',aa),('sse_B',bb),('sse_P',pp),('sse_oracle',os),('linear',lin),('quadratic',quad),('net',ld),('positive_loss',max(0,ld)),('wrong_positive_loss',max(0,ld) if wrong else 0),('overshoot_positive_loss',max(0,ld) if over else 0),('partial_oracle_gain',aa-os if partial else 0),('total_oracle_gain',aa-os),('gate_sum',gate),('residual_sum',r),('delta_sum',delta),('cost_positive',max(0,cost)),('cost_negative',max(0,-cost)),('cost_gate_positive',max(0,cost)*gate),('cost_gate_negative',max(0,-cost)*gate)]:vals[name].append(value)
        for name,value in [('wrong_rows',wrong),('wrong_changed',wrong and step!=0),('overshoot_rows',over),('B_better',cost>0),('A_better',cost<0),('partial_help_rows',partial),('up_rows',delta>0),('down_rows',delta<0)]:counts[name]+=int(value)
    out={name:math.fsum(values) for name,values in vals.items()};out.update(counts);out['n']=int(mask.sum());out['days']=len(frame.loc[mask,['farm','day']].drop_duplicates());out['anchor_sum']=math.fsum(map(float,frame.loc[mask,'anchor']));out['ref_n_sum']=math.fsum(map(float,frame.loc[mask,'ref_n']));assert abs(out['net']-out['linear']-out['quadratic'])<1e-9;return out
def main():
    D.frozen();refs,sig,p=D.prepare();assert p==D.load(H/'preparation_v1.json');receipt=D.load(H/'fit_receipt_v1.json');assert receipt['status']=='COMPLETE_DIAGNOSIS_META270_NO_ADOPTION' and receipt['source']==D.sha(H/'run_v1.py');assert receipt['registration_sha']==D.sha(H/'registration_v1.json');assert D.sha(H/'fold_summaries_v1.csv')==receipt['summary_sha'] and D.sha(H/'daily_v1.csv')==receipt['daily_sha'] and D.sha(H/'cases_v1.json')==receipt['cases_sha']
    summaries=pd.read_csv(H/'fold_summaries_v1.csv',float_precision='round_trip');assert not summaries.duplicated(['mode','validator','fold','seed','scope','segment']).any();index={(r.mode,r.validator,int(r.fold),int(r.seed),r.scope,r.segment):r._asdict() for r in summaries.itertuples(index=False)};checked=set();maxerr=0;gradmax={m:0. for m in D.M.MODES}
    for rec in receipt['files']:
        path=D.OUT/rec['path'];assert D.sha(path)==rec['sha'];f=pd.read_csv(path,float_precision='round_trip');assert len(f)==rec['n'] and f.row_id.is_unique;mode,v,k,s,scope=[rec[key] for key in ['mode','validator','fold','seed','scope']];ref=refs[v,k];inner=scope!='outer_actual';q,a,b,x,ok=D.B.pair(ref,s,inner);assert np.array_equal(f.row_id,q.row_id);D.close(f.A,a);D.close(f.B,b);D.close(f.y,q.sub_ec);assert np.array_equal(f.high,f.groupby(['farm','day']).y.transform('mean')>=1)
        comparisons=[(scope,f,f.g.to_numpy())]
        if scope=='outer_actual':
            NT,QT,ST=D.R.neighbors(ref['a'],q,sig);thin=ref.copy();thin.update(NQ=NT,QQ=QT);tq,ta,tb,tx,tok=D.B.pair(thin,s,False);m=D.load(D.M.OUT/mode/f'{v}_{k}_{s}_fit.json');g=np.where(ok,D.M.forward(mode,x,m),0);tg=np.where(tok,D.M.forward(mode,tx,m),0);D.close(f.g,g);D.close(f.B_thin,tb);D.close(f.g_thin,tg);D.close(f.anchor_thin,NT[:,0]);D.close(f.ref_n_thin,np.expm1(NT[:,4]));tf=f.copy();tf['B']=tb;tf['anchor']=NT[:,0];tf['ref_n']=np.expm1(NT[:,4]);delta=f.B.to_numpy()-f.A.to_numpy();r=f.y.to_numpy()-f.A.to_numpy();oracle=np.clip(np.divide(r,delta,out=np.zeros(len(f)),where=delta!=0),0,1)
            comparisons += [('thin_B_fixed_g',tf,g),('thin_x_fixed_B',f,tg),('thin_B_thin_g',tf,tg),('no_down',f,np.where(delta<0,0,g)),('no_up',f,np.where(delta>0,0,g)),('oracle_no_wrong',f,np.where((r*delta<=0)&(delta!=0),0,g)),('oracle_cap_optimum',f,np.minimum(g,oracle))]
        elif scope=='inner_resub':
            m=D.load(D.M.OUT/mode/f'{v}_{k}_{s}_fit.json');D.close(f.g,np.where(ok,D.M.forward(mode,x,m),0))
        else:
            seen=np.zeros(len(f),int);pred=np.full(len(f),np.nan)
            meta=[z for z in receipt['meta'] if z['mode']==mode and z['validator']==v and z['fold']==k and z['seed']==s];assert len(meta)==3
            for j,record in enumerate(meta):
                assert record['meta_fold']==j;ti,vi=D.meta_split(ref,j);assert ti.tolist()==record['train_indices'] and vi.tolist()==record['valid_indices'];new=D.meta_ref(ref,s,ti,vi);assert D.R.ids(new['b'].row_id)==record['train_ids'] and D.R.ids(new['q'].row_id)==record['valid_ids'];assert not set(new['b'].row_id)&set(new['q'].row_id)
                forbidden={(ff,int(day)+dd) for ff,day in new['q'][['farm','day']].itertuples(index=False,name=None) for dd in [-1,0,1]};assert not set(new['b'][['farm','day']].itertuples(index=False,name=None))&forbidden
                fp=D.OUT/f'meta_{mode}_{v}_{k}_{s}_{j}.json';assert D.sha(fp)==record['model_sha'];model=D.load(fp);gradmax[mode]=max(gradmax[mode],V.learning(mode,new,s,model));mq,ma,mb,mx,mok=D.B.pair(new,s,False);D.close(mb,b[vi]);pred[vi]=np.where(mok,D.M.forward(mode,mx,model),0);seen[vi]+=1
                if (k,s,j)==(0,7,0):
                    with D.M.threadpool_limits(limits=2):pp,gg,repeated=D.M.fit(mode,new,s)
                    assert repeated==model;D.close(gg,pred[vi]);D.close(pp,ma+pred[vi]*(mb-ma))
            assert (seen==1).all();D.close(f.g,pred)
        for name,frame,g in comparisons:
            masks=[('all',np.ones(len(frame),bool)),('high',frame.high.to_numpy()),('ordinary',~frame.high.to_numpy()),('pass1',(frame.day<179).to_numpy()),('pass2',(frame.day>=179).to_numpy())]+[(farm+'_pass'+str(phase),((frame.farm==farm)&((frame.day>=179)==(phase==2))).to_numpy()) for farm in ['F13','F47'] for phase in [1,2]]
            for segment,mask in masks:
                if not mask.any():continue
                key=(mode,v,k,s,name,segment);assert key in index;manual=scalar(frame,g,mask);record=index[key]
                for col,value in manual.items():
                    error=abs(value-record[col]);maxerr=max(maxerr,error);assert error<1e-8,(key,col,error)
                checked.add(key)
        print('VERIFY',scope,mode,v,k,s,flush=True)
    assert checked==set(index);daily=pd.read_csv(H/'daily_v1.csv',float_precision='round_trip');assert np.max(abs(daily.net-daily.level-daily['shape']))<1e-10
    for (mode,v,k,s),g in daily.groupby(['mode','validator','fold','seed']):
        row=index[mode,v,int(k),int(s),'outer_actual','all'];assert abs(math.fsum(g.net)-row['net'])<1e-9
    numeric=[c for c in summaries.columns if c not in ['mode','validator','fold','seed','scope','segment']];pooled=summaries.groupby(['mode','validator','seed','scope','segment'],sort=False)[numeric].sum().reset_index();pooled['rmse_A']=np.sqrt(pooled.sse_A/pooled.n);pooled['rmse_B']=np.sqrt(pooled.sse_B/pooled.n);pooled['rmse_P']=np.sqrt(pooled.sse_P/pooled.n);pooled['change_pct']=100*(pooled.rmse_P/pooled.rmse_A-1);pooled['wrong_share_positive']=np.divide(pooled.wrong_positive_loss,pooled.positive_loss,out=np.zeros(len(pooled)),where=pooled.positive_loss!=0);pooled['overshoot_share_positive']=np.divide(pooled.overshoot_positive_loss,pooled.positive_loss,out=np.zeros(len(pooled)),where=pooled.positive_loss!=0);pooled['partial_share_oracle_gain']=np.divide(pooled.partial_oracle_gain,pooled.total_oracle_gain,out=np.zeros(len(pooled)),where=pooled.total_oracle_gain!=0);pooled['weighted_B_win_rate']=pooled.cost_positive/(pooled.cost_positive+pooled.cost_negative);pooled['gate_mean']=pooled.gate_sum/pooled.n;pooled['ref_n_mean']=pooled.ref_n_sum/pooled.n;pooled['residual_mean']=pooled.residual_sum/pooled.n;pooled['delta_mean']=pooled.delta_sum/pooled.n;pooled.to_csv(H/'pooled_v1.csv',index=False)
    D.save(H/'verification_v1.json',dict(status='PASS_DIAGNOSIS_AND_META270',summary_rows=len(summaries),checked_rows=len(checked),meta_models=270,fresh_repeat_checks=3,max_scalar_error=maxerr,gradient_max=gradmax,source_sha=D.sha(H/'run_v1.py'),verifier_sha=D.sha(Path(__file__)),pooled_sha=D.sha(H/'pooled_v1.csv'),adoption=False,limitations=['posthoc public diagnostics','meta b repeats across outer folds: occurrence counts are not independent days','reference intervention holds outer A fixed, not smaller-a base retraining','oracle reads outer y; never deploy']))
    print('PASS_DIAGNOSIS_AND_META270',flush=True)
if __name__=='__main__':main()
