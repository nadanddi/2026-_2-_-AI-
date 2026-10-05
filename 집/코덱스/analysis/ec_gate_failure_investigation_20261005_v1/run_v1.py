"""Exact loss accounting, reference intervention and honest second-stage meta OOF."""
from pathlib import Path
import sys,json,math,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sp=importlib.util.spec_from_file_location('gate_models',H.parent/'ec_gate_models_20261005_v1/run_v2.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
np,pd=M.np,M.pd;B=M.B;R=B.R
OUT=ROOT/'집/코덱스/local'/H.name
sha,save,load,close=M.sha,M.save,M.load,M.close
CASES=[('F47',229),('F47',231),('F13',151),('F47',154),('F13',231),('F13',233),('F47',216)]
def prepare():
    M.frozen();refs,prior=M.prep();assert prior==load(H.parent/'ec_gate_models_20261005_v1/preparation_v1.json');assert load(H.parent/'ec_gate_models_20261005_v1/verification_all_v1.json')['status']=='PASS_ALL198'
    wanted=set().union(*(set(ref['q'].row_id) for (v,k),ref in refs.items() if v=='DIAG10'));raw=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+R.CC);raw=raw[raw.row_id.isin(wanted)].copy();raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int);assert len(raw)==8640;sig=R.signatures(raw)
    return refs,sig,dict(status='PREPARED_FIT0_SCORE0',source=sha(Path(__file__)),prior_sha=sha(H.parent/'ec_gate_models_20261005_v1/preparation_v1.json'),prior=prior,plan_sha=sha(H/'questions_v1.md'),primary='DIAG10 with A/B supporting tables',meta_folds=3,meta_block=5,meta_purge=1,meta_expected=270)
def spine(q,a,b,N):
    d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec.to_numpy();d['A']=a;d['B']=b;d['anchor']=N[:,0];d['d1']=N[:,1];d['ref_n']=np.expm1(N[:,4]);d['high']=d.groupby(['farm','day']).y.transform('mean')>=1;return d
def measures(d,g):
    a=d.A.to_numpy();b=d.B.to_numpy();y=d.y.to_numpy();delta=b-a;r=y-a;step=np.asarray(g)*delta;p=a+step;linear=-2*r*step;quad=step*step;ld=(p-y)**2-(a-y)**2;close(ld,linear+quad)
    oracle=np.clip(np.divide(r,delta,out=np.zeros(len(d)),where=delta!=0),0,1);return dict(a=a,b=b,y=y,d=delta,r=r,g=np.asarray(g),p=p,linear=linear,quad=quad,ld=ld,oracle=oracle,cost=(a-y)**2-(b-y)**2,wrong=(r*delta<=0)&(delta!=0),overshoot=(r*delta>0)&(abs(step)>2*abs(r)))
def summaries(d,g,mode,v,k,s,scope):
    z=measures(d,g);masks=[('all',np.ones(len(d),bool)),('high',d.high.to_numpy()),('ordinary',~d.high.to_numpy()),('pass1',(d.day<179).to_numpy()),('pass2',(d.day>=179).to_numpy())]
    masks += [(f+'_pass'+str(p),((d.farm==f)&((d.day>=179)==(p==2))).to_numpy()) for f in ['F13','F47'] for p in [1,2]]
    result=[]
    for name,mask in masks:
        if not mask.any():continue
        a,b,y,p,g,cost,r,delta,ld,lin,quad,o=[z[key][mask] for key in ['a','b','y','p','g','cost','r','d','ld','linear','quad','oracle']];wrong=z['wrong'][mask];over=z['overshoot'][mask];positive=cost>0;negative=cost<0;partial=(o>0)&(o<.5)&negative
        result.append(dict(mode=mode,validator=v,fold=k,seed=s,scope=scope,segment=name,n=int(mask.sum()),days=len(d.loc[mask,['farm','day']].drop_duplicates()),sse_A=float(np.sum((a-y)**2)),sse_B=float(np.sum((b-y)**2)),sse_P=float(np.sum((p-y)**2)),sse_oracle=float(np.sum((a+o*delta-y)**2)),linear=float(lin.sum()),quadratic=float(quad.sum()),net=float(ld.sum()),positive_loss=float(ld[ld>0].sum()),wrong_rows=int(wrong.sum()),wrong_changed=int((wrong&(g*delta!=0)).sum()),wrong_positive_loss=float(ld[wrong&(ld>0)].sum()),overshoot_rows=int(over.sum()),overshoot_positive_loss=float(ld[over&(ld>0)].sum()),B_better=int(positive.sum()),A_better=int(negative.sum()),partial_help_rows=int(partial.sum()),partial_oracle_gain=float(((a-y)**2-(a+o*delta-y)**2)[partial].sum()),total_oracle_gain=float(((a-y)**2-(a+o*delta-y)**2).sum()),gate_sum=float(g.sum()),up_rows=int((delta>0).sum()),down_rows=int((delta<0).sum()),residual_sum=float(r.sum()),delta_sum=float(delta.sum()),anchor_sum=float(d.loc[mask,'anchor'].sum()),ref_n_sum=float(d.loc[mask,'ref_n'].sum()),cost_positive=float(cost[positive].sum()),cost_negative=float(-cost[negative].sum()),cost_gate_positive=float((cost[positive]*g[positive]).sum()),cost_gate_negative=float((-cost[negative]*g[negative]).sum())))
    return result
def daily(d,g,mode,v,k,s):
    z=measures(d,g);t=d[['farm','day','hour']].copy();t['y']=z['y'];t['A']=z['a'];t['B']=z['b'];t['P']=z['p'];t['g']=g;t['ld']=z['ld'];t['linear']=z['linear'];t['quad']=z['quad'];records=[]
    for (f,day),q in t.groupby(['farm','day']):
        yy,aa,pp=q.y.mean(),q.A.mean(),q.P.mean();level=len(q)*((pp-yy)**2-(aa-yy)**2);shape=((q.P-pp-(q.y-yy))**2-(q.A-aa-(q.y-yy))**2).sum();close([q.ld.sum()],[level+shape]);records.append(dict(mode=mode,validator=v,fold=k,seed=s,farm=f,day=int(day),n=len(q),mean_y=yy,mean_A=aa,mean_B=q.B.mean(),mean_P=pp,mean_gate=q.g.mean(),net=q.ld.sum(),linear=q.linear.sum(),quadratic=q.quad.sum(),level=level,shape=shape))
    return records
def meta_split(ref,j):
    q=ref['b'];chosen=set()
    for f,g in q.groupby('farm'):
        days=sorted(g.day.unique());chosen|={(f,int(day)) for i,day in enumerate(days) if (i//5)%3==j}
    banned={(f,d+offset) for f,d in chosen for offset in [-1,0,1]};valid=np.array([(f,int(d)) in chosen for f,d in zip(q.farm,q.day)]);train=np.array([(f,int(d)) not in banned for f,d in zip(q.farm,q.day)]);assert not (train&valid).any();assert valid.any() and train.any();return np.flatnonzero(train),np.flatnonzero(valid)
def meta_ref(ref,s,ti,vi):
    q=ref['b'];a=ref['inner'][s];return dict(b=q.iloc[ti].reset_index(drop=True),inner={s:a[ti]},NB=ref['NB'][ti],QB=ref['QB'][ti],innerbounds=ref['innerbounds'],q=q.iloc[vi].reset_index(drop=True),baseline={s:a[vi]},NQ=ref['NB'][vi],QQ=ref['QB'][vi],bounds=ref['innerbounds'])
def frozen():
    reg=load(H/'registration_v1.json')
    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name
def actual():
    frozen();refs,sig,receipt=prepare();assert receipt==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);rows=[];days=[];cases=[];files=[];metas=[]
    for (v,k),ref in refs.items():
        if v not in ['DIAG10','A','B']:continue
        NT,QT,ST=R.neighbors(ref['a'],ref['q'],sig);thin=ref.copy();thin.update(NQ=NT,QQ=QT)
        sources={z['row_id']:z['anchors'] for z in ref['SQ']};labelref=ref['tr'].groupby(['farm','day']).sub_ec.mean()
        for s in M.SEEDS:
            iq,ia,ib,ix,iok=B.pair(ref,s,True);oq,oa,ob,ox,ook=B.pair(ref,s,False);tq,ta,tb,tx,tok=B.pair(thin,s,False);inner=spine(iq,ia,ib,ref['NB']);outer=spine(oq,oa,ob,ref['NQ']);outthin=spine(oq,oa,tb,NT)
            for mode in M.MODES:
                m=load(M.OUT/mode/f'{v}_{k}_{s}_fit.json');original=pd.read_csv(M.OUT/mode/f'{v}_{k}_{s}.csv',float_precision='round_trip');assert np.array_equal(original.row_id,oq.row_id);g=np.where(ook,M.forward(mode,ox,m),0);close(g,original.g);close(oa+g*(ob-oa),original.candidate)
                ig=np.where(iok,M.forward(mode,ix,m),0);tg=np.where(tok,M.forward(mode,tx,m),0);z=measures(outer,g)
                scopes=[('outer_actual',outer,g),('inner_resub',inner,ig),('thin_B_fixed_g',outthin,g),('thin_x_fixed_B',outer,tg),('thin_B_thin_g',outthin,tg),('no_down',outer,np.where(z['d']<0,0,g)),('no_up',outer,np.where(z['d']>0,0,g)),('oracle_no_wrong',outer,np.where(z['wrong'],0,g)),('oracle_cap_optimum',outer,np.minimum(g,z['oracle']))]
                for scope,frame,weights in scopes:rows+=summaries(frame,weights,mode,v,k,s,scope)
                days+=daily(outer,g,mode,v,k,s)
                for scope,frame,weights in [('outer_actual',outer,g),('inner_resub',inner,ig)]:
                    f=frame.copy();f['g']=weights
                    if scope=='outer_actual':f['B_thin']=tb;f['g_thin']=tg;f['anchor_thin']=NT[:,0];f['ref_n_thin']=np.expm1(NT[:,4])
                    path=OUT/f'{scope}_{mode}_{v}_{k}_{s}.csv';f.to_csv(path,index=False);files.append(dict(scope=scope,mode=mode,validator=v,fold=k,seed=s,path=path.name,sha=sha(path),n=len(f)))
                if v=='DIAG10':
                    mg=np.full(len(iq),np.nan);seen=np.zeros(len(iq),int)
                    for j in range(3):
                        ti,vi=meta_split(ref,j);new=meta_ref(ref,s,ti,vi);frozen()
                        with M.threadpool_limits(limits=2):mp,gg,mm=M.fit(mode,new,s)
                        mq,ma,mb,mx,mok=B.pair(new,s,False);close(mb,ib[vi]);close(mx,ix[vi]);close(mp,ma+gg*(mb-ma));mg[vi]=gg;seen[vi]+=1
                        rec=dict(mode=mode,validator=v,fold=k,seed=s,meta_fold=j,train_indices=ti.tolist(),valid_indices=vi.tolist(),train_ids=R.ids(new['b'].row_id),valid_ids=R.ids(new['q'].row_id),model_sha=None)
                        fp=OUT/f'meta_{mode}_{v}_{k}_{s}_{j}.json';save(fp,mm);rec['model_sha']=sha(fp);metas.append(rec)
                    assert np.isfinite(mg).all() and (seen==1).all();rows+=summaries(inner,mg,mode,v,k,s,'inner_meta_oof');f=inner.copy();f['g']=mg;path=OUT/f'inner_meta_oof_{mode}_{v}_{k}_{s}.csv';f.to_csv(path,index=False);files.append(dict(scope='inner_meta_oof',mode=mode,validator=v,fold=k,seed=s,path=path.name,sha=sha(path),n=len(f)))
                for ii,r in oq.iterrows():
                    if (r.farm,int(r.day)) not in CASES or int(r.hour) not in [0,6,12,23] or v!='DIAG10':continue
                    sd=sources[r.row_id];sy=[float(labelref[r.farm,int(day)]) for day in sd];assert all(day<int(r.day) for day in sd)
                    if sy:close([np.mean(sy)],[ref['NQ'][ii,0]])
                    cases.append(dict(mode=mode,seed=s,fold=k,row_id=r.row_id,farm=r.farm,day=int(r.day),hour=int(r.hour),y=float(r.sub_ec),A=float(oa[ii]),B=float(ob[ii]),residual=float(r.sub_ec-oa[ii]),delta=float(ob[ii]-oa[ii]),gate=float(g[ii]),anchor=float(ref['NQ'][ii,0]),prefix_A=float(ox[ii,1]),reference_days=sd,reference_labels=sy,ref_n=int(round(np.expm1(ref['NQ'][ii,4]))),B_thin=float(tb[ii]),gate_thin=float(tg[ii])))
            print('DIAGNOSIS',v,k,s,'COMPLETE',flush=True)
    assert len(metas)==270;pd.DataFrame(rows).to_csv(H/'fold_summaries_v1.csv',index=False);pd.DataFrame(days).to_csv(H/'daily_v1.csv',index=False);save(H/'cases_v1.json',cases);save(H/'fit_receipt_v1.json',dict(status='COMPLETE_DIAGNOSIS_META270_NO_ADOPTION',source=sha(Path(__file__)),preparation_sha=sha(H/'preparation_v1.json'),registration_sha=sha(H/'registration_v1.json'),files=files,meta=metas,summary_sha=sha(H/'fold_summaries_v1.csv'),daily_sha=sha(H/'daily_v1.csv'),cases_sha=sha(H/'cases_v1.json')));print('META270_DIAGNOSIS_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:refs,sig,r=prepare();save(H/'preparation_v1.json',r);print('PREPARATION_PASS',flush=True)
    else:actual()
