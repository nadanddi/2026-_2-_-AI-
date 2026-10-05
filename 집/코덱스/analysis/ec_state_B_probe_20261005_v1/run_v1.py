"""One fixed residual B and mean control, honest date OOF, equal-size resub control."""
from pathlib import Path
import sys,json,math,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sp=importlib.util.spec_from_file_location('diagnosis',H.parent/'ec_gate_failure_investigation_20261005_v1/run_v1.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
np,pd=D.np,D.pd;M=D.M;B=D.B;R=D.R
from lightgbm import LGBMRegressor
OUT=ROOT/'집/코덱스/local'/H.name
sha,save,load,close=D.sha,D.save,D.load,D.close
COLS=[0,1]+list(range(8,23));SEEDS=[7,101,2024];MODES=['MEAN','STATE']
CFG=dict(family=31,alpha=.025/31,features=COLS,cap=.3,params=dict(n_estimators=100,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=96,min_split_gain=.01,reg_lambda=10,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True),gate_fits=0)
def pair(ref,s,inner):
    q,a,old,x,ok=B.pair(ref,s,inner);return q,a,x[:,COLS]
def equal_train(q,ti,vi):
    included=[]
    for farm in ['F13','F47']:
        t=[int(i) for i in ti if q.iloc[i].farm==farm];v=[int(i) for i in vi if q.iloc[i].farm==farm];assert len(t)>=len(v)
        t=sorted(t,key=lambda i:(int(q.iloc[i].day),int(q.iloc[i].hour)))
        included+=t[len(v):]+v
    ii=np.array(sorted(included),int);assert len(ii)==len(ti) and len(set(ii))==len(ii) and set(vi)<=set(ii)
    for f in ['F13','F47']:assert (q.iloc[ii].farm==f).sum()==(q.iloc[ti].farm==f).sum()
    return ii
def prepare():
    refs,sig,p=D.prepare();assert p==load(H.parent/'ec_gate_failure_investigation_20261005_v1/preparation_v1.json');records=[]
    for (v,k),ref in refs.items():
        for s in SEEDS:
            iq,ia,ix=pair(ref,s,True);oq,oa,ox=pair(ref,s,False);assert ix.shape[1]==17 and np.isfinite(ix).all() and np.isfinite(ox).all()
            assert not set(iq.row_id)&set(oq.row_id)
            records.append(dict(v=v,k=k,s=s,inner_ids=R.ids(iq.row_id),outer_ids=R.ids(oq.row_id),inner_x=R.ar(ix),outer_x=R.ar(ox),inner_a=R.ar(ia),outer_a=R.ar(oa)))
            if v=='DIAG10':
                for j in range(3):ti,vi=D.meta_split(ref,j);equal_train(iq,ti,vi)
    return refs,dict(config=CFG,records=records,parent_sha=sha(H.parent/'ec_gate_failure_investigation_20261005_v1/preparation_v1.json'),status='PREPARED_FIT0')
def fit(mode,q,a,x,indices,s):
    t=q.sub_ec.to_numpy()[indices]-a[indices];m=dict(mode=mode,seed=s,indices=np.asarray(indices).tolist(),ids=R.ids(q.row_id.iloc[indices]),target_sha=R.ar(t),n=len(t),mean=float(t.mean()))
    if mode=='STATE':
        model=LGBMRegressor(**CFG['params'],random_state=s).fit(x[indices],t);m['tree']=model.booster_.dump_model();close(model.predict(x[indices]),forward(x[indices],m));m['importance']=model.feature_importances_.tolist()
    return m
def forward(x,m):return np.full(len(x),m['mean']) if m['mode']=='MEAN' else M.tree_logits(x,m['tree'])
def pred(a,x,m,bounds):return np.clip(a+np.clip(forward(x,m),-CFG['cap'],CFG['cap']),*bounds)
def frozen():
    for n,h in load(H/'registration_v1.json')['hashes'].items():assert sha(H/n)==h,n
def actual():
    frozen();refs,pr=prepare();assert pr==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);files=[];models=[]
    for (v,k),ref in refs.items():
        for s in SEEDS:
            q,a,x=pair(ref,s,True);oq,oa,ox=pair(ref,s,False)
            for mode in MODES:
                with M.threadpool_limits(limits=2):model=fit(mode,q,a,x,np.arange(len(q)),s)
                fp=OUT/f'full_{mode}_{v}_{k}_{s}.json';save(fp,model);models.append(dict(scope='outer',mode=mode,v=v,k=k,s=s,path=fp.name,sha=sha(fp)))
                p=pred(oa,ox,model,ref['bounds']);close(p,pred(oa[::-1],ox[::-1],model,ref['bounds'])[::-1]);f=oq[['row_id','farm','day','hour']].copy();f['y']=oq.sub_ec;f['A']=oa;f['P']=p;f['v']=v;f['k']=k;f['s']=s;f['mode']=mode;f['scope']='outer';path=OUT/f'outer_{mode}_{v}_{k}_{s}.csv';f.to_csv(path,index=False);files.append(dict(path=path.name,sha=sha(path),scope='outer',mode=mode,v=v,k=k,s=s))
                if v!='DIAG10':continue
                excluded=np.full(len(q),np.nan);included=excluded.copy();seen=np.zeros(len(q),int)
                for j in range(3):
                    ti,vi=D.meta_split(ref,j);ii=equal_train(q,ti,vi)
                    for context,idx in [('excluded',ti),('included_equal_n',ii)]:
                        with M.threadpool_limits(limits=2):mm=fit(mode,q,a,x,idx,s)
                        fp=OUT/f'meta_{mode}_{k}_{s}_{j}_{context}.json';save(fp,mm);models.append(dict(scope=context,mode=mode,v=v,k=k,s=s,j=j,path=fp.name,sha=sha(fp)))
                        pp=pred(a[vi],x[vi],mm,ref['innerbounds']);(excluded if context=='excluded' else included)[vi]=pp
                    seen[vi]+=1
                assert (seen==1).all() and np.isfinite(excluded).all() and np.isfinite(included).all()
                for context,p in [('excluded',excluded),('included_equal_n',included)]:
                    f=q[['row_id','farm','day','hour']].copy();f['y']=q.sub_ec;f['A']=a;f['P']=p;f['v']=v;f['k']=k;f['s']=s;f['mode']=mode;f['scope']=context;path=OUT/f'{context}_{mode}_{v}_{k}_{s}.csv';f.to_csv(path,index=False);files.append(dict(path=path.name,sha=sha(path),scope=context,mode=mode,v=v,k=k,s=s))
            frozen();print('B_PROBE',v,k,s,'COMPLETE',flush=True)
    assert len(models)==492 and len(files)==252
    save(H/'receipt_v1.json',dict(status='COMPLETE_B_ONLY_GATE0',files=files,models=models,registration_sha=sha(H/'registration_v1.json')));print('B_PROBE_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:refs,p=prepare();save(H/'preparation_v1.json',p);print('PREP_PASS',flush=True)
    else:actual()
