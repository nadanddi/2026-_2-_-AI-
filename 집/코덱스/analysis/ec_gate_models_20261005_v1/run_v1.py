"""Three predeclared, cost-sensitive classification soft gates on identical A/B."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,argparse,warnings
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sp=importlib.util.spec_from_file_location('soft_base',H.parent/'ec_soft_gate_20261005_v1/run_v2.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
sha,save,load,close=B.sha,B.save,B.load,B.close
MODES=['LR','LGB','MLP'];SEEDS=B.R.SEEDS;KEYS=B.R.KEYS;OUT=ROOT/'집/코덱스/local'/H.name
CFG=dict(families=dict(LR=28,LGB=29,MLP=30),alpha=.025/30,seeds=SEEDS,keys=[list(k) for k in KEYS],gate='weighted B-wins probability soft blend',tie_epsilon=1e-14,judged=['DIAG10','A','B'],lr=dict(C=1,max_iter=3000,tol=1e-8),lgb=dict(n_estimators=150,learning_rate=.03,num_leaves=7,max_depth=3,min_child_samples=30,min_split_gain=.01,reg_lambda=10,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True),mlp=dict(hidden_layer_sizes=[8],activation='tanh',solver='lbfgs',alpha=10,max_iter=3000,max_fun=100000,tol=1e-7))
def prep():
    B.frozen();refs,prior=B.prep();assert prior==load(H.parent/'ec_soft_gate_20261005_v1/preparation_v2.json')
    rows=[]
    for (v,k),ref in refs.items():
        for s in SEEDS:
            q,a,b,x,ok=B.pair(ref,s,True);cost=(a-q.sub_ec.to_numpy())**2-(b-q.sub_ec.to_numpy())**2;eligible=ok&(abs(cost)>CFG['tie_epsilon']);rows.append(dict(validator=v,fold=k,seed=s,eligible_ids=B.R.ids(q.row_id[eligible]),cost_sha=B.R.ar(cost),x_sha=B.R.ar(x),positive=int((cost[eligible]>0).sum()),n=int(eligible.sum())))
    return refs,dict(status='PREPARED_FIT0_SCORE0',source=sha(Path(__file__)),config=CFG,prior_sha=sha(H.parent/'ec_soft_gate_20261005_v1/preparation_v2.json'),prior=prior,records=rows)
def tree_value(node,row):
    while 'split_index' in node:
        assert node['decision_type']=='<=' and node.get('missing_type') in ['None','NaN','Zero'];val=row[int(node['split_feature'])]
        left=val<=float(node['threshold'])
        if node.get('missing_type')=='Zero' and val==0:left=node['default_left']
        if np.isnan(val):left=node['default_left']
        node=node['left_child'] if left else node['right_child']
    return float(node['leaf_value'])
def forward(mode,x,m):
    if m['constant'] is not None:return np.full(len(x),m['constant'])
    z=(x-np.asarray(m['mean']))/m['scale']
    if mode=='LR':return expit(z@np.asarray(m['coef']).reshape(-1)+m['intercept'])
    if mode=='MLP':return expit((np.tanh(z@np.asarray(m['coefs'][0])+m['intercepts'][0])@np.asarray(m['coefs'][1])+m['intercepts'][1]).reshape(-1))
    assert m['tree']['objective'].startswith('binary') and m['tree']['average_output']==False
    return expit(np.array([math.fsum(tree_value(t['tree_structure'],row) for t in m['tree']['tree_info']) for row in z]))
def fit(mode,ref,s):
    q,a,b,x,ok=B.pair(ref,s,True);oq,oa,ob,ox,ook=B.pair(ref,s,False);y=q.sub_ec.to_numpy();cost=(a-y)**2-(b-y)**2;eligible=ok&(abs(cost)>CFG['tie_epsilon']);n=int(eligible.sum())
    m=dict(mode=mode,seed=s,n=n,eligible_ids=B.R.ids(q.row_id[eligible]),cost_sha=B.R.ar(cost),constant=0.,mean=[0.]*23,scale=[1.]*23)
    if n:
        scale=StandardScaler().fit(x[eligible]);z=scale.transform(x[eligible]);t=(cost[eligible]>0).astype(int);w=abs(cost[eligible]);w=w/w.mean();m.update(mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),weight_sha=B.R.ar(w),label_sha=B.R.ar(t),positive=int(t.sum()))
        if len(np.unique(t))<2:m['constant']=float(t[0])
        else:
            if mode=='LR':model=LogisticRegression(**CFG['lr'],random_state=s)
            elif mode=='LGB':model=LGBMClassifier(**CFG['lgb'],random_state=s)
            else:
                params=CFG['mlp'].copy();params['hidden_layer_sizes']=tuple(params['hidden_layer_sizes']);model=MLPClassifier(**params,random_state=s)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(z,t,sample_weight=w)
            assert np.array_equal(model.classes_,[0,1]);m['constant']=None
            if mode=='LR':m.update(coef=model.coef_.tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]))
            elif mode=='LGB':m.update(tree=model.booster_.dump_model(),iterations=model.booster_.num_trees())
            else:m.update(coefs=[p.tolist() for p in model.coefs_],intercepts=[p.tolist() for p in model.intercepts_],iterations=int(model.n_iter_),loss=float(model.loss_))
            close(model.predict_proba(scale.transform(ox))[:,1],forward(mode,ox,m));close(model.predict_proba(z)[:,1],forward(mode,x[eligible],m))
        tg=forward(mode,x[eligible],m);r=a[eligible]+tg*(b[eligible]-a[eligible])-y[eligible];m.update(train_mse=float(np.mean(r*r)),train_a_mse=float(np.mean((a[eligible]-y[eligible])**2)),train_logloss=float(np.average(-(t*np.log(np.clip(tg,1e-15,1))+(1-t)*np.log(np.clip(1-tg,1e-15,1))),weights=w)))
    g=np.where(ook,forward(mode,ox,m),0);p=oa+g*(ob-oa);assert np.array_equal(p[~ook],oa[~ook]);return p,g,m
def frozen():
    reg=load(H/'registration_v1.json')
    for name,digest in reg['hashes'].items():assert sha(H/name)==digest,name
def actual():
    frozen();refs,receipt=prep();assert receipt==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False)
    for mode in MODES:
        dest=OUT/mode;dest.mkdir();manifest=[]
        for (v,k),ref in refs.items():
            for s in SEEDS:
                frozen()
                with threadpool_limits(limits=2):p,g,m=fit(mode,ref,s)
                q,a,b,x,ok=B.pair(ref,s,False);close(g,np.where(ok,forward(mode,x[::-1],m)[::-1],0));assert np.all((g>=0)&(g<=1))
                if (v,k,s)==('DIAG10',0,7):
                    with threadpool_limits(limits=2):p2,g2,m2=fit(mode,ref,s)
                    close(p,p2);close(g,g2);assert m==m2
                    close(g[:24],np.array([forward(mode,x[i:i+1],m)[0] if ok[i] else 0 for i in range(24)]));alter=x[:8].copy();alter[1:]+=1e4;close(forward(mode,x[:1],m),forward(mode,alter,m)[:1]);save(H/f'first_{mode}_v1.json',dict(status='PASS',repeat=close(p,p2),serialized_query_independence=True))
                d=q[['row_id','farm','day','hour']].copy();d['y']=q.sub_ec;d['A']=a;d['B']=b;d['g']=g;d['candidate']=p;d['eligible']=ok;d['validator']=v;d['fold']=k;d['seed']=s;path=dest/f'{v}_{k}_{s}.csv';d.to_csv(path,index=False);save(dest/f'{v}_{k}_{s}_fit.json',m);manifest.append(dict(validator=v,fold=k,seed=s,csv_sha=sha(path),fit_sha=sha(dest/f'{v}_{k}_{s}_fit.json')))
            print(mode,v,k,'COMPLETE',flush=True)
        assert len(manifest)==66;agg=pd.concat([pd.read_csv(dest/f'{v}_{k}_{s}.csv',float_precision='round_trip') for v,k in KEYS for s in SEEDS],ignore_index=True);agg.to_csv(dest/'oof.csv',index=False);save(H/f'fit_{mode}_v1.json',dict(status='COMPLETE66_SCORE0',rows=len(agg),manifest=manifest,aggregate_sha=sha(dest/'oof.csv'),registration_sha=sha(H/'registration_v1.json')));print(mode,'ALL66_COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:refs,r=prep();save(H/'preparation_v1.json',r);print('PREPARATION_PASS',flush=True)
    else:actual()
