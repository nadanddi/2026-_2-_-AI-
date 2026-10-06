"""Full saved-tree audit; independent routing/support/permutation attribution; no refit."""
from pathlib import Path
import sys,json,csv,math,hashlib,datetime,itertools,gc
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OLD=ROOT/'연구실/코덱스/analysis/ec_resolution_sequence_20261006_v1';sys.path.insert(0,str(OLD));import verify_stage1_v1 as B
np,pd,M=B.np,B.pd,B.M;L=ROOT/'연구실/코덱스/local'/H.name;D=H/'results_v4';U=H/'summary_v1';PAIRS=[('F47',160,161),('F13',98,112)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def same(a,b,tol=1e-10):assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
def closearray(a,b,tol=1e-10):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def avg(v):return math.fsum(map(float,v))/len(v)
def idsmeta(ids):return pd.DataFrame(dict(row_id=ids,farm=[r[:3] for r in ids],day=[int(r[4:7]) for r in ids],hour=[int(r[8:10]) for r in ids]))
def route(X,left,right,feature,threshold,y=None):
    n=len(left);nodes=np.zeros(len(X),np.int32);alive=np.arange(len(X));counts=np.zeros(n,np.int64);sums=np.zeros(n,float)
    for _ in range(n+1):
        if not len(alive):break
        here=nodes[alive];counts+=np.bincount(here,minlength=n)
        if y is not None:sums+=np.bincount(here,weights=y[alive],minlength=n)
        isleaf=left[here]==-1;assert np.all(right[here[isleaf]]==-1);alive=alive[~isleaf]
        if len(alive):
            here=nodes[alive];cols=feature[here];assert np.all((cols>=0)&(cols<X.shape[1]));nxt=np.where(X[alive,cols]<=threshold[here],left[here],right[here]);assert np.all((nxt>=0)&(nxt<n));nodes[alive]=nxt
    else:raise AssertionError('Cyclic or malformed tree')
    return nodes,counts,sums
def weights_from_leaves(train,query):
    w=np.zeros((len(query),len(train)),float);T=train.shape[1]
    for tree in range(T):
        cohorts=defaultdict(list)
        for i,leaf in enumerate(train[:,tree]):cohorts[int(leaf)].append(i)
        for i,leaf in enumerate(query[:,tree]):ix=cohorts[int(leaf)];assert ix;w[i,ix]+=1/(T*len(ix))
    assert np.all(w>=0);closearray(w.sum(axis=1),np.ones(len(query)),1e-12);return w
prep=js(H/'preparation_v4.json');receipt=js(D/'completion.json');assert not (L/'worker.lock').exists();assert prep['source_sha']==receipt['source_sha']==sha(H/'run_v4.py');assert receipt['preparation_sha']==sha(H/'preparation_v4.json');assert prep['core_sha']==sha(H/'leaf_core_v1.py');assert prep['protocol_sha']==sha(H/'PROTOCOL_v1.md');assert prep['original_preparation_sha']==sha(OLD/'preparation_v2.json');assert prep['full_columns']==M.FULL
for name,digest in receipt['files'].items():assert sha(D/name)==digest
for name,digest in js(U/'completion.json')['files'].items():assert sha(U/name)==digest
z=pd.read_csv(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv',float_precision='round_trip');y=z[z.validator=='DIAG10'][['row_id','y']].drop_duplicates();assert len(y)==8640
raw=pd.read_csv(Path(B.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True);f=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',how='inner',validate='one_to_one');vec=M.vectors(raw)
profiles=read(D/'support_profiles.csv');support=read(D/'support_rows.csv');supportdays=read(D/'support_days.csv');changes=read(D/'pair_weight_changes.csv');splits=read(D/'first_split_training.csv');phicsv=read(D/'weight_shapley_rows.csv');smooth=read(U/'smooth_day_support_profiles.csv');smoothdays=read(U/'smooth_support_days.csv')
expected_split={};expected_support={};expected_days={};expected_changes={};expected_phi={};modelchecks=[];smoothchecks=[];phi_checks=[];total_nodes=0
for job,decl in zip(prep['jobs'],receipt['models']):
    name=job['name'];assert decl['name']==name;path=L/(name+'.npz');assert sha(path)==decl['npz_sha']==js(path.with_suffix('.json'))['npz_sha'];assert sha(job['cache_path'])==job['cache_sha']
    with np.load(path,allow_pickle=False) as store:a={k:store[k].copy() for k in store.files}
    tid=a['train_row_id'].astype(str);qid=a['query_row_id'].astype(str);assert not set(tid)&set(qid);t,q=B.transform(B.order(f,tid),B.order(f,qid),vec);assert B.fh(t)==job['train_hash']==decl['train_hash'] and B.fh(q)==job['query_hash']==decl['query_hash'];assert len(t)==job['train_rows'] and len(q)==job['query_rows'];assert np.array_equal(a['train_y'],t.sub_ec.to_numpy()) and np.array_equal(a['query_y'],q.sub_ec.to_numpy());Tmeta,Qmeta=idsmeta(tid),idsmeta(qid)
    xt=t[M.FULL].to_numpy(float);qt=q[M.FULL].to_numpy(float);med=np.nanmedian(xt,axis=0);assert np.isfinite(med).all();closearray(med,a['imputer_median'],1e-12);X=np.where(np.isnan(xt),med,xt).astype(np.float32);Q=np.where(np.isnan(qt),med,qt).astype(np.float32);assert np.array_equal(X,a['train_X']) and np.array_equal(Q,a['query_X']);assert X.shape[1]==38
    offsets=a['offsets'];assert len(offsets)==601 and offsets[0]==0 and np.all(np.diff(offsets)>0);assert offsets[-1]==len(a['value'])==decl['nodes'];total_nodes+=int(offsets[-1]);tl=np.empty_like(a['train_leaf']);ql=np.empty_like(a['query_leaf']);cp={};cv={};treepred=np.zeros((len(q),600));ty=a['train_y'];ym=t.groupby(['farm','day']).sub_ec.transform('mean').to_numpy();daycodes=[(r.farm,int(r.day)) for r in t.itertuples()];pairids={(r.farm,int(r.day),int(r.hour)):i for i,r in enumerate(q.itertuples())}
    if name=='common_7':
        for farm in ['F47','F13']:cp[farm]=np.empty_like(a[f'coalition_{farm}_leaf']);cv[farm]=np.empty((32,600))
    nodeerr=0
    for tree in range(600):
        lo,hi=map(int,offsets[tree:tree+2]);left,right,feature,threshold=[a[c][lo:hi] for c in ['left','right','feature','threshold']];value=a['value'][lo:hi];leaf,cnt,ysum=route(X,left,right,feature,threshold,ty);tl[:,tree]=leaf;assert np.all(cnt>0);assert np.array_equal(cnt,a['n_samples'][lo:hi]);assert np.array_equal(cnt,a['weighted_n_samples'][lo:hi]);err=float(np.max(abs(ysum/cnt-value)));nodeerr=max(nodeerr,err);assert err<1e-10
        qleaf,_,_=route(Q,left,right,feature,threshold);ql[:,tree]=qleaf;treepred[:,tree]=value[qleaf]
        for farm in cp:
            cleaf,_,_=route(a[f'coalition_{farm}_X'],left,right,feature,threshold);cp[farm][:,tree]=cleaf;cv[farm][:,tree]=value[cleaf]
        for farm,good,bad in PAIRS:
            if (farm,good,0) not in pairids or (farm,bad,0) not in pairids:continue
            gi,bi=pairids[farm,good,0],pairids[farm,bad,0];node=0;mask=np.ones(len(t),bool)
            while left[node]!=-1:
                c=feature[node];th=threshold[node];ga=Q[gi,c]<=th;ba=Q[bi,c]<=th
                if ga!=ba:
                    for side,branch in [('good',ga),('bad',ba)]:
                        sel=mask&((X[:,c]<=th)==branch);child=int(left[node] if branch else right[node]);rec=dict(parent=node,child=child,feature=M.FULL[c],threshold=th,good_value=Q[gi,c],bad_value=Q[bi,c],rows=int(sel.sum()),days=len({daycodes[i] for i in np.flatnonzero(sel)}),parent_rows=int(mask.sum()),parent_mean_ec=avg(ty[mask]),parent_high_row_fraction=avg(ty[mask]>=1),parent_high_day_fraction=avg(ym[mask]>=1),parent_days=len({daycodes[i] for i in np.flatnonzero(mask)}),mean_ec=avg(ty[sel]),high_row_fraction=avg(ty[sel]>=1),high_day_fraction=avg(ym[sel]>=1),F13_fraction=avg(t.farm.to_numpy()[sel]=='F13'));expected_split[name,farm,tree,side]=rec
                    break
                mask&=(X[:,c]<=th)==ga;node=int(left[node] if ga else right[node])
    assert np.array_equal(tl,a['train_leaf']) and np.array_equal(ql,a['query_leaf']);w=weights_from_leaves(tl,ql);closearray(w,a['weights'],1e-12);pred=np.array([avg(r) for r in treepred]);wp=np.array([math.fsum(float(ww)*float(yy) for ww,yy in zip(r,ty)) for r in w]);closearray(pred,a['raw_et']);closearray(wp,pred)
    with np.load(job['cache_path'],allow_pickle=False) as old:
        assert old['train_row_id'].astype(str).tolist()==tid.tolist();pos={str(r):i for i,r in enumerate(old['row_id'])};target=old['raw_et'][[pos[r] for r in qid]];replay=float(np.max(abs(pred-target)));assert replay<=1e-10
    for farm,day in q[['farm','day']].drop_duplicates().itertuples(index=False,name=None):
        ix=np.flatnonzero((q.farm==farm)&(q.day==day));ix=ix[np.argsort(q.hour.iloc[ix])];assert q.hour.iloc[ix].tolist()==list(range(24));vectors={'hour0':w[ix[0]],'daymean':w[ix].mean(axis=0)}
        # Prefix smoothing is linear on training-support weights. This is not a changed model.
        sw=np.array([.5*w[ii]+.5*w[ix[:h+1]].mean(axis=0) for h,ii in enumerate(ix)]);sv=sw.mean(axis=0);ss=next(r for r in smooth if r['context']==name and r['farm']==farm and int(r['day'])==day);smvals=dict(prediction=float(sv@ty),true_day=avg(a['query_y'][ix]),high_row_weight=float(sv[ty>=1].sum()),high_day_weight=float(sv[ym>=1].sum()),F13_weight=float(sv[t.farm.eq('F13')].sum()),support_days=len({daycodes[i] for i in np.flatnonzero(sv>0)}))
        for c,v in smvals.items():same(v,ss[c])
        smgroups=defaultdict(lambda:[0.,0.])
        for i in np.flatnonzero(sv>0):g=smgroups[daycodes[i]];g[0]+=sv[i];g[1]+=sv[i]*ty[i]
        subset=[r for r in smoothdays if r['context']==name and r['query_farm']==farm and int(r['query_day'])==day];assert len(subset)==len(smgroups)
        for r in subset:v,c=smgroups[r['farm'],int(r['day'])];same(v,r['weight']);same(c,r['contribution']);same(c/v,r['weighted_ec'])
        smoothchecks.append(dict(context=name,farm=farm,day=int(day),**smvals))
        for level,v in vectors.items():
            pp=next(r for r in profiles if r['context']==name and r['farm']==farm and int(r['day'])==day and r['level']==level);sel=v>0;truth=a['query_y'][ix[0]] if level=='hour0' else avg(a['query_y'][ix]);vals=dict(prediction=float(v@ty),true_ec=truth,bias=float(v@ty-truth),high_row_weight=float(v[ty>=1].sum()),high_day_weight=float(v[ym>=1].sum()),F13_weight=float(v[t.farm.eq('F13')].sum()),F47_weight=float(v[t.farm.eq('F47')].sum()),support_rows=int(sel.sum()),support_days=len({daycodes[i] for i in np.flatnonzero(sel)}),train_global_mean=avg(ty),weighted_co2=float(v@X[:,M.FULL.index('in_co2')]),weighted_heating=float(v@X[:,M.FULL.index('act_heating')]),weighted_co2_h0=float(v@X[:,M.FULL.index('in_co2_h0')]),weighted_heating_h0=float(v@X[:,M.FULL.index('act_heating_h0')]),co2_missing_weight=float(v[t.in_co2.isna()].sum()),heating_missing_weight=float(v[t.act_heating.isna()].sum()),weighted_training_day_ec=float(v@ym))
            for c,value in vals.items():same(value,pp[c])
            sums=defaultdict(lambda:[0.,0.])
            for i in np.flatnonzero(sel):expected_support[name,farm,int(day),level,tid[i]]=(v[i],v[i]*ty[i]);g=sums[daycodes[i]];g[0]+=v[i];g[1]+=v[i]*ty[i]
            for (tf,td),(mass,contrib) in sums.items():expected_days[name,farm,int(day),level,tf,td]=(mass,contrib)
    for farm,good,bad in PAIRS:
        if (farm,good,0) not in pairids or (farm,bad,0) not in pairids:continue
        for level in ['hour0','daymean']:
            wg=w[pairids[farm,good,0]] if level=='hour0' else w[(q.farm==farm)&(q.day==good)].mean(axis=0);wb=w[pairids[farm,bad,0]] if level=='hour0' else w[(q.farm==farm)&(q.day==bad)].mean(axis=0);delta=wb-wg;same(math.fsum(delta),0,1e-12)
            for i in np.flatnonzero(delta!=0):expected_changes[name,farm,level,tid[i]]=(delta[i],delta[i]*ty[i],delta[i]*(ty[i]-ty.mean()))
    if name=='common_7':
        oldphis=read(OLD/'stage1_explanation_v1/grouped_contributions.csv')
        for farm in ['F47','F13']:
            assert np.array_equal(cp[farm],a[f'coalition_{farm}_leaf']);cw=weights_from_leaves(tl,cp[farm]);closearray(cw,a[f'coalition_{farm}_weights'],1e-12);names=a[f'coalition_{farm}_groups'].astype(str).tolist();assert len(names)==5;phi=np.zeros((5,len(t)))
            # Independent permutation form, not the worker's factorial-subset accumulation.
            for perm in itertools.permutations(range(5)):
                mask=0
                for i in perm:nextmask=mask|(1<<i);phi[i]+=(cw[nextmask]-cw[mask])/120;mask=nextmask
            closearray(phi,a[f'coalition_{farm}_phi'],1e-12);closearray(phi.sum(axis=0),cw[-1]-cw[0],1e-12);closearray(phi.sum(axis=1),np.zeros(5),1e-12);predcoal=np.array([avg(r) for r in cv[farm]]);closearray(cw@ty,predcoal)
            for i,n in enumerate(names):
                value=math.fsum(float(x)*float(y) for x,y in zip(phi[i],ty));ref=next(r for r in oldphis if r['model']=='ET' and r['farm']==farm and r['group']==n);same(value,ref['prediction_change']);phi_checks.append(dict(farm=farm,group=n,model_prediction_change=value))
                for j in np.flatnonzero(a[f'coalition_{farm}_phi'][i]!=0):expected_phi[farm,n,tid[j]]=(phi[i,j],phi[i,j]*ty[j],phi[i,j]*(ty[j]-ty.mean()))
    modelchecks.append(dict(name=name,train_rows=len(t),query_rows=len(q),trees=600,nodes=int(offsets[-1]),replay_maxdiff=replay,node_mean_maxdiff=nodeerr,weighted_prediction_maxdiff=float(np.max(abs(wp-pred)))))
    print('INDEPENDENT_MODEL_PASS',name,'nodes',int(offsets[-1]),flush=True);del a,X,Q,tl,ql,w,treepred;gc.collect()
def check_sparse(csvrows,expected,keyfn,fields):
    seen=set()
    for r in csvrows:
        key=keyfn(r);assert key not in seen;seen.add(key);values=expected[key]
        for field,value in zip(fields,values):same(value,r[field])
    assert seen==set(expected),(len(seen),len(expected))
check_sparse(support,expected_support,lambda r:(r['context'],r['query_farm'],int(r['query_day']),r['level'],r['row_id']),['weight','contribution'])
check_sparse(supportdays,{k:(v[0],v[1],v[1]/v[0]) for k,v in expected_days.items()},lambda r:(r['context'],r['query_farm'],int(r['query_day']),r['level'],r['train_farm'],int(r['train_day'])),['weight','contribution','weighted_ec'])
check_sparse(changes,expected_changes,lambda r:(r['context'],r['query_farm'],r['level'],r['row_id']),['delta_weight','raw_contribution','centered_contribution'])
check_sparse(phicsv,expected_phi,lambda r:(r['query_farm'],r['group'],r['row_id']),['phi_weight','raw_contribution','centered_contribution'])
seen=set()
for r in splits:
    key=(r['context'],r['farm'],int(r['tree']),r['side']);assert key not in seen;seen.add(key);expected=expected_split[key]
    for c,v in expected.items():
        if c=='feature':assert v==r[c]
        else:same(v,r[c])
assert seen==set(expected_split)
out=dict(status='PASS_FULL_TREE_SUPPORT_TRACE',new_fit=0,models=modelchecks,query_rows=sum(r['query_rows'] for r in modelchecks),trees=3600,nodes_reconstructed=total_nodes,first_split_child_rows=len(splits),support_rows=len(support),support_day_rows=len(supportdays),pair_delta_rows=len(changes),phi_rows=len(phicsv),phi_checks=phi_checks,smooth_day_profiles=smoothchecks,limitations=['saved fixed ET arithmetic and real training support, not retraining removal effect','raw ET and linearly smoothed support are distinct','no PFN/full-ensemble/physical causality or adoption proof'])
path=H/('verify_trace_v2_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
print('FULL_TRACE_AUDIT_PASS',total_nodes,path,flush=True)
