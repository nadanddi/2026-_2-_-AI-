"""Post-hoc mechanism diagnostic: exact training-day forest weights, one known hard fold.
No candidate, target reweighting, hidden labels, EL1, test, or tuned selection.
"""
from pathlib import Path
import sys,json,math,gc
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import pandas as pd,numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
lab,core,wv,folds,outer=S.loadec()
v,k,tm,vm=next(x for x in folds if x[0]=='DIAG10' and x[1]==5)
tr,va=S.seasonal(lab[tm],lab[vm],wv);tr=tr.reset_index(drop=True);va=va.reset_index(drop=True)
cols=[c for c in core.FULL if c!='day']+['season']
model=core.et(7);model.steps[-1][1].n_jobs=2
with threadpool_limits(limits=2):raw=core.predict_model(model,tr,va,cols)
cache=pd.read_csv(ROOT/'집/클로드/research/local/ec3_DI1_all.csv',float_precision='round_trip')
ref=cache[(cache.validator==v)&(cache.validation_fold==k)].set_index('row_id').etS_7.reindex(va.row_id).to_numpy()
assert np.max(np.abs(core.shrink(raw,va)-ref))<1e-9
ztr=model.steps[0][1].transform(tr[cols]);zva=model.steps[0][1].transform(va[cols])
codes,keys=pd.factorize(pd.MultiIndex.from_frame(tr[['farm','day']]))
daily_y=tr.groupby(['farm','day']).sub_ec.mean().reindex(keys).to_numpy()
high=daily_y>=1
W=np.zeros((len(va),len(keys)));hi_y_sum=np.zeros(len(va));lo_y_sum=np.zeros(len(va));checks=0
for tree in model.steps[-1][1].estimators_:
 lt,lq=tree.apply(ztr),tree.apply(zva);count=np.bincount(lt,minlength=tree.tree_.node_count)
 A=csr_matrix((1/count[lt],(lt,codes)),shape=(tree.tree_.node_count,len(keys)))
 W+=A[lq,:].toarray()/len(model.steps[-1][1].estimators_)
 hy=np.bincount(lt,weights=tr.sub_ec.to_numpy()*high[codes],minlength=len(count))
 ly=np.bincount(lt,weights=tr.sub_ec.to_numpy()*(~high[codes]),minlength=len(count))
 hi_y_sum+=hy[lq]/count[lq]/len(model.steps[-1][1].estimators_)
 lo_y_sum+=ly[lq]/count[lq]/len(model.steps[-1][1].estimators_)
 checks+=1
assert np.max(np.abs(W.sum(1)-1))<1e-12
assert np.max(np.abs(hi_y_sum+lo_y_sum-raw))<1e-12
assert (W>=0).all()
# The forest selects training rows, not a full-day label. Daily_y is used only to label its sources.
p=outer[(outer.validator==v)&(outer.validation_fold==k)&(outer.seed==7)].set_index('row_id').season_v2.reindex(va.row_id).to_numpy()
rows=[];sources=[]
for (farm,day),ix in va.groupby(['farm','day']).indices.items():
 ix=np.asarray(ix);ix=ix[np.argsort(va.iloc[ix].hour.to_numpy())]
 assert len(ix)==24
 for hour in [0,6,12,23]:
  prefix=ix[:hour+1]
  w=.5*W[ix[hour]]+.5*W[prefix].mean(0)
  hi_sum=.5*hi_y_sum[ix[hour]]+.5*hi_y_sum[prefix].mean()
  lo_sum=.5*lo_y_sum[ix[hour]]+.5*lo_y_sum[prefix].mean()
  mass=float(w[high].sum());pred=float(core.shrink(raw,va)[ix[hour]])
  assert abs(hi_sum+lo_sum-pred)<1e-12
  rows.append(dict(farm=farm,day=int(day),hour=hour,actual_day_mean=float(va.iloc[ix].sub_ec.mean()),actual_current=float(va.iloc[ix[hour]].sub_ec),actual_v2_current=float(p[ix[hour]]),et_shrunk=pred,high_training_day_mass=mass,selected_high_row_mean=float(hi_sum/mass) if mass>0 else None,selected_other_row_mean=float(lo_sum/(1-mass)) if mass<1 else None,effective_source_days=float(1/(w*w).sum()),contributor_day_mean=float(w@daily_y),available_high_days=int(high.sum()),train_day_max=float(daily_y.max())))
  for j in np.argsort(w)[-5:][::-1]:
   ff,dd=keys[j];sources.append(dict(query_farm=farm,query_day=int(day),hour=hour,source_farm=ff,source_day=int(dd),weight=float(w[j]),source_day_ec=float(daily_y[j])))
# Independent manual reconstruction of representative hard-day current/previous weighted contributions.
case=va[(va.farm=='F47')&(va.day==132)].sort_values('hour')
for r in case.iloc[[0,6,12,23]].itertuples():
 i=va.index[va.row_id==r.row_id][0];prefix=case[case.hour<=r.hour].index.to_numpy()
 manual=.5*float(raw[i])+.5*math.fsum(float(raw[j]) for j in prefix)/len(prefix)
 assert abs(manual-ref[i])<1e-12;checks+=1
pd.DataFrame(rows).to_csv(H/'query_prefix_sources_v1.csv',index=False)
pd.DataFrame(sources).to_csv(H/'top_source_days_v1.csv',index=False)
(H/'routing_verification_v1.json').write_text(json.dumps(dict(status='PASS',seed=7,validator=v,fold=k,n_train_rows=len(tr),n_train_days=len(keys),n_valid_days=len(va[['farm','day']].drop_duplicates()),checks=checks,weight_sum_maxdiff=float(np.max(np.abs(W.sum(1)-1))),prediction_reconstruction_maxdiff=float(np.max(np.abs(hi_y_sum+lo_y_sum-raw))),cached_et_maxdiff=float(np.max(np.abs(core.shrink(raw,va)-ref))),scope='Single preselected known hard fold only; post-hoc source labels, no causal proof or candidate.',source_sha256=S.sha(Path(__file__))),indent=2),encoding='utf-8')
q=pd.DataFrame(rows);print(q[(q.farm=='F47')&(q.day==132)].to_string(index=False));print('ALL_DIAGNOSTICS_COMPLETE')
