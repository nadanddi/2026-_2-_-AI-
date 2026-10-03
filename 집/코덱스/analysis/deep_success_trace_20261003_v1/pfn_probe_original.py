from pfn_attribution import *
def main():
 world=joblib.load(O/'world.joblib');lab=world['lab'];tx,_,sx=S.safeload();cf=S.build_features(tx,sx).set_index('row_id');native=lab.copy();native[S.FEATURE_COLUMNS]=cf.loc[lab.row_id,S.FEATURE_COLUMNS].to_numpy()
 diff=[]
 for c in S.FEATURE_COLUMNS:
  aa=lab[c].to_numpy(float);bb=native[c].to_numpy(float);delta=float(np.nanmax(np.abs(aa-bb)))
  if delta>0:diff.append(dict(column=c,maxdiff=delta))
 print('PFN_FEATURE_DIFFERENCES',diff,flush=True)
 fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==3);tm,vm=S.common.split_mask(lab,fd);tr,va=native[tm],native[vm];stored=np.load(R/'집/코덱스/local/temp_tk_season_20261003_v1/pfn_DIAG10_3_1.npz');idx=pd.Index(tr.row_id).get_indexer(stored['context_row_id']);torch.set_num_threads(4)
 model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=1,ignore_pretraining_limits=True,inference_precision=torch.float32)
 model.fit(tr[S.FEATURE_COLUMNS].to_numpy(np.float32)[idx],tr.sub_temp.to_numpy()[idx]);pred=model.predict(va[S.FEATURE_COLUMNS].to_numpy(np.float32));assert np.array_equal(va.row_id.to_numpy(dtype=str),stored['row_id']);delta=float(np.max(np.abs(pred-stored['base'])));print('FULL_BASE',delta,flush=True)
 sub=va[(va.farm=='F47')&(va.day==72)];pp=model.predict(sub[S.FEATURE_COLUMNS].to_numpy(np.float32));mp=dict(zip(va.row_id,pred));sd=float(np.max(np.abs(pp-[mp[i] for i in sub.row_id])));print('SUBSET_BATCH_DIFF',sd,flush=True)
 savej(H/'pfn_probe.json',dict(feature_differences=diff,full_base_difference=delta,subset_batch_difference=sd))
if __name__=='__main__':main()
