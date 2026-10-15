"""No-fit causal and fold isolation audit."""
from runtime_v1 import *
from collections import Counter
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
lab,test,ct,phc,_=V.build_frames(loader)
cols=list(dict.fromkeys(ct+phc+['cx__'+c for c in FEATURE_COLUMNS]))
w=TF.row_weights(lab,.2,w_noisy=.2)
checks=[]
def log(name,cells):checks.append({'name':name,'cells':cells});print(name,cells,flush=True)

def fullcf(raw,smooth=False):
    tx,_,sx=loader(raw)
    if smooth:tx=median3(tx);sx=median3(sx)
    return build_features(tx,sx).set_index('row_id')

basecf=fullcf(RAW,True)
for farm in ['F13','F47']:
    for cut in [5,17]:
        raw=loader();sx=raw[2];pivot=sorted(sx.loc[sx.farm==farm,'day'].unique())[len(sx.loc[sx.farm==farm,'day'].unique())//2];T=24*pivot+cut;hit=(sx.farm==farm)&(sx.t>T)
        sx.loc[hit,common.USABLE]=sx.loc[hit,common.USABLE]*1.7+123
        l2,t2,_,_,_=V.build_frames(lambda:loader(raw))
        log('MASK_train_future_'+farm+'_'+str(cut),eq(lab,l2,cols))
        take=(test.farm==farm)&(test.t<=T)
        log('query_future_'+farm+'_'+str(cut),eq(test[take],t2[take],cols))
        cf2=fullcf(raw,True)
        ids=test.loc[take,'row_id']
        log('MED3_future_'+farm+'_'+str(cut),eq(basecf.loc[ids],cf2.loc[ids],FEATURE_COLUMNS))
    raw=loader();sx=raw[2];hit=sx.farm!=farm
    sx.loc[hit,common.USABLE]=sx.loc[hit,common.USABLE]*1.9-321
    l2,t2,_,_,_=V.build_frames(lambda:loader(raw))
    log('MASK_train_other_'+farm,eq(lab,l2,cols))
    take=test.farm==farm
    log('query_other_'+farm,eq(test[take],t2[take],cols))
    ids=test.loc[take,'row_id'];cf2=fullcf(raw,True)
    log('MED3_other_'+farm,eq(basecf.loc[ids],cf2.loc[ids],FEATURE_COLUMNS))
raw=loader();raw[2].loc[:,common.USABLE]=7777
l2,_,_,_,_=V.build_frames(lambda:loader(raw))
log('MASK_all_test',eq(lab,l2,cols))
old=common.load_raw;common.load_raw=lambda:loader(raw)
try:w2=TF.row_weights(l2,.2,w_noisy=.2)
finally:common.load_raw=old
assert np.array_equal(w,w2);log('weights_test_all',len(w))
cf=cframes();med=cframes(smooth=True)
notco2=[c for c in FEATURE_COLUMNS if c not in CO2]
log('MED3_only_five_columns',eq(cf,med,notco2))
sh=tuple(d.sample(frac=1,random_state=9109).reset_index(drop=True) for d in loader())
log('MED3_shuffle',eq(fullcf(RAW,True),fullcf(sh,True).loc[basecf.index],FEATURE_COLUMNS))
trm,vam=common.split_mask(lab,diag_folds(lab)[0]);tr=lab[trm];wa=weights(tr)
# Independently mutate every non-training input and ensure fold-local weights unchanged.
tx,ty,sx=loader();tx.loc[~tx.row_id.isin(tr.row_id),common.USABLE]=8888
localtx=tx[tx.row_id.isin(tr.row_id)].copy()
common.load_raw=lambda:(localtx,ty,sx)
try:wb=TF.row_weights(tr,.2,w_noisy=.2)
finally:common.load_raw=loader
assert np.array_equal(wa,wb);log('fold_weights_excluded_input',len(wa))
assert all(sha(CODE.parent/n)==v for n,v in P['temp_package_files'].items())
report={'status':'PASS_FEATURE_AUDIT_ONLY','checks':checks,'cells':sum(c['cells'] for c in checks),'weight_counts':{str(k):int(v) for k,v in Counter(w).items()},'train_rows':len(lab),'query_rows':len(test),'features':cols,'co2_columns':CO2,'ec_numeric_labels_read':False,'model_prediction_full_causality_test':False,'new_fit':0,'package_unchanged':True}
save('feature_audit_v2.json',report)
print('AUDIT_PASS',report['cells'],report['weight_counts'],flush=True)
