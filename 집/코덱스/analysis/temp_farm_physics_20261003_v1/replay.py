from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experiment',HERE/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.common.load_raw=m.TM.masked_loader
try:lab,ct,phc=m.TM.build_world()
finally:m.common.load_raw=m.TM.ORIG;m.harness._CACHE.clear()
tx,_,sx=m.TM.masked_loader();cf=m.build_features(tx,sx).set_index('row_id')
for c in m.FEATURE_COLUMNS:
    if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
w=m.TF.row_weights(lab,.2,w_noisy=.2);fd=m.diag_folds(lab)[0];tm,vm=m.common.split_mask(lab,fd)
pred,stats=m.fit_fold(lab[tm].copy(),lab[vm].copy(),w[tm],726,False)
saved=dict(m.np.load(m.OUT/'DIAG10_0.npz',allow_pickle=True));diff={k:float(m.np.max(m.np.abs(pred[k]-saved[k+'_726']))) for k in ['F1','physics']}
assert max(diff.values())<1e-10
result=dict(status='PASS',fold=0,seed=726,maxdiff=diff,training=stats,scope='Raw MASK features rebuilt and first original DIAG fold refit; not all folds replayed')
(HERE/'replay_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
