from pathlib import Path
import importlib.util,json
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('conditional_runner',H/'run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
r.common.load_raw=r.TM.masked_loader
try:lab,ct,phc=r.TM.build_world()
finally:r.common.load_raw=r.TM.ORIG;r.harness._CACHE.clear()
tx,_,sx=r.TM.masked_loader();cf=r.build_features(tx,sx).set_index('row_id')
for c in r.FEATURE_COLUMNS:
    if c not in lab:lab[c]=cf.loc[lab.row_id,c].to_numpy()
w=r.TF.row_weights(lab,.2,w_noisy=.2);tm,vm=r.common.split_mask(lab,r.diag_folds(lab)[0]);pred,stats=r.fit_fold(lab[tm].copy(),lab[vm].copy(),w[tm],726,'DIAG10',ct,phc)
saved=dict(r.np.load(r.OUT/'DIAG10_0.npz',allow_pickle=True));diff={tag:float(r.np.max(r.np.abs(pred[tag]-saved[tag+'_726']))) for tag in ['G1','w_BASE','w_CODEX','w_PFN','cal_BASE','cal_CODEX','cal_design']};assert max(diff.values())<1e-10
answer=dict(status='PASS',fold=0,seed=726,maxdiff=diff,scope='Raw MASK features rebuilt, first outer fold with all three inner BASE/CODEX fits replayed; not all outer folds refit')
(H/'replay_verification.json').write_text(json.dumps(answer,indent=2),encoding='utf-8');print(json.dumps(answer))
