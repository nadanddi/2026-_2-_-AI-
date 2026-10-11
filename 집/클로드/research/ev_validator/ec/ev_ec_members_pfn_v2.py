# -*- coding: utf-8 -*-
"""Runtime fallback for the TabPFN member (machine shared with another job; ~600 s per context under contention).
Identical TabPFN settings to ev_ec_members.py, but loop is CONTEXT-MAJOR (context 1 for all my folds, then 2, ...) and
every (context, fold) prediction is saved at once to out_v1/pfn_v2/c<c>_f<k>.csv, so that if the time budget ends,
ev_ec_post_v2.py can use the contexts finished for ALL 8 folds (stated as an approximation in the report).
Usage: ev_ec_members_pfn_v2.py <fold,fold,...>"""
from ev_ec_common import *  # noqa
import gc, time, torch
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

folds = [int(v) for v in sys.argv[1].split(",")]
CK = Path(PK) / "tabpfn-v2-regressor.ckpt"; D = OUT / "pfn_v2"
x, tr = load_train(); VEC = vectors(x); t0 = time.time()
torch.set_num_threads(4); torch.set_num_interop_threads(1)
FD = {k: fold_data(k, x, tr, VEC) for k in folds}
for c in (1, 2, 3, 4):
    for k in folds:
        f = D / ("c%d_f%d.csv" % (c, k))
        if f.exists(): continue
        trn, vb, vs, val, dele = FD[k]
        Q = pd.concat([vb.assign(arm="BASE"), vs.assign(arm="SE1")], ignore_index=True); y = trn.sub_ec.to_numpy(float)
        X = trn[M.FULL].to_numpy(np.float32); QQ = Q[M.FULL].to_numpy(np.float32)
        ix = np.random.default_rng(c).choice(len(trn), size=2000, replace=False)
        mdl = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", model_path=str(CK), n_estimators=4, random_state=c,
                                                         ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            mdl.fit(X[ix], y[ix]); p = np.asarray(mdl.predict(QQ), float)
        pd.DataFrame(dict(fold=k, row_id=Q.row_id.values, arm=Q.arm.values, member="pfn_%d" % c, pred=p)).to_csv(f, index=False)
        del mdl; gc.collect(); print("context %d fold %d done %.0fs" % (c, k, time.time() - t0), flush=True)
print("DONE %.0fs" % (time.time() - t0))
