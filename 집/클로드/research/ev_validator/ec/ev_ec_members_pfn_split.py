# -*- coding: utf-8 -*-
"""Runtime split of ev_ec_members.py 'pfn' part (identical TabPFN settings; only parallelisation over folds and
4 torch threads per process).  Usage: ev_ec_members_pfn_split.py <fold,fold,...> -> out_v1/members_pfn_part_<folds>.csv
Merge: ev_ec_members_pfn_split.py merge -> out_v1/members_pfn.csv (input of ev_ec_post.py)."""
from ev_ec_common import *  # noqa
import gc, time, glob, torch
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

if sys.argv[1] == "merge":
    parts = sorted(glob.glob(str(OUT / "members_pfn_part_*.csv")))
    D = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True); assert sorted(D.fold.unique()) == list(range(8))
    D.to_csv(OUT / "members_pfn.csv", index=False); print("merged", parts, len(D)); sys.exit()
folds = [int(v) for v in sys.argv[1].split(",")]
CK = Path(PK) / "tabpfn-v2-regressor.ckpt"
x, tr = load_train(); VEC = vectors(x); t0 = time.time(); rows = []
torch.set_num_threads(4); torch.set_num_interop_threads(1)
for k in folds:
    trn, vb, vs, val, dele = fold_data(k, x, tr, VEC)
    Q = pd.concat([vb.assign(arm="BASE"), vs.assign(arm="SE1")], ignore_index=True); y = trn.sub_ec.to_numpy(float)
    X = trn[M.FULL].to_numpy(np.float32); QQ = Q[M.FULL].to_numpy(np.float32)
    for c in (1, 2, 3, 4):
        ix = np.random.default_rng(c).choice(len(trn), size=2000, replace=False)
        mdl = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", model_path=str(CK), n_estimators=4, random_state=c,
                                                         ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            mdl.fit(X[ix], y[ix]); p = np.asarray(mdl.predict(QQ), float)
        rows.append(pd.DataFrame(dict(fold=k, row_id=Q.row_id.values, arm=Q.arm.values, member="pfn_%d" % c, pred=p)))
        del mdl; gc.collect(); print("fold %d pfn context %d done %.0fs" % (k, c, time.time() - t0), flush=True)
pd.concat(rows, ignore_index=True).to_csv(OUT / ("members_pfn_part_%s.csv" % "-".join(map(str, folds))), index=False)
print("PART DONE %.0fs" % (time.time() - t0))
