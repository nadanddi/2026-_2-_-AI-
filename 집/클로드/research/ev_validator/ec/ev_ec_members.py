# -*- coding: utf-8 -*-
"""EV-EC members (fixed before running).  Per EV fold: the package's real members, unchanged hyper-parameters
(model.et 600 trees FULL_R3, model.lg tweedie BASE_R3, model.mlp BASE_R3; seeds 7, 101, 2024) and TabPFN v2
(package checkpoint, contexts 1..4 = 2000 random training rows each, n_estimators 4, FULL features), trained on the
fold's training records only, predicting val rows with BASE season and with SE1 season.
Only change vs package: thread counts (n_jobs) for speed.
Usage: ev_ec_members.py r3   |   ev_ec_members.py pfn     -> out_v1/members_<part>.csv (row_id, arm, member, pred)
"""
from ev_ec_common import *  # noqa
import gc, time, torch
from threadpoolctl import threadpool_limits
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

part = sys.argv[1]; assert part in ("r3", "pfn")
CK = Path(PK) / "tabpfn-v2-regressor.ckpt"
x, tr = load_train(); VEC = vectors(x); t0 = time.time(); rows = []
for k in range(8):
    trn, vb, vs, val, dele = fold_data(k, x, tr, VEC)
    Q = pd.concat([vb.assign(arm="BASE"), vs.assign(arm="SE1")], ignore_index=True)
    y = trn.sub_ec.to_numpy(float)
    if part == "r3":
        for seed in M.SEEDS:
            for name, mdl, cols in [("et", M.et(seed), M.FULL_R3), ("lgb", M.lg(seed), M.BASE_R3), ("mlp", M.mlp(seed), M.BASE_R3)]:
                if name == "et": mdl.steps[-1][1].n_jobs = 10
                if name == "lgb": mdl.set_params(n_jobs=10)
                with threadpool_limits(limits=10):
                    mdl.fit(trn[cols], y); p = np.asarray(mdl.predict(Q[cols]), float)
                rows.append(pd.DataFrame(dict(fold=k, row_id=Q.row_id.values, arm=Q.arm.values, member="%s_%d" % (name, seed), pred=p)))
                del mdl; gc.collect()
            print("fold %d seed %d R3 done %.0fs" % (k, seed, time.time() - t0), flush=True)
    else:
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        X = trn[M.FULL].to_numpy(np.float32); QQ = Q[M.FULL].to_numpy(np.float32)
        for c in (1, 2, 3, 4):
            ix = np.random.default_rng(c).choice(len(trn), size=2000, replace=False)
            mdl = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", model_path=str(CK), n_estimators=4, random_state=c,
                                                             ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
            with threadpool_limits(limits=6):
                mdl.fit(X[ix], y[ix]); p = np.asarray(mdl.predict(QQ), float)
            rows.append(pd.DataFrame(dict(fold=k, row_id=Q.row_id.values, arm=Q.arm.values, member="pfn_%d" % c, pred=p)))
            del mdl; gc.collect()
            print("fold %d pfn context %d done %.0fs" % (k, c, time.time() - t0), flush=True)
    pd.concat(rows, ignore_index=True).to_csv(OUT / ("members_%s.csv" % part), index=False)
print("ALL DONE %.0fs" % (time.time() - t0))
