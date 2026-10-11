# -*- coding: utf-8 -*-
"""SE1 on the 60 evaluation records with the REAL submitted EC chain (descriptive only, no submission file).
Fixed before running (same SE1 rule as se1_cv.py): evaluation row (farm, record d >= 179, hour h): if h >= 5 and any
TRAINING record (pass-1, season = record day; pass-2 training, season = the submitted mapping's training value) has
outdoor z-RMSE <= .05 over hours 0..h against the row's own outdoor hours 0..h (mu/sd = season.py training pass-1),
season = mean of their seasons; else the submitted interp season.
Chain = package EC/model.py members (R3: ET600 / LGB / MLP x seeds 7,101,2024; TabPFN v2 contexts 1-4), fitted ONCE and
used to predict both the BASE query (submitted seasons) and the SE1 query -> shrink + clip -> SG2 (sg2post, all 400
labelled days as reference, as model.py) -> EA2 linker v7 anchors (as EC/ea2_post.py).  BASE must reproduce the
submission_22 EC column; reported: per-record change SE1 - BASE (and vs submission_22).
Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u se1_season_twin/se1_eval.py
"""
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import env_extra  # noqa: F401
import os, time, gc
from pathlib import Path
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC"
sys.path.insert(0, PK)
for k in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TABPFN_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_TELEMETRY']: os.environ[k] = '1'
import numpy as np, pandas as pd, torch
import model as M
import sg2post
from season import vectors, W, identify
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from threadpoolctl import threadpool_limits

OUT = Path(r"C:\work\farmai\집\클로드\research\se1_season_twin")
DATA = Path(PK) / "data"; CKPT = Path(PK) / "tabpfn-v2-regressor.ckpt"
SUB22 = os.path.join(env.ROOT, u"제출", u"미제출", u"submission_22_측정감쇠앵커_v7EC앵커_가설검정", "submission_22.csv")
ANCH = os.path.join(env.ROOT, u"집", u"클로드", "research", "local", "linker_v7", "eval60_confident_1step_anchor_v7.csv")


def se1_season(x, tx, q, ts, notes):
    vec = vectors(pd.concat([x[["row_id"] + W], tx[["row_id"] + W]], ignore_index=True))
    mu, sd = np.array(notes["standardization"]["mu"]), np.array(notes["standardization"]["sd"])
    refs = sorted(ts); rs = np.array([ts[k] for k in refs])
    R = np.stack([(vec[k] - mu[:, None]) / sd[:, None] for k in refs])
    tab = {}
    for (f, d) in set(zip(q.farm, q.day)):
        qz = (vec[(f, int(d))] - mu[:, None]) / sd[:, None]; s = np.full(24, np.nan); n = np.zeros(24, int)
        for h in range(5, 24):
            dist = np.sqrt(np.nanmean((R[:, :, :h + 1] - qz[None, :, :h + 1]) ** 2, axis=(1, 2))); sel = dist <= .05
            if sel.any(): s[h] = rs[sel].mean(); n[h] = sel.sum()
        tab[(f, int(d))] = (s, n)
    s = np.array([tab[(f, int(d))][0][h] for f, d, h in zip(q.farm, q.day, q.hour)])
    return np.where(np.isfinite(s), s, q.season.values), tab


def members(tr, Qs):
    r3 = {k: [] for k in Qs}; pfn = {k: [] for k in Qs}
    for seed in M.SEEDS:
        preds = {k: [] for k in Qs}
        for name, mdl, cols in [('et', M.et(seed), M.FULL_R3), ('lgb', M.lg(seed), M.BASE_R3), ('mlp', M.mlp(seed), M.BASE_R3)]:
            with threadpool_limits(limits=2):
                mdl.fit(tr[cols], tr.sub_ec.to_numpy(float))
                if name == 'et': mdl.steps[-1][1].n_jobs = 1
                for k, q in Qs.items(): preds[k].append(np.asarray(mdl.predict(q[cols]), float))
            del mdl; gc.collect()
        for k in Qs: r3[k].append(.6 * preds[k][0] + .3 * preds[k][1] + .1 * preds[k][2])
        print('R3 seed%d done' % seed, flush=True)
    X = tr[M.FULL].to_numpy(np.float32); y = tr.sub_ec.to_numpy(float)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    for seed in [1, 2, 3, 4]:
        t = time.monotonic(); ix = np.random.default_rng(seed).choice(len(tr), size=2000, replace=False)
        mdl = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', model_path=str(CKPT), n_estimators=4, random_state=seed,
                                                         ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
        with threadpool_limits(limits=4):
            mdl.fit(X[ix], y[ix])
            for k, q in Qs.items(): pfn[k].append(np.asarray(mdl.predict(q[M.FULL].to_numpy(np.float32)), float))
        del mdl; gc.collect(); print('TabPFN context%d %.0fs' % (seed, time.monotonic() - t), flush=True)
    return {k: (np.mean(r3[k], 0), np.mean(pfn[k], 0)) for k in Qs}


def main():
    x, tx, tr, q, ids, ts, table, notes = M.load(DATA)
    q["day"] = q.row_id.str[4:7].astype(int)
    s1, tab = se1_season(x, tx, q, ts, notes)
    q1 = q.copy(); q1["season"] = s1
    ch = pd.DataFrame({"row_id": q.row_id, "base": q.season.values, "se1": s1})
    ch["rec"] = ch.row_id.str[:7]
    print("SE1 season: rows changed %d / %d, records with any change %d" % (int((ch.base != ch.se1).sum()), len(ch), ch[ch.base != ch.se1].rec.nunique()), flush=True)
    raw = members(tr, {"BASE": q, "SE1": q1})
    z = np.load(Path(PK) / "output_14" / "prediction_details.npz")
    assert (z["row_id"] == q.row_id.values).all()
    print("raw reproduction vs output_14: R3 max|d| %.2e, PFN max|d| %.2e" % (
        np.abs(raw["BASE"][0] - z["raw_r3_by_seed"].mean(0)).max(), np.abs(raw["BASE"][1] - z["raw_pfn_by_context"].mean(0)).max()), flush=True)
    y = pd.read_csv(DATA / 'train_y.csv', usecols=['row_id', 'sub_ec']); yi = identify(y)
    ec = yi.groupby(['farm', 'day']).sub_ec.mean(); ref = set(ec.index); lo, hi = float(y.sub_ec.min()), float(y.sub_ec.max())
    S = sg2post.prepare(pd.concat([x, tx], ignore_index=True)); cal = sg2post.ref_calendar(S, ref)
    A = pd.read_csv(ANCH, encoding="utf-8-sig"); AN = dict(zip(A.row, A.anchor_ec23))
    res = {}
    for k, (r3, pf) in raw.items():
        p13 = np.clip(M.shrink(.8 * r3 + .2 * pf, q), lo, hi)
        p14 = np.clip(sg2post.correct(q[['row_id']], p13, S, ec, ref, cal), lo, hi)
        F = pd.DataFrame({"row_id": q.row_id.values, "farm": q.farm.values, "day": q.day.values, "hour": q.hour.values, "ec14": p14, "p13": p13})
        F = F.sort_values(["farm", "day", "hour"]); F["rec"] = F.row_id.str[:7]
        cm = F.groupby("rec").ec14.transform(lambda s: s.expanding().mean()); a = F.rec.map(AN)
        F["ec22"] = np.clip(np.where(a.notna(), F.ec14 + (a - cm), F.ec14), lo, hi)
        res[k] = F.set_index("row_id")
    s22 = pd.read_csv(SUB22).set_index("row_id").sub_ec
    B, E = res["BASE"], res["SE1"]
    print("BASE chain vs submission_22 EC: max|d| %.2e (6-decimal file)" % float((B.ec22 - s22.loc[B.index]).abs().max()), flush=True)
    D = pd.DataFrame({"rec": B.rec, "base": B.ec22, "se1": E.ec22.loc[B.index], "sub22": s22.loc[B.index],
                      "p13_base": B.p13, "p13_se1": E.p13.loc[B.index]}).join(ch.set_index("row_id")[["base", "se1"]].rename(columns={"base": "season_base", "se1": "season_se1"}))
    D.to_csv(OUT / "se1_eval_rows.csv")
    G = D.groupby("rec").agg(season_base=("season_base", "mean"), season_se1_h23=("season_se1", "last"),
                             rows_changed=("season_se1", lambda s: 0), base=("base", "mean"), se1=("se1", "mean"), sub22=("sub22", "mean"))
    G["rows_changed"] = D.assign(c=D.season_base != D.season_se1).groupby("rec").c.sum()
    G["d_rec"] = G.se1 - G.base; G["d_vs_sub22"] = G.se1 - G.sub22
    G["anchored"] = G.index.isin(list(AN))
    G.to_csv(OUT / "se1_eval_records.csv"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(G.round(4).to_string())
    d = D.se1 - D.base
    print("SE1 - BASE (final EC chain): row RMS %.4f, mean %+.4f, max|d| %.4f; records |mean d|>.02: %d; anchored records' RMS %.4f, unanchored %.4f" % (
        np.sqrt((d ** 2).mean()), d.mean(), d.abs().max(), int((G.d_rec.abs() > .02).sum()),
        np.sqrt((d[D.rec.isin(list(AN))] ** 2).mean()), np.sqrt((d[~D.rec.isin(list(AN))] ** 2).mean())))
    d13 = D.p13_se1 - D.p13_base
    print("SE1 - BASE before SG2/EA2 (p13): row RMS %.4f, mean %+.4f" % (np.sqrt((d13 ** 2).mean()), d13.mean()))


if __name__ == "__main__":
    main()
