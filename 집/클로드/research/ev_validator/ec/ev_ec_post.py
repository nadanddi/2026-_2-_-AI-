# -*- coding: utf-8 -*-
"""EV-EC scoring (2026-10-12 집 클로드 하위 에이전트).  PLAN FIXED BEFORE RUNNING.
Chain per EV fold, per season arm (BASE / SE1), per seed config (7, 101, 2024 = that seed's R3 + the 4 TabPFN contexts;
'avg' = R3 averaged over the 3 seeds = the package configuration):
  R3   : clip(model.shrink(.8 * (.6 ET + .3 LGB + .1 MLP) + .2 * mean(TabPFN)), fold-training EC range)   [package predict_all]
  +SG2 : sg2post.correct with references = fold-training labelled records only (ec / ref / ref_calendar), prepare() on
         train_X + test_X inputs WITHOUT the fold's 'del' records (they have no inputs in evaluation); clip.
  +EA2 : ea2_post formula pred + (A - running mean of +SG2 over hours 0..h), A = anchor_ec23 from
         local/linker_v7/cv_all400_choice_v7m.csv (mirror-chunk linker CV, same chunking) for val records with p >= .8,
         prev_is_labelled, AND prev not a val/del record of this EV fold (else no anchor; count reported); clip.
Scores: RMSE on pass-2 val rows (record >= 179) and on all val rows, overall and per farm.
Test: record-level paired bootstrap (B = 4000, pass-2 val records pooled over folds) P(SE1 RMSE >= BASE RMSE), per seed
config, primary = +EA2 chain (also shown for R3, +SG2).  Reading (same as SE1 rule): SE1 better in every seed config AND
P < .025 in every seed config.
"""
from ev_ec_common import *  # noqa
import time
import sg2post

t0 = time.time()
x, tr = load_train()
cols = ["row_id"] + W + M.RAW
XALL = pd.concat([x[cols], pd.read_csv(DATA / "test_X.csv", usecols=cols)], ignore_index=True)
XALL = XALL[XALL.row_id.str[:3].isin(["F13", "F47"])]; XALL["rec"] = XALL.row_id.str[:7]
VEC = vectors(x)
MB = pd.concat([pd.read_csv(OUT / "members_r3.csv"), pd.read_csv(OUT / "members_pfn.csv")], ignore_index=True)
CH = pd.read_csv(r"C:\work\farmai\집\클로드\research\local\linker_v7\cv_all400_choice_v7m.csv", encoding="utf-8-sig")
CH = CH[(CH.p >= .8) & CH.prev_is_labelled.astype(bool)]
SEEDC = ["7", "101", "2024", "avg"]
res = []; anc_log = []
for k in range(8):
    trn, vb, vs, val, dele = fold_data(k, x, tr, VEC)
    lo, hi = float(trn.sub_ec.min()), float(trn.sub_ec.max())
    ec = trn.groupby(["farm", "day"]).sub_ec.mean(); ref = set(ec.index)
    S = sg2post.prepare(XALL[~XALL.rec.isin(dele)].drop(columns="rec")); cal = sg2post.ref_calendar(S, ref)
    ch = CH[CH.rec.isin(val)]; ok = ~ch.prev.isin(val | dele)
    anc_log.append(dict(fold=k, anchors_v7m=len(ch), dropped_prev_hidden=int((~ok).sum())))
    AN = dict(zip(ch.rec[ok], ch.anchor_ec23[ok]))
    m = MB[MB.fold == k]
    for arm, fr in (("BASE", vb), ("SE1", vs)):
        P = m[m.arm == arm].pivot_table(index="row_id", columns="member", values="pred").loc[fr.row_id]
        pfn = P[["pfn_%d" % c for c in (1, 2, 3, 4)]].mean(1).values
        r3s = {str(s): .6 * P["et_%d" % s].values + .3 * P["lgb_%d" % s].values + .1 * P["mlp_%d" % s].values for s in M.SEEDS}
        r3s["avg"] = np.mean([r3s[str(s)] for s in M.SEEDS], axis=0)
        for sc in SEEDC:
            p0 = np.clip(M.shrink(.8 * r3s[sc] + .2 * pfn, fr), lo, hi)
            p1 = np.clip(sg2post.correct(fr[["row_id"]], p0, S, ec, ref, cal), lo, hi)
            G = pd.DataFrame(dict(row_id=fr.row_id.values, rec=fr.rec.values, hour=fr.hour.values, p1=p1))
            cm = G.groupby("rec").p1.transform(lambda s: s.expanding().mean()).values   # fr sorted by farm, day, hour
            a = G.rec.map(AN).values.astype(float)
            p2 = np.clip(np.where(np.isfinite(a), p1 + (a - cm), p1), lo, hi)
            res.append(pd.DataFrame(dict(fold=k, arm=arm, seed=sc, row_id=fr.row_id.values, rec=fr.rec.values, farm=fr.farm.values,
                                         day=fr.day.values, hour=fr.hour.values, season=fr.season.values, y=fr.sub_ec.values,
                                         anchored=np.isfinite(a), R3=p0, SG2=p1, EA2=p2)))
    print("fold %d done (%d val recs, anchors %d) %.0fs" % (k, len(val), len(AN), time.time() - t0), flush=True)
O = pd.concat(res, ignore_index=True); O.to_csv(OUT / "ev_ec_rows.csv", index=False)
pd.DataFrame(anc_log).to_csv(OUT / "ev_ec_anchor_log.csv", index=False); print(pd.DataFrame(anc_log).to_string(index=False))

r = lambda e: float(np.sqrt(np.mean(np.square(e))))
tab = []
for (sc, arm), g in O.groupby(["seed", "arm"]):
    for scope, gg in (("p2", g[g.day >= 179]), ("all", g)):
        for farm, g3 in (("both", gg), ("F13", gg[gg.farm == "F13"]), ("F47", gg[gg.farm == "F47"])):
            tab.append(dict(seed=sc, arm=arm, scope=scope, farm=farm, n_rec=g3.rec.nunique(), **{c: r(g3[c] - g3.y) for c in ("R3", "SG2", "EA2")}))
T = pd.DataFrame(tab); T.to_csv(OUT / "ev_ec_summary.csv", index=False)
pd.set_option("display.width", 250); print(T.round(4).to_string(index=False))
rng = np.random.default_rng(20261012); bt = []
for sc in SEEDC:
    g = O[(O.seed == sc) & (O.day >= 179)]
    for c in ("R3", "SG2", "EA2"):
        Pv = g.pivot_table(index="row_id", columns="arm", values=c); Y = g[g.arm == "BASE"].set_index("row_id").y.loc[Pv.index]
        E = pd.DataFrame({"rec": Pv.index.str[:7], "b": (Pv.BASE - Y) ** 2, "s": (Pv.SE1 - Y) ** 2}).groupby("rec").agg(b=("b", "sum"), s=("s", "sum"), n=("b", "size"))
        idx = rng.integers(0, len(E), size=(4000, len(E)))
        bb, ss, nn = E.b.values[idx].sum(1), E.s.values[idx].sum(1), E.n.values[idx].sum(1)
        rb, rs = np.sqrt(E.b.sum() / E.n.sum()), np.sqrt(E.s.sum() / E.n.sum())
        bt.append(dict(seed=sc, chain=c, n_rec=len(E), BASE=rb, SE1=rs, pct=100 * (rs / rb - 1), P_worse=float(np.mean(np.sqrt(ss / nn) >= np.sqrt(bb / nn))),
                       recs_changed=int((np.abs(E.b - E.s) > 1e-12).sum())))
B = pd.DataFrame(bt); B.to_csv(OUT / "ev_ec_bootstrap.csv", index=False); print(B.round(4).to_string(index=False))
e = B[(B.chain == "EA2") & (B.seed != "avg")]
print("reading (EA2 chain, seeds 7/101/2024): SE1 better all = %s ; P < .025 all = %s" % (bool((e.SE1 < e.BASE).all()), bool((e.P_worse < .025).all())))
print("total %.0fs" % (time.time() - t0))
