# -*- coding: utf-8 -*-
"""CT2: CT1 curtain-schedule features checked on the SUBMISSION pipeline, with the CT1 critique's conditions.
(2026-10-09 집 클로드, user: "1~4 해봐").  Fixed before running.

Pipeline (as submission_14): final = clip(shrink(0.8 R3 + 0.2 PFN)), then SG2 on pass-2 rows (ec3_SG2 correction,
reference = labelled records outside the fold), clip.  R3 = .6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1.
PFN = TabPFN v2 on FULL (without day) + season, 2000 random training rows, contexts 9-12 (never used), n_est 4,
float32, GPU; SHARED by all configurations (curtain features go into R3 only, as the critique proposed).
Configurations (R3 feature sets):
  REF   current features
  THS5  REF + th_full_hours_td, th_night_closed, th_h9, th_h9_partial, th_sched_score   (CT1 set minus th_full_run)
  THB   REF + th_full_hours_td, th_night_closed, th_h9                                   (broad, no template)
NEW seeds 8383 / 1919 / 7171.  Exclusions same farm +-1, lock-40 +-1, other farm d-3..d+3.
Folds: DIAG10 folds holding pass-2 days, EL1, P2LOO (pass-2 rows scored).
RULE (k = 2, alpha = .0125), candidate X vs REF, final predictions after SG2, pass-2 rows:
  (a) every seed x {DIAG10, EL1, P2LOO} better (9/9);
  (b) P2LOO seed-mean day bootstrap (farm, day blocks, 20,000) share(X not better) < .0125;
  (c) auxiliary (seed mean, P2LOO): still better after removing the NOSIG days in pass 2 (F47 229) AND, separately,
      after removing the curtain-pattern days in pass 2 (pattern = night 0, 10-15 h at 100).
  PASS = all of (a)-(c).  Reported, not judged: removing the top-5 gain days; high vs normal days.
Stage 3 (test shift, item 3): R3 REF/THS5/THB fitted on all 360 non-lock days (same seeds), predicted on test_X rows;
  shift = shrink(0.8 x (R3_X - R3_REF)) (PFN shared cancels; SG2 not applied - limitation).  On the 16 test
  curtain-pattern days: shift distribution, and the worst-case RMSE change if the truth on those days equalled the
  REF prediction (score .1384 used only as a scale for this bound; no tuning).  No submission file is written.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ct2_thermal_on_submission_pipeline_v1.py {1|2|3|sum}
  stage 1 = DIAG10, stage 2 = EL1 + P2LOO, stage 3 = test shift.
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd, torch
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct1", os.path.join(HERE, "ct1_thermal_schedule_features_v1.py"))
ct1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct1)
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
wt0, dp1, dc4, p3, core = ct1.wt0, ct1.dp1, ct1.dc4, ct1.p3, ct1.core
SEEDS = (8383, 1919, 7171)
CTX = (9, 10, 11, 12)
CK = os.path.join(env.LOCAL, "ct2_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
THS5 = ["th_full_hours_td", "th_night_closed", "th_h9", "th_h9_partial", "th_sched_score"]
THB = ["th_full_hours_td", "th_night_closed", "th_h9"]
CONFIGS = ("REF", "THS5", "THB")
NOSIG_P2 = {("F47", 229)}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def shrink(p, fr):
    d = fr[["farm", "day", "hour"]].copy(); d["p"] = np.asarray(p, float)
    d = d.sort_values(["farm", "day", "hour"])
    avg = d.groupby(["farm", "day"]).p.transform(lambda s: s.expanding().mean())
    return pd.Series(.5 * d.p + .5 * avg, index=d.index).reindex(fr.index).values


def pfn(Xc, yc, Xq, seed):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda" if torch.cuda.is_available() else "cpu",
                                                   n_estimators=4, random_state=seed, ignore_pretraining_limits=True,
                                                   inference_precision=torch.float32)
    m.fit(Xc, yc)
    return np.asarray(m.predict(Xq), float)


def prep():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(ct1.thermal_features(full), on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    return raw, full, lab, lock, fds


def feature_sets():
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    return {"REF": (FS, BS), "THS5": (FS + THS5, BS + THS5), "THB": (FS + THB, BS + THB)}, [c for c in core.FULL if c != "day"] + ["season"]


def folds(lab, fds):
    if STAGE == "1":
        return [x for x in fds if x[0] == "DIAG10" and any(d >= 179 for _, d in x[2])]
    out = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            out.append(("EL1", 300 + len(out), {(f, int(d)) for d in ds[k:k + 5]}))
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            out.append(("P2LOO", 300 + len(out), {(f, int(d))}))
    return out


def run_cv():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, fds = prep()
    wv = dc4.weather_vectors(full)
    cols, PF = feature_sets()
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    for name, i, vd in folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        assert not (tr_m & va_m).any() and va_m.any()
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
        Xtr, ytr, Xva = tr[PF].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[PF].to_numpy(np.float32)
        pf = np.mean([pfn(Xtr[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)],
                          ytr[np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)], Xva, c) for c in CTX], axis=0)
        G = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True)
        G["validator"], G["validation_fold"] = name, i
        vaR = va.reset_index(drop=True)
        ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr, va, s, *cols[c])
                r3 = W[0] * e + W[1] * l + W[2] * m
                G["pre_%s_%d" % (c, s)] = np.clip(shrink(.8 * r3 + .2 * pf, vaR), lo, hi)
                G["fin_%s_%d" % (c, s)] = np.clip(sg2.correction(G, "pre_%s_%d" % (c, s), vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        assert G.notna().all().all()
        G.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def day_share(g, a, b, n=20000, seed=0):
    keys, inv = np.unique((g.farm + "_" + g.day.astype(str)).values, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def summarize():
    raw, full, lab, lock, fds = prep()
    pat = ct1.pattern_days(full)
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.day >= 179].reset_index(drop=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    G["dm"] = G.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean")
    P2pat = {k for k in pat if k[1] >= 179}; print("pass-2 curtain-pattern days:", sorted(P2pat))
    mean = lambda g, c: np.mean([g["fin_%s_%d" % (c, s)] for s in SEEDS], axis=0)
    verdict = {}
    for c in ("THS5", "THB"):
        print("\n==== %s vs REF (final after SG2, pass-2 rows)" % c)
        cells = []
        for v in ("DIAG10", "EL1", "P2LOO"):
            g = G[G.validator == v]; y = g.sub_ec.to_numpy(float)
            sr = [(r(g["fin_REF_%d" % s] - y), r(g["fin_%s_%d" % (c, s)] - y)) for s in SEEDS]
            cells += [b < a for a, b in sr]
            pre = (r(np.mean([g["pre_REF_%d" % s] for s in SEEDS], 0) - y), r(np.mean([g["pre_%s_%d" % (c, s)] for s in SEEDS], 0) - y))
            mR, mC = mean(g, "REF"), mean(g, c)
            print("  %-6s REF %.4f  %s %.4f  %+.2f%%  seeds %s | before SG2 %.4f -> %.4f | high days %+.1f%%, normal %+.1f%%" % (
                v, r(mR - y), c, r(mC - y), 100 * (r(mC - y) / r(mR - y) - 1), "".join("+" if b < a else "-" for a, b in sr),
                pre[0], pre[1],
                100 * (r((mC - y)[g.dm >= 1.2]) / r((mR - y)[g.dm >= 1.2]) - 1), 100 * (r((mC - y)[g.dm < 1.2]) / r((mR - y)[g.dm < 1.2]) - 1)))
        g = G[G.validator == "P2LOO"].reset_index(drop=True); y = g.sub_ec.to_numpy(float); mR, mC = mean(g, "REF"), mean(g, c)
        share = day_share(g, (mR - y) ** 2, (mC - y) ** 2)
        keys = list(zip(g.farm, g.day))
        aux = {}
        for nm, drop in (("without NOSIG F47 229", NOSIG_P2), ("without pass-2 pattern days", P2pat)):
            m = np.array([k not in drop for k in keys]); aux[nm] = r(mC[m] - y[m]) < r(mR[m] - y[m])
            print("  aux %-30s REF %.4f %s %.4f -> %s" % (nm, r(mR[m] - y[m]), c, r(mC[m] - y[m]), "better" if aux[nm] else "NOT better"))
        gain = pd.Series((mR - y) ** 2 - (mC - y) ** 2).groupby(pd.MultiIndex.from_tuples(keys)).sum().sort_values(ascending=False)
        top5 = set(gain.index[:5]); m = np.array([k not in top5 for k in keys])
        print("  (reported) without top-5 gain days %s: REF %.4f %s %.4f" % (sorted(top5), r(mR[m] - y[m]), c, r(mC[m] - y[m])))
        complete = nf.get("P2LOO") == 46 and nf.get("EL1") == 10 and nf.get("DIAG10", 0) >= 1
        ok = complete and len(cells) == 9 and all(cells) and share < .0125 and all(aux.values())
        verdict[c] = ok
        print("  VERDICT %s: seed x set better %d/%d, P2LOO day-share %.4f, aux %s -> %s" % (
            c, sum(cells), len(cells), share, all(aux.values()), ("PASS" if ok else "FAIL") if complete else "INCOMPLETE"))


def test_shift():
    raw, full, lab, lock, fds = prep()
    tX, ty, sX = __import__("common").load_raw()
    st = sX[sX.row_id.str[:3].isin(["F13", "F47"])][["row_id"] + list(p3.RAW14)].reset_index(drop=True)
    fullT = core.identify(st)
    T = core.features(st).merge(fullT[["row_id"] + list(p3.RAW14[:4])], on="row_id", validate="one_to_one")
    T = T.join(pd.concat([dp1.day_feats(g) for _, g in T.groupby(["farm", "day"])]))
    T = T.merge(ct1.thermal_features(fullT), on="row_id", how="left", validate="one_to_one")
    wv = dc4.weather_vectors(pd.concat([full, fullT], ignore_index=True))
    tr = lab.copy()
    tdays = tr[["farm", "day"]].drop_duplicates(); vdays = T[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tdays, vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
    T = T.merge(vdays, on=["farm", "day"], how="left")
    cols, _ = feature_sets()
    R3 = {}
    for c in CONFIGS:
        R3[c] = np.mean([sum(w * p for w, p in zip(W, wt0.members(tr, T, s, *cols[c]))) for s in SEEDS], axis=0)
    pat = ct1.pattern_days(fullT)
    m = np.array([(f, d) in pat for f, d in zip(T.farm, T.day)])
    print("test curtain-pattern days: %d (%s)" % (len(pat), sorted(pat)))
    for c in ("THS5", "THB"):
        sh = shrink(.8 * (R3[c] - R3["REF"]), T)
        D = pd.Series(sh[m]).groupby([T.farm[m].values, T.day[m].values]).mean()
        worst = np.sqrt(.1384 ** 2 + np.sum(sh[m] ** 2) / len(T)) / .1384 - 1
        print("%s: shift on pattern days mean %+.3f, day means %s | other days mean %+.3f sd %.3f | worst-case bound %+.1f%% of .1384"
              % (c, sh[m].mean(), D.round(2).to_dict(), sh[~m].mean(), sh[~m].std(), 100 * worst))


if __name__ == "__main__":
    if STAGE in ("1", "2"):
        run_cv(); summarize()
    elif STAGE == "3":
        test_shift()
    else:
        summarize()
