# -*- coding: utf-8 -*-
"""LK1: ONE-TIME independent confirmation of the curtain-schedule features (THS5, CT2 'hold' candidate) on the
LOCKED 40 days.  (2026-10-09 집 클로드, user: "잠금 40일로 커튼 특징(THS5)을 독립 확인").  Fixed before running; the
result is reported whatever it is and the lock is considered SPENT after this run.
Why the lock: the curtain schedule was discovered on the 360 non-lock days; the lock days (drawn without labels,
2026-09-29; opened once on 10-02 for the season index) were not used in that discovery.
Validation: each of the 40 lock days on its own (40 folds, like P2LOO); training = the 360 non-lock days minus the
same farm's d-1..d+1 and the other farm's d-3..d+3 (other lock days are never in training).
Pipeline (submission, as CT2): final = clip(shrink(0.8 R3 + 0.2 PFN)) [+ SG2 on pass-2 lock days, reference =
non-lock labelled days, lock days excluded as candidates]; clip = training label range.
  REF  current R3 features (FULL w/o day + season + DP1)
  THS5 REF + th_full_hours_td, th_night_closed, th_h9, th_h9_partial, th_sched_score (R3 only; PFN shared)
NEW seeds 2525 / 4646 / 6767; PFN contexts 13-16 (unused), TabPFN v2, n_est 4, GPU.
PRIMARY (k = 1, alpha .025), all 960 rows of the 40 lock days:
  CONFIRMED iff every seed has lower RMSE with THS5 AND seed-mean day bootstrap (farm, day blocks, 20,000)
  share(THS5 not better) < .025.  Otherwise NOT CONFIRMED (CT2 stays a hold candidate / rejected).
Reported: the 4 lock high days on the curtain schedule (F13 123, 127; F47 134, 143), pass-1 / pass-2, high / normal,
days better, top-day concentration.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u lk1_lock40_curtain_confirm_v1.py [sum]
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
ct1, sg2, wt0, dp1, dc4, p3, core = ct2.ct1, ct2.sg2, ct2.wt0, ct2.dp1, ct2.dc4, ct2.p3, ct2.core
import common
SEEDS = (2525, 4646, 6767)
CTX = (13, 14, 15, 16)
CK = os.path.join(env.LOCAL, "lk1_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
CONFIGS = ("REF", "THS5")
CURT_HIGH = {("F13", 123), ("F13", 127), ("F47", 134), ("F47", 143)}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def build_all():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    build = core.features(raw).merge(full[["row_id"] + p3.RAW14[:4]], on="row_id", validate="one_to_one")
    tX, ty, sX = common.load_raw()
    y = ty[["row_id", "sub_ec"]]
    build["farm"] = build.row_id.str[:3]; build["day"] = build.row_id.str[4:7].astype(int)
    L = build[[(f, int(d)) in lock for f, d in zip(build.farm, build.day)]].merge(y, on="row_id", validate="one_to_one")
    L = L[L.sub_ec.notna()].copy()
    assert L[["farm", "day"]].drop_duplicates().shape[0] == 40, L[["farm", "day"]].drop_duplicates().shape
    TF = ct1.thermal_features(full)
    out = []
    for D in (lab, L):
        D = D.join(pd.concat([dp1.day_feats(g) for _, g in D.groupby(["farm", "day"])]))
        D = D.merge(TF, on="row_id", how="left", validate="one_to_one").set_index(D.index if D is not None else None)
        out.append(D)
    return raw, full, out[0], out[1], lock


def run():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, L, lock = build_all()
    wv = dc4.weather_vectors(full)
    cols, PF = ct2.feature_sets()
    cols = {k: v for k, v in cols.items() if k in CONFIGS}
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    for (f, d) in sorted(set(zip(L.farm, L.day))):
        path = os.path.join(CK, "%s_%d.csv" % (f, d))
        if os.path.exists(path):
            continue
        forb = {(f, d + j) for j in (-1, 0, 1)} | {(OTHER[f], d + j) for j in range(-3, 4)}
        tr = lab[[(a, int(b)) not in forb for a, b in zip(lab.farm, lab.day)]].copy()
        va = L[(L.farm == f) & (L.day == d)].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(a, b)] for a, b in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
        Xtr, ytr, Xva = tr[PF].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[PF].to_numpy(np.float32)
        ps = []
        for c in CTX:
            ix = np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)
            ps.append(ct2.pfn(Xtr[ix], ytr[ix], Xva, c))
        pf = np.mean(ps, axis=0)
        G = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True)
        vaR = va.reset_index(drop=True)
        vd = {(f, d)}; ref = set(labset); cal = sg2.ref_calendar(R, WV, ref)
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr, va, s, *cols[c])
                G["pre_%s_%d" % (c, s)] = np.clip(ct2.shrink(.8 * (W[0] * e + W[1] * l + W[2] * m) + .2 * pf, vaR), lo, hi)
                G["fin_%s_%d" % (c, s)] = np.clip(sg2.correction(G, "pre_%s_%d" % (c, s), vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        assert G.notna().all().all()
        G.to_csv(path, index=False)
        print("%s %d done" % (f, d), flush=True)


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nd = G[["farm", "day"]].drop_duplicates().shape[0]; print("lock days scored:", nd)
    G["dm"] = G.groupby(["farm", "day"]).sub_ec.transform("mean")
    mean = lambda g, c: np.mean([g["fin_%s_%d" % (c, s)] for s in SEEDS], axis=0)
    y = G.sub_ec.to_numpy(float); mR, mT = mean(G, "REF"), mean(G, "THS5")
    seeds = [(r(G["fin_REF_%d" % s] - y), r(G["fin_THS5_%d" % s] - y)) for s in SEEDS]
    share = ct2.day_share(G, (mR - y) ** 2, (mT - y) ** 2)
    print("ALL 40 lock days: REF %.4f  THS5 %.4f  %+.2f%%  seeds %s  day-share %.4f" % (
        r(mR - y), r(mT - y), 100 * (r(mT - y) / r(mR - y) - 1), " ".join("%.4f->%.4f" % sv for sv in seeds), share))
    views = [("pass-1", G.day < 179), ("pass-2", G.day >= 179), ("high (>=1.2)", G.dm >= 1.2), ("normal (<1.2)", G.dm < 1.2),
             ("4 curtain high days", np.array([(f, d) in CURT_HIGH for f, d in zip(G.farm, G.day)]))]
    for nm, m in views:
        m = np.asarray(m)
        if m.any():
            print("  %-22s days %2d  REF %.4f  THS5 %.4f  %+.2f%%" % (nm, G[m][["farm", "day"]].drop_duplicates().shape[0],
                  r(mR[m] - y[m]), r(mT[m] - y[m]), 100 * (r(mT[m] - y[m]) / r(mR[m] - y[m]) - 1)))
    D = G.assign(gain=(mR - y) ** 2 - (mT - y) ** 2, pR=mR, pT=mT).groupby(["farm", "day"]).agg(
        y=("sub_ec", "mean"), REF=("pR", "mean"), THS5=("pT", "mean"), gain=("gain", "sum")).sort_values("gain", ascending=False)
    print("  days better %d / %d; top-3 share of gain %.0f%%" % ((D.gain > 0).sum(), len(D), 100 * D.gain.head(3).sum() / D.gain.sum() if D.gain.sum() else np.nan))
    print(D.round(3).to_string())
    ok = nd == 40 and all(b < a for a, b in seeds) and share < .025
    print("\nVERDICT: %s" % ("CONFIRMED" if ok else "NOT CONFIRMED"))


if __name__ == "__main__":
    if MODE != "sum":
        run()
    summarize()
