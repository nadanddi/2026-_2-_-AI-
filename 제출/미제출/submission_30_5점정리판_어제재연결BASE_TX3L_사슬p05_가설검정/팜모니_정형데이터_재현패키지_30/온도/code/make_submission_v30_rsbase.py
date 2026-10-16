# -*- coding: utf-8 -*-
"""submission_30 temperature, stage 0 (2026-10-16 집 클로드; '5-point clean' BASE — catalog 7.031~7.036, CLN2/CLN4J/CLN5K/CLN6J).
NX (make_submission_v26_temp.py: W40G-S without `day` in BASE and without `second` in Codex/TabPFN) keeps one competition-file
dependency: BASE's cross-day history features (lags, rolling 24/72/168 h, ewm, previous-day aggregates, ph_* 24/48/72 h) are built on
a per-farm axis ordered by RECORD NUMBER, so at early hours they often carry another house's history.
Here the BASE member is refitted with those features RE-STITCHED to the same house's yesterday:
  for every target record r (all labelled F13/F47 training records = training rows; all F13/F47 evaluation records = test rows) its
  history days d-1 .. d-7 on its farm axis are REPLACED by its field-linker (FL) predecessor chain c1..c7:
     step(x) = FL previous record of x with probability >= .5 (training records: fl3_train_links.csv; evaluation records: fl_eval_links.csv
     prev0/p0); NO link -> the chain STOPS (positions beyond = blank).  Never falls back to the previous record number.
  Visibility: training targets use only training records; an evaluation target may use training records and SAME-farm EARLIER-numbered
  evaluation records (never other-farm evaluation inputs, never later inputs).  The production feature builders are unchanged
  (V13.build_frames: MASK world for training rows, full world for test rows); 8 worlds (record number mod 8) so rewritten windows never
  overlap a target.  Labels are never used to build features.
Model: BASE recipe, row weights, seeds = NX (V13 / v26) exactly; Codex/TabPFN unchanged (NX).  Since the W40G-S BASE weight is a constant .4,
  the new cell = NX cell + .4 * (BASE_restitched - BASE_NX), with BASE_NX refitted here by the same code (no `day`).
Run:  cd code && python make_submission_v30_rsbase.py <nx27 csv> <fl3_train_links.csv> <fl_eval_links.csv> <out.csv> [<members_out.csv>]
"""
import os, sys, time
import env  # noqa: F401  MUST be the first project import
import env_extra  # noqa: F401
import numpy as np, pandas as pd
import make_submission_v13_temp as V13
import train_flags_v6 as TF
import resid_reset_features as RR
import common
from make_submission_v3 import m_lgbh, m_ridge, m_nys
from make_submission_v7 import fit_mean_w
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

DEPTH, NCOL, PMIN = 7, 8, 0.5
FARMS = ("F13", "F47")
RAW0 = common.load_raw   # original loader (build_frames temporarily replaces common.load_raw)
IDC = ("row_id", "farm", "day", "hour", "t")


def base_fit(lab, test, ct, phc, w):
    y = lab.sub_temp.values
    imp = SimpleImputer(strategy="median").fit(lab[phc])
    lin = LinearRegression().fit(imp.transform(lab[phc]), y, sample_weight=w)
    b_tr, b_te = lin.predict(imp.transform(lab[phc])), lin.predict(imp.transform(test[phc]))
    return (V13.W_TEMP["resid_lgb"] * (b_te + fit_mean_w(m_lgbh, lab[ct], y - b_tr, test[ct], w))
            + V13.W_TEMP["ridge"] * fit_mean_w(m_ridge, lab[ct], y, test[ct], w, "ridge")
            + V13.W_TEMP["nys"] * fit_mean_w(m_nys, lab[ct], y, test[ct], w, "ridge"))


def key(r):
    return (r[:3], int(r[4:7]))


def main():
    nxp, trl, evl, out = sys.argv[1:5]; memp = sys.argv[5] if len(sys.argv) > 5 else None
    assert not os.path.exists(out), "refusing to overwrite " + out
    if "second" in RR.FEATURE_COLUMNS: RR.FEATURE_COLUMNS.remove("second")
    t0 = time.time()
    # ---- NX frames and BASE (reference) ----
    lab, test, ct, phc, sX = V13.build_frames(); ct = [c for c in ct if c != "day"]
    w = TF.row_weights(lab, V13.W_FLAG, w_noisy=V13.W_NOISY)
    b_nx = base_fit(lab, test, ct, phc, w); print("NX frames + BASE done (%.0fs) | train rows %d test rows %d" % (time.time() - t0, len(lab), len(test)), flush=True)
    # ---- raw content and FL links ----
    tX, ty, sX0 = RAW0(); cols = [c for c in tX.columns if c not in IDC]
    cont, train, testr = {}, set(), set()
    for src, D in (("t", tX), ("s", sX0)):
        D = D[D.farm.isin(FARMS)]
        for (f, d), g in D.groupby(["farm", "day"]):
            a = np.full((24, len(cols)), np.nan); a[g.hour.values] = g[cols].values; cont[(f, int(d))] = a
            (testr if src == "s" else train).add((f, int(d)))
    labelled = set(zip(ty.row_id.str[:3], ty.row_id.str[4:7].astype(int))) & train
    fmin = {f: min(d for ff, d in cont if ff == f) for f in FARMS}
    TL = pd.read_csv(trl); EL = pd.read_csv(evl, encoding="utf-8-sig")
    link = {key(r): key(p) for r, p, q in zip(TL.rec, TL.prev, TL.p) if isinstance(p, str) and q >= PMIN}
    link.update({key(r): key(p) for r, p, q in zip(EL.rec, EL.prev0, EL.p0) if isinstance(p, str) and q >= PMIN})
    def chain(r, vis):
        o, cur, seen = [], r, {r}
        for _ in range(DEPTH):
            nx = link.get(cur)
            if nx is None or nx not in cont or nx in seen or not vis(nx): break
            o.append(nx); seen.add(nx); cur = nx
        return o
    vis_tr = lambda x: x in train
    def vis_te(r):
        return lambda x: not (x in testr and (x[0] != r[0] or x[1] >= r[1]))
    T_tr = sorted(r for r in labelled if r[0] in FARMS); T_te = sorted(testr)
    stats = []
    labRS, testRS = [], []
    for j in range(NCOL):
        tj_tr = [r for r in T_tr if r[1] % NCOL == j]; tj_te = [r for r in T_te if r[1] % NCOL == j]
        writes = {}
        for r, vis in [(r, vis_tr) for r in tj_tr] + [(r, vis_te(r)) for r in tj_te]:
            ch = chain(r, vis); stats.append(dict(rec="%s_%03d" % r, test=r in testr, depth=len(ch)))
            for k in range(1, DEPTH + 1):
                pos = (r[0], r[1] - k)
                if pos[1] < fmin[pos[0]]: continue
                assert pos not in writes, "overlapping windows"
                writes[pos] = cont[ch[k - 1]] if k <= len(ch) else np.full((24, len(cols)), np.nan)
        def loader(writes=writes):
            a, b, c = RAW0(); a = a.copy(); c = c.copy()
            for D in (a, c):
                sel = D.farm.isin(FARMS).values; sub = D.loc[sel, ["farm", "day", "hour"]]
                hit = np.array([(f, d) in writes for f, d in zip(sub.farm, sub.day)])
                if hit.any():
                    ix = sub.index[hit]; ss = sub[hit]
                    D.loc[ix, cols] = np.stack([writes[(f, d)][h] for f, d, h in zip(ss.farm, ss.day, ss.hour)])
            have = set(zip(a.farm, a.day)) | set(zip(c.farm, c.day)); add = []
            for (f, d), arr in writes.items():
                if (f, d) in have or np.isnan(arr).all(): continue
                df = pd.DataFrame(arr, columns=cols); df["row_id"] = ["%s_%03d_%02d" % (f, d, h) for h in range(24)]
                df["farm"] = f; df["day"] = d; df["hour"] = np.arange(24); df["t"] = d * 24 + df.hour; add.append(df[a.columns])
            if add: a = pd.concat([a] + add, ignore_index=True)
            return a, b, c
        lj, tj, _, _, _ = V13.build_frames(loader)
        s_tr = set(tj_tr); s_te = set(tj_te)
        labRS.append(lj[[(f, int(d)) in s_tr for f, d in zip(lj.farm, lj.day)]]); testRS.append(tj[[(f, int(d)) in s_te for f, d in zip(tj.farm, tj.day)]])
        print("world %d/%d done (%.0fs) | targets train %d test %d" % (j + 1, NCOL, time.time() - t0, len(tj_tr), len(tj_te)), flush=True)
    labRS = pd.concat(labRS).drop_duplicates("row_id").set_index("row_id").loc[lab.row_id].reset_index()
    testRS = pd.concat(testRS).drop_duplicates("row_id").set_index("row_id").loc[test.row_id].reset_index()
    assert (labRS.sub_temp.values == lab.sub_temp.values).all()
    b_rs = base_fit(labRS, testRS, ct, phc, w)
    S = pd.DataFrame(stats); print("chain depth: training targets mean %.2f (depth0 %d) | evaluation targets mean %.2f (depth0 %d of %d)"
                                  % (S[~S.test].depth.mean(), int((~S.test & (S.depth == 0)).sum()), S[S.test].depth.mean(), int((S.test & (S.depth == 0)).sum()), int(S.test.sum())), flush=True)
    NX = pd.read_csv(nxp).set_index("row_id").sub_temp.loc[test.row_id].values
    new = NX + 0.4 * (b_rs - b_nx)
    O = pd.DataFrame({"row_id": test.row_id.values, "sub_temp": new}); O = O.set_index("row_id").loc[sX.row_id].reset_index()
    assert len(O) == 1440 and O.row_id.is_unique and np.isfinite(O.sub_temp).all()
    O.to_csv(out, index=False, encoding="utf-8", float_format="%.6f")
    if memp: pd.DataFrame({"row_id": test.row_id.values, "base_nx": b_nx, "base_rs": b_rs, "nx_cell": NX, "cell_rs": new}).to_csv(memp, index=False)
    d = new - NX; print("wrote %s | cell change mean %+.3f rms %.3f max %.3f | %.0fs" % (out, d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(), time.time() - t0))


if __name__ == "__main__":
    main()
