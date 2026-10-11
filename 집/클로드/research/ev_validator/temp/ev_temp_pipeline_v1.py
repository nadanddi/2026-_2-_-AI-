# -*- coding: utf-8 -*-
"""EVT1 (2026-10-12 집 클로드): the TEMPERATURE pipeline scored on the evaluation-geometry validator EV (ev_folds_v1.csv).

Everything below is FIXED BEFORE RUNNING.

Geometry (make_ev_folds_v1.py): 8 folds; per fold 'val' records (labels hidden, scored), 'del' records (inputs AND labels removed),
everything else training.  Every labelled F13/F47 record is a val record exactly once (400 records).

Worlds per fold k (common.py continuous axis -> a deleted record == all-NaN inputs, as in TF1):
  TR_k  (training rows): train_X inputs of val_k U del_k records blanked, all test_X inputs NaN (MASK world).  Training rows =
        labelled rows not in val_k U del_k.  Row weights = production train_flags_v6 weights (audit lab_feats.pkl, by row_id;
        approximation: weights not recomputed per fold).
  VA_k,f (val rows of farm f): del_k blanked, the OTHER farm's val_k records blanked (never other-ID val inputs), test_X inputs
        kept (as production evaluation features).  Features are causal by construction (production audit).
Members:
  BASE  : production recipe (0.65 (physics linear + LGB huber residual) + 0.25 Ridge + 0.10 Nystroem), seeds 7 and 101,
          RETRAINED per fold on TR_k, predicted on VA_k features.
  CODEX_REF, TabPFN A / B : NOT retrained (cost).  APPROXIMATION: stored DIAG10 out-of-fold predictions (tt1_ckpt) - models never
          saw the row's own label, but their training set differs from EV and their val features kept neighbour records.
  Cells: BASE seed {7, 101} x PFN family {A, B} = 4 cells; W40G-S gate formula.  Reported numbers = cell-mean prediction,
          plus all-4-cells direction.
v18 chain corrector (retrained per fold): chain tables built in the fold's geometry -
  training rows: MASK (train_X inputs minus val_k U del_k), record list = all records minus del_k;
  val rows: train_X minus del_k plus test_X inputs, record list = all records minus del_k;
  corrector trained (production hyper-parameters) on hours 0-7 of the stored DIAG10 OOF W40G-S residuals of records NOT in
  val_k U del_k; cell (seed s, family F) uses the corrector built from (base_REF_s, F) as in the DIAG10 frames; shrink .5.
Anchors (linker v7 mirror-CV choices cv_all400_choice_v7m.csv, same chunking): prev_is_labelled, p >= .8, 1 step, anchor known,
  |dT_air| <= 1.5; additionally dropped if prev is in val_k U del_k (count reported).  q = v18 output, q0 = its hour-0 value.
  round-12: q_h + (1 - h/8) clip(A - .36 - q0, -1, 1), h <= 7.     v22: q_h + BETA[h] clip(A - .36 - q0, -2, 2), all h,
  BETA = submitted constants.  Context (no verdict): v22r = BETA re-measured per fold on DIAG10 pass-1 anchored records
  excluding val_k U del_k (ta8 recipe).
TF1 variants on val rows:
  (b) within-record causal fill (tf1 fill_b, all groups) on "first" val records = val records whose literal previous record
      (num - 1) has no inputs in VA_k (deleted / absent); BASE re-predicted; then corrector / anchors recomputed.
  (d) lit1_in23 / lit2_in23 = NaN unless the literal previous / second-previous existing record is day-1 / day-2 (tf1 lit_fix),
      on training AND val chain tables; corrector retrained.
  (bd) both.
Scoring: rows of val records; rough days (rw1_rough_days_v1, 30 strict CO2-rough days) EXCLUDED (temperature protocol) - with-rough
  reported for context; scopes all val rows and pass-2 rows (num >= 179); per farm.  RMSE of cell-mean predictions.
DIAG10 (stored, context): same configs on the stored DIAG10 frames (tc1 layout_frame, rough excluded, C2.corrected cells,
  same anchors without fold exclusion) - (b)/(d) cannot be expressed there (no deleted records).
Bootstrap: record-level (val records pooled over folds, with replacement, 4000 draws, seed 0), P(RMSE_new >= RMSE_ref).
Pre-fixed verdicts (EV is a NEW validator; verdicts are EV-only notes, the catalog keeps DIAG10 rules):
  V1 v22 vs round-12 (ref): PASS iff cell-mean AND all 4 cells better on pass-2 and on all val rows, AND P pass-2 < .025.
  V2 (b) vs current, V3 (d) vs current, compared at the v22 stage: PASS iff all 4 cells better on pass-2 and all rows AND
      P (pass-2 and all) < .0125 (Bonferroni over the 2 candidates).
Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u ev_validator/temp/ev_temp_pipeline_v1.py
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RES = os.path.join(env.ROOT, u"집", u"클로드", "research")
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도"
sys.path.insert(0, os.path.join(PK, "code"))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


TF1 = _load("tf1", os.path.join(RES, "tf1_chunkstart", "tf1_chunkstart_fill_v1.py"))
import make_temp_v18_chain as V18  # noqa: E402
import lightgbm as lgb  # noqa: E402

OUT = os.path.join(RES, "ev_validator", "temp"); CK = os.path.join(OUT, "local_ckpt")
AUD = os.path.join(RES, "audit_temp"); OOF = os.path.join(env.LOCAL, "tt1_ckpt")
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
EVF = os.path.join(RES, "ev_validator", "ev_folds_v1.csv")
RD = TF1.RD
SEEDS = (7, 101); FAMS = ("A", "B"); CELLS = [(s, f) for s in SEEDS for f in FAMS]
NB = 4000
BETA = np.array([0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566,
                 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355])
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
log = TF1.log
ORIG = TF1.ORIG; INPUTS = TF1.INPUTS


def loader(blank, mask_sx):
    def ld():
        tX, ty, sX = ORIG(); tX = tX.copy(); sX = sX.copy()
        if mask_sx:
            sX[INPUTS] = np.nan
        m = np.array([(f, d) in blank for f, d in zip(tX.farm, tX.day)])
        tX.loc[m, [c for c in tX.columns if c not in ("row_id", "farm", "day", "hour", "t")]] = np.nan
        return tX, ty, sX
    return ld


# ------------------------------------------------------------------ part A: BASE per fold
def base_fold(k, val, dele, L0, ct, phc, RAW, wmap):
    path = os.path.join(CK, "evb_fold%d.pkl" % k)
    if os.path.exists(path):
        return pd.read_pickle(path)
    t0 = time.time()
    LT, c2, p2 = TF1.world(loader(val | dele, True)); assert c2 == ct and p2 == phc
    keyT = list(zip(LT.farm, LT.day)); trm = np.array([x not in val and x not in dele for x in keyT])
    TR = LT[trm].reset_index(drop=True); w = TR.row_id.map(wmap).values.astype(float); assert np.isfinite(w).all()
    parts = []
    for f in ("F13", "F47"):
        other = {x for x in val if x[0] != f}
        LV, c2, p2 = TF1.world(loader(dele | other, False)); assert c2 == ct
        m = np.array([(ff, d) in val and ff == f for ff, d in zip(LV.farm, LV.day)])
        parts.append(LV[m])
    VA = pd.concat(parts, ignore_index=True)
    avail_tx = set(zip(*ORIG()[0][["farm", "day"]].values.T)); avail_sx = set(zip(*ORIG()[2][["farm", "day"]].values.T))
    have = (avail_tx | avail_sx) - dele
    first = {(f, d) for f, d in val if (f, d - 1) not in have}
    VB = TF1.fill_b(VA, first, ct, RAW)
    out = VA[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy(); out["fold"] = k
    out["first"] = [(f, d) in first for f, d in zip(out.farm, out.day)]
    for s in SEEDS:
        p = TF1.fit_base(TR, ct, phc, w, s); out["a_%d" % s] = p(VA); out["b_%d" % s] = p(VB)
    out.to_pickle(path)
    log("fold %d: train rows %d, val rows %d (records %d, first %d) | %.0fs" % (k, len(TR), len(out), len(val), len(first), time.time() - t0))
    return out


# ------------------------------------------------------------------ chain tables / corrector
def chain_tables(tX, sX, recs_all, dele, val, fixd):
    sd = tX[V18.V].std()
    records = [r for r in recs_all if r not in dele]
    Xm = tX[[(f, d) not in dele and (f, d) not in val for f, d in zip(tX.farm, tX.day)]]
    Xv = pd.concat([tX[[(f, d) not in dele for f, d in zip(tX.farm, tX.day)]], sX])
    CTm = V18.chain_table(Xm, records, sd); CTv = V18.chain_table(Xv, records, sd)
    if fixd:
        CTm = TF1.lit_fix(CTm, records); CTv = TF1.lit_fix(CTv, records)
    return CTm, CTv


def corrector_preds(D, CTm, excl, Q):
    """D: stored DIAG10 OOF (all rows); returns {(s, fam): correction on Q rows (hours <= 7 else 0)}."""
    Dm = D.merge(CTm, on=["farm", "day"], how="left"); Dm["jump"] = Dm.in_temp - Dm.ch_in23
    keep = (Dm.hour <= 7).values & np.array([(f, d) not in excl for f, d in zip(Dm.farm, Dm.day)])
    Dm = Dm[keep].reset_index(drop=True)
    early = (Q.hour <= 7).values; out = {}
    for s in SEEDS:
        for fam in FAMS:
            p = V18.w40(Dm, "base_REF_%d" % s, fam); Dk = Dm.assign(p=p, p_air=p - Dm.in_temp.values)
            m = lgb.LGBMRegressor(n_estimators=100, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
            m.fit(Dk[V18.FEAT], Dk.sub_temp.values - p)
            out[(s, fam)] = m
    return out


def w40(base, codex, pfn, it):
    return TF1.w40(base, codex, pfn, it)


def anchor_apply(key, h, q, AN, rule, beta=BETA):
    out = q.copy()
    q0 = pd.Series(q[h == 0], index=key[h == 0])
    for rec, a in AN.items():
        if rec not in q0.index:
            continue
        m = key == rec; hh = h[m]; gap = a - .36 - q0[rec]
        if rule == "r12":
            wv = np.where(hh <= 7, 1 - hh / 8, 0.0); out[m] = q[m] + wv * np.clip(gap, -1, 1)
        else:
            out[m] = q[m] + beta[hh] * np.clip(gap, -2, 2)
    return out


def boot(key, y, pa, pb, mask, nb=NB):
    D = pd.DataFrame(dict(k=key[mask], sa=(pa[mask] - y[mask]) ** 2, sb=(pb[mask] - y[mask]) ** 2))
    G = D.groupby("k").agg(sa=("sa", "sum"), sb=("sb", "sum"), n=("sa", "size"))
    rng = np.random.default_rng(0); idx = rng.integers(0, len(G), (nb, len(G)))
    sa, sb, n = G.sa.values[idx].sum(1), G.sb.values[idx].sum(1), G.n.values[idx].sum(1)
    return float(np.mean(np.sqrt(sb / n) >= np.sqrt(sa / n))), len(G)


# ------------------------------------------------------------------ DIAG10 stored
def diag10_stored(AN_all):
    T = _load("tc1", os.path.join(RES, "tc1_chain_history_early_hours_v1.py"))
    B = _load("tc1b", os.path.join(RES, "tc1b_chain_history_common_train_v1.py"))
    C2 = _load("tc2", os.path.join(RES, "tc2_chain_hours_extend_v1.py"))
    CT = T.chain_table(); M = T.layout_frame("DIAG10", CT)
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values
    P = {}
    for bi, b in enumerate(("base_REF_7", "base_REF_101")):
        for fam in FAMS:
            p0 = B.w40(M, b, "codex_REF", fam); q = C2.corrected(M, p0, M, bi, fam, 7)
            P[("blend", bi, fam)] = p0; P[("v18", bi, fam)] = q
            P[("r12", bi, fam)] = anchor_apply(key, h, q, AN_all, "r12"); P[("v22", bi, fam)] = anchor_apply(key, h, q, AN_all, "v22")
    return M, key, P


def main():
    os.makedirs(CK, exist_ok=True)
    L0 = pd.read_pickle(os.path.join(AUD, "lab_feats.pkl")); C = json.load(open(os.path.join(AUD, "cols.json"))); ct, phc = C["ct"], C["phc"]
    wmap = dict(zip(L0.row_id, L0.w)); RAW = TF1.raw_panel()
    F = pd.read_csv(EVF)
    folds = sorted(F.fold.unique())
    R = []
    for k in folds:
        val = set(map(tuple, F[(F.fold == k) & (F.role == "val")][["farm", "num"]].values))
        dele = set(map(tuple, F[(F.fold == k) & (F.role == "del")][["farm", "num"]].values))
        R.append(base_fold(k, val, dele, L0, ct, phc, RAW, wmap))
    R = pd.concat(R, ignore_index=True)
    st = pd.concat([pd.read_csv(os.path.join(OOF, "DIAG10_%d.csv" % k)) for k in range(10)], ignore_index=True)
    stI = st.set_index("row_id").loc[R.row_id]
    for c in ("codex_REF", "pfn_A", "pfn_B", "base_REF_7", "base_REF_101"):
        R["st_" + c] = stI[c].values
    log("val rows %d, records %d | EV BASE vs stored DIAG10 BASE: corr s7 %.4f s101 %.4f, RMS diff %.4f / %.4f"
        % (len(R), R.groupby(["farm", "day"]).ngroups, np.corrcoef(R.a_7, R.st_base_REF_7)[0, 1], np.corrcoef(R.a_101, R.st_base_REF_101)[0, 1],
           rm(R.a_7 - R.st_base_REF_7), rm(R.a_101 - R.st_base_REF_101)))

    # chain / corrector / anchors per fold
    D0 = os.path.join(env.ROOT, u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
    tX = V18.ident(pd.read_csv(os.path.join(D0, "train_X.csv"))); sX = V18.ident(pd.read_csv(os.path.join(D0, "test_X.csv")))
    recs_all = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
    Lk = pd.read_csv(LK)
    Lk = Lk[Lk.prev_is_labelled.astype(bool) & (Lk.p >= .8) & (Lk.steps_to_anchor == 1) & Lk.anchor_st23.notna() & (Lk.dT_air.abs() <= 1.5)]
    AN_all = dict(zip(Lk.rec, Lk.anchor_st23)); PREV = dict(zip(Lk.rec, Lk.prev))
    # DIAG10 stored frames (for v22r BETA re-measure and the context table)
    M, keyM, PD = diag10_stored(AN_all)
    Rk = []; dropped = 0; fired = 0
    for k in folds:
        val = set(map(tuple, F[(F.fold == k) & (F.role == "val")][["farm", "num"]].values))
        dele = set(map(tuple, F[(F.fold == k) & (F.role == "del")][["farm", "num"]].values))
        excl = val | dele; exk = {"%s_%03d" % x for x in excl}
        Q = R[R.fold == k].reset_index(drop=True).copy()
        key = (Q.farm + "_" + Q.day.map("{:03d}".format)).values; h = Q.hour.values
        AN = {}
        for r in set(key):
            if r in AN_all:
                if PREV[r] in exk:
                    dropped += 1
                else:
                    AN[r] = AN_all[r]
        fired += len(AN)
        # v22r: BETA re-measured on DIAG10 pass-1 anchored records outside val_k U del_k
        num = np.zeros(24); den = np.zeros(24); hM = M.hour.values; yM = M.sub_temp.values
        for bi in (0, 1):
            for fam in FAMS:
                q = PD[("v18", bi, fam)]; q0 = pd.Series(q[hM == 0], index=keyM[hM == 0])
                for rec, a in AN_all.items():
                    if rec not in q0.index or int(rec[4:7]) >= 179 or rec in exk:
                        continue
                    g = a - .36 - q0[rec]; mm = keyM == rec
                    num[hM[mm]] += g * (yM[mm] - q[mm]); den[hM[mm]] += g * g
        beta_r = num / den
        for fixd in (False, True):
            CTm, CTv = chain_tables(tX, sX, recs_all, dele, val, fixd)
            Qc = Q.merge(CTv, on=["farm", "day"], how="left"); Qc["jump"] = Qc.in_temp - Qc.ch_in23
            assert (Qc.row_id.values == Q.row_id.values).all()
            mods = corrector_preds(st, CTm, excl, Qc)
            early = (h <= 7)
            for t in ("a", "b"):
                tag = t + ("d" if fixd else "")
                for s, fam in CELLS:
                    p = w40(Q["%s_%d" % (t, s)].values, Q.st_codex_REF.values, Q["st_pfn_" + fam].values, Q.in_temp.values)
                    Qk = Qc.assign(p=p, p_air=p - Qc.in_temp.values)
                    q = p.copy(); q[early] = p[early] + .5 * mods[(s, fam)].predict(Qk.loc[early, V18.FEAT])
                    c = "%d%s" % (s, fam)
                    if not fixd:
                        Q["blend_%s_%s" % (t, c)] = p
                    Q["v18_%s_%s" % (tag, c)] = q
                    Q["r12_%s_%s" % (tag, c)] = anchor_apply(key, h, q, AN, "r12")
                    Q["v22_%s_%s" % (tag, c)] = anchor_apply(key, h, q, AN, "v22")
                    if tag == "a":
                        Q["v22r_a_%s" % c] = anchor_apply(key, h, q, AN, "v22", beta_r)
        Q["anchored"] = [x in AN for x in key]
        Rk.append(Q)
        log("fold %d: anchored val records %d | beta_r h0..3 %s" % (k, len(AN), " ".join("%.2f" % b for b in beta_r[:4])))
    R = pd.concat(Rk, ignore_index=True)
    log("anchors fired %d, dropped because prev in val/del of the fold %d" % (fired, dropped))
    cfgs = ["blend_a", "v18_a", "r12_a", "v22_a", "v22r_a", "blend_b", "v18_b", "r12_b", "v22_b", "v18_ad", "r12_ad", "v22_ad",
            "v18_bd", "r12_bd", "v22_bd"]
    for c in cfgs:
        R[c] = np.mean([R["%s_%d%s" % (c, s, f)] for s, f in CELLS], 0)
    R["base_a"] = np.mean([R.a_7, R.a_101], 0)
    R.to_pickle(os.path.join(CK, "ev_all.pkl"))

    rough = np.array([(f, d) in RD for f, d in zip(R.farm, R.day)]); p2 = (R.day >= 179).values
    key = (R.farm + "_" + R.day.map("{:03d}".format)).values; y = R.sub_temp.values
    scopes = {"all": ~rough, "pass2": p2 & ~rough, "all_withrough": np.ones(len(R), bool), "pass2_withrough": p2,
              "F13_all": ~rough & (R.farm == "F13").values, "F47_all": ~rough & (R.farm == "F47").values,
              "F13_pass2": p2 & ~rough & (R.farm == "F13").values, "F47_pass2": p2 & ~rough & (R.farm == "F47").values,
              "first_all": ~rough & R["first"].values, "anchored_pass2": p2 & ~rough & R.anchored.values}
    rows = []
    for sc, m in scopes.items():
        row = dict(scope=sc, n_rows=int(m.sum()), n_rec=len(set(key[m])), base_a=rm((R.base_a - y)[m]))
        for c in cfgs:
            row[c] = rm((R[c].values - y)[m])
        rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "ev_temp_scores_v1.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    log("\n================ EV RMSE (cell-mean predictions; rough days excluded unless *_withrough) ================")
    log(T.set_index("scope").round(4).T.to_string())
    # cells
    crow = []
    for sc in ("all", "pass2"):
        m = scopes[sc]
        for s, f in CELLS:
            c = "%d%s" % (s, f); d = dict(scope=sc, cell=c)
            for cf in cfgs:
                d[cf] = rm((R["%s_%s" % (cf, c)].values - y)[m])
            crow.append(d)
    TC = pd.DataFrame(crow); TC.to_csv(os.path.join(OUT, "ev_temp_cells_v1.csv"), index=False)
    log("\nper cell:"); log(TC.round(4).to_string(index=False))

    # DIAG10 stored, same configs
    yM = M.sub_temp.values; p2M = M.day.values >= 179
    drows = []
    for sc, m in (("all", np.ones(len(M), bool)), ("pass2", p2M), ("F13_all", (M.farm == "F13").values), ("F47_all", (M.farm == "F47").values),
                  ("F13_pass2", p2M & (M.farm == "F13").values), ("F47_pass2", p2M & (M.farm == "F47").values)):
        d = dict(scope=sc, n_rows=int(m.sum()))
        for cf in ("blend", "v18", "r12", "v22"):
            pm = np.mean([PD[(cf, bi, f)] for bi in (0, 1) for f in FAMS], 0); d[cf] = rm((pm - yM)[m])
            d[cf + "_cells"] = " ".join("%.4f" % rm((PD[(cf, bi, f)] - yM)[m]) for bi in (0, 1) for f in FAMS)
        drows.append(d)
    TD = pd.DataFrame(drows); TD.to_csv(os.path.join(OUT, "ev_temp_diag10_stored_v1.csv"), index=False)
    log("\n================ DIAG10 stored (same configs, rough excluded) ================"); log(TD.to_string(index=False))

    # verdicts
    log("\n================ bootstrap / verdicts ================")

    def cmp(ref, new, alpha, name):
        res = {}
        for sc in ("pass2", "all"):
            m = scopes[sc]; P, n = boot(key, y, R[ref].values, R[new].values, m)
            cm = rm((R[new].values - y)[m]) < rm((R[ref].values - y)[m])
            cells = [rm((R["%s_%d%s" % (new, s, f)].values - y)[m]) < rm((R["%s_%d%s" % (ref, s, f)].values - y)[m]) for s, f in CELLS]
            res[sc] = (P, n, cm, all(cells), sum(cells))
            log("  %-26s %-6s records %3d  RMSE %.4f -> %.4f (%+.2f%%)  cells better %d/4  P(new worse) %.4f"
                % (name, sc, n, rm((R[ref].values - y)[m]), rm((R[new].values - y)[m]),
                   100 * (rm((R[new].values - y)[m]) / rm((R[ref].values - y)[m]) - 1), sum(cells), P))
        ok = all(res[sc][2] and res[sc][3] for sc in ("pass2", "all")) and res["pass2"][0] < alpha and (name.startswith("V1") or res["all"][0] < alpha)
        log("  -> %s %s (alpha %.4f)" % (name, "PASS" if ok else "FAIL", alpha))
        return res
    cmp("r12_a", "v22_a", .025, "V1 v22 vs round-12")
    cmp("v22_a", "v22_b", .0125, "V2 (b) vs current @v22")
    cmp("v22_a", "v22_ad", .0125, "V3 (d) vs current @v22")
    log("context (no verdict):")
    cmp("v22_a", "v22_bd", 1.0, "ctx (b)+(d) @v22")
    cmp("blend_a", "blend_b", 1.0, "ctx (b) @blend")
    cmp("v18_a", "v18_ad", 1.0, "ctx (d) @v18")
    cmp("blend_a", "v18_a", 1.0, "ctx v18 vs blend")
    cmp("v18_a", "r12_a", 1.0, "ctx round-12 vs v18")
    cmp("v18_a", "v22_a", 1.0, "ctx v22 vs v18")
    cmp("v22_a", "v22r_a", 1.0, "ctx v22 refit-beta vs v22")
    # first records only for (b)
    m = scopes["first_all"]
    if m.any():
        P, n = boot(key, y, R.v22_a.values, R.v22_b.values, m)
        log("  ctx (b) on first records only: records %d RMSE %.4f -> %.4f P %.4f" % (n, rm((R.v22_a - y)[m]), rm((R.v22_b - y)[m]), P))


if __name__ == "__main__":
    main()
