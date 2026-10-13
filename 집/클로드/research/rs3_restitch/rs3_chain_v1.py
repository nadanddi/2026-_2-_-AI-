# -*- coding: utf-8 -*-
"""RS3 (2026-10-13 집 클로드; 워크플로 과제).  사전 고정(이 파일 커밋 후 실행).  온도(sub_temp, F13/F47).
요약: RS2(6.795)는 BASE 만 바꾸고 v18 사슬 보정기는 옛 BASE OOF 로 학습된 그대로 → 이중 보정 우려.  RS3 = RS2 BASE 위에서 v18 보정기를
  RS2 기반 혼합으로 다시 학습(R3) + 보정기 사슬 특징을 v15R 짝으로(R3B, B4 정의) + TN8 추가 앵커(R3BT).  주 판정 = R3BT vs 현재(CUR).

Everything below is FIXED BEFORE RUNNING.

Harness: EV temperature harness = ev_validator/temp/ev_temp_pipeline_v1.py config 'a' geometry (8 EV folds, val rows scored, 30 strict
  CO2-rough days excluded as in RS1/RS2, pass-2 = num >= 179), cells = BASE seed {7, 101} x TabPFN family {A, B}; P values on the
  cell-mean prediction.  CODEX_REF / TabPFN A, B = stored DIAG10 OOF (as in the harness).
Arms (final stage = after anchors):
  CUR  = stored current final chain (ev_all.pkl v22_a cells), recomputed here through the same code path (must match exactly).
  R2   = RS2 BASE (rs2_restitch/local_ckpt/rs2b_fold*.pkl r_7 / r_101) in the unchanged chain (stored-source v18 corrector, v22
         anchors) = RS2 'final_rs1'; must reproduce rs2_ev_all.pkl xv22_r cells exactly.
  R3   = RS2 BASE + v18 corrector RETRAINED on the RS2-based blend (recipe unchanged: TC1c LightGBM 100 trees, lr .03, 15 leaves,
         min_child 100, subsample .8, colsample .8, seed 1, hours 0-7, shrink .5; one corrector per cell), v22 anchors.
  R3B  = R3 with the corrector's chain features from the v15R pick (B4 definition), on training AND val rows (corrector retrained).
  R3BT = R3B + TN8 extra anchors on top of the v22 anchors.

Corrector training source.  How the stored v18 picks it (read from EVP.corrector_preds / EVP.main before writing this):
  for EV fold k the training rows are the stored DIAG10 out-of-fold rows (local/tt1_ckpt/DIAG10_*.csv: 10-fold CV over the 400
  labelled F13/F47 records; features in the MASK world = test_X inputs NaN; each row predicted by a model that never saw its own
  record's label) of records NOT in val_k U del_k, hours 0-7; p = W40G-S(base_REF_s, codex_REF, pfn_F); target y - p; chain
  features from CTm_k (EVP.chain_tables: fold-k MASK chain table, train_X inputs minus val_k U del_k, record list minus del_k).
  Mapping used by R3 / R3B: the SAME rows, the same code (EVP.corrector_preds), the same codex/pfn columns; only base_REF_s is
  replaced by m_s = an RS2-feature, out-of-fold, MASK-world BASE prediction:
    m_s(r), r a labelled record, j = the EV fold where r is a val record: prediction of the RS2 fold-j BASE model (trained on TR_j =
    labelled records not in val_j U del_j, so it never saw r's label) on r's RS2 re-stitched features built in "VM_j" = RS2's VA_j
    world (del_j blanked, RS2 visibility rules) with test_X inputs NaN and test records not usable as chain predecessors.
  Why not the stored RS2 val predictions r_s: they were built with test_X inputs kept (evaluation-row world).  As features of the
    corrector's TRAINING rows they would break the training-row MASK rule ("nothing fitted on test inputs").  The fold-j models
    were not saved, so they are REFITTED with the identical deterministic recipe and data (RS2.build_rows + TF1.fit_base, same
    seeds, same TR_j).  The refit is accepted only if it reproduces the stored rs2b_fold j r_s on the VA_j rows (max |diff| < 1e-6
    for both seeds, every fold) - i.e. it is the same BASE model, not a new BASE; otherwise the report STOPS.  BASE predictions of
    val rows are always the stored checkpoint values.
  Fold safety: corrector of EV fold k never trains on rows of val_k or del_k records.  Second-order exposure (identical in kind to
    the stored v18): the fold-j model that produced m_s(r) for r outside val_k U del_k may have been trained on val_k labels (the
    stored DIAG10 OOF models were likewise trained on folds containing val_k records).
B4 chain features (R3B, R3BT): ch_in23 (pick's 23 h in_temp), ch_eve (pick's 18-23 h mean in_temp), jump = in_temp - ch_in23 are taken
  from the v15R first pick when p0 >= .5 and the pick has an hour-23 input row visible to the row, else the TC1c value; ch_cost and
  lit1/lit2_in23 unchanged (B4 = b4r_chain_v15r_dawn_v1.py / ext1_combo_b4_tn8_window_v1.py).
    training rows (corrector fit, fold k): pick = local/linker_v10/v15r15_mirror.csv (as B4 on DIAG10); visible iff the pick is a
      train_X record not in val_k U del_k; content = its train_X inputs (MASK world).
    val rows: pick = v15r15_ev.csv (linker refitted per EV fold, as TN8 on EV); visible iff not in del_k AND (a train_X record not in
      val_k, OR same ID with a smaller number); content = its train_X / test_X inputs.
TN8 (R3BT; tb1r EV part): rec in val_k, v15r15_ev prev0 with p0 >= .8 and lab0 (labelled in the fold world), rec NOT in the v7 set
  (cv_all400_choice_v7m.csv p >= .8 & prev_is_labelled & |dT_air| <= 1.5), prev0 not in val_k U del_k (dropped otherwise, counted);
  A = sub_temp label of prev0 at hour 23 (linker_v3 base.pkl ST, as tb1r); final = q_h + BETA[h] clip(A - .36 - q0, -2, 2) with q = the
  v18-stage output of R3B and q0 = q at hour 0 (v22 formula; BETA = EVP.BETA = ta8_beta).  v22 anchors (v7 list, fold drop) as EVP
  for every arm; TN8 records are disjoint from v7 anchors by construction.
PRE-FIXED PASS (primary, single candidate, no Bonferroni): R3BT vs CUR at the final stage - RMSE better in EVERY one of the 4 cells on
  pass-2 rows AND on all val rows (rough excluded), AND record-level bootstrap (val records pooled over folds, with replacement, 4000
  draws, seed 0, cell-mean predictions; EVP.boot) P(RMSE_R3BT >= RMSE_CUR on pass-2) < .025.
Ablations / context (no verdict): each arm vs CUR and vs the previous step (R2 vs CUR, R3 vs R2, R3B vs R3, R3BT vs R3B): cell % changes
  and P on pass-2 / all rows; per farm; cold rows (in_temp <= 8 C, EXT-like); with rough days; BASE alone and v18 stage; number of
  pass-2 records improved / worsened (cell-mean SSE).
Approximations (known before running, inherited from EVP / RS2 / TN8): CODEX / TabPFN stored DIAG10 OOF, row weights not recomputed per
  fold; linker files consumed as given (fit on all training links; the mirror-geometry candidate softmax of a TRAINING record can
  include same-ID earlier test records, so the "p0 >= .5" decision of a training row may depend on test inputs - the feature value
  itself uses only visible train_X inputs; the number of such training picks is reported); RS2 restitch caveats (ewm tails etc.).
Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 python -u rs3_restitch/rs3_chain_v1.py --mask 0,4   (refit + MASK OOF
      per fold, any split over processes), then  ... rs3_chain_v1.py --report
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time, pickle, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RES = os.path.join(env.ROOT, u"집", u"클로드", "research")
OUT = os.path.join(RES, "rs3_restitch"); CK = os.path.join(OUT, "local_ckpt")


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


RS2 = _load("rs2", os.path.join(RES, "rs2_restitch", "rs2_restitch_v1.py"))
EVP = RS2.EVP; TF1 = EVP.TF1; V18 = EVP.V18
EVF = RS2.EVF
SEEDS, FAMS, CELLS = EVP.SEEDS, EVP.FAMS, EVP.CELLS
rm = EVP.rm; log = EVP.log; boot = EVP.boot
L10 = os.path.join(env.LOCAL, "linker_v10")
PM_F = os.path.join(L10, "v15r15_mirror.csv"); PE_F = os.path.join(L10, "v15r15_ev.csv")
L7_F = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
BASEPKL = os.path.join(env.LOCAL, "linker_v3", "base.pkl")
P_B4 = .5; P_TN8 = .8; ALPHA = .025; REPRO_TOL = 1e-6
ARMS = ["CUR", "R2", "R3", "R3B", "R3BT"]
rec_s = lambda x: "%s_%03d" % (x[0], int(x[1]))
rec_t = lambda s: (s[:3], int(s[4:7]))


def fold_sets(F, k):
    val = set((f, int(d)) for f, d in F[(F.fold == k) & (F.role == "val")][["farm", "num"]].values)
    dele = set((f, int(d)) for f, d in F[(F.fold == k) & (F.role == "del")][["farm", "num"]].values)
    return val, dele


# ------------------------------------------------------------------ part 1: RS2 BASE refit (reproduction) + MASK-world OOF
def mask_fold(j, F, raw, ct, phc, wmap, L0):
    path = os.path.join(CK, "rs3m_fold%d.pkl" % j)
    if os.path.exists(path):
        log("fold %d: checkpoint exists" % j); return
    t0 = time.time()
    val, dele = fold_sets(F, j)
    labrec = set((f, int(d)) for f, d in zip(L0.farm, L0.day))
    trT = sorted(r for r in labrec if r not in val and r not in dele)
    vis_tr = lambda r: (lambda x: x in raw.train and x not in val and x not in dele)
    TR = RS2.build_rows(raw, trT, val | dele, True, vis_tr, ct, phc, [])

    def vis_va(r):                       # = RS2.base_fold vis_va
        def v(x):
            if x in dele:
                return False
            if x in val and (x[0] != r[0] or x[1] >= r[1]):
                return False
            if x in raw.test and x[0] != r[0]:
                return False
            return True
        return v

    def vis_vm(r):                       # MASK version: test records never visible
        v0 = vis_va(r)
        return lambda x: v0(x) and x in raw.train
    VA = RS2.build_rows(raw, sorted(val), dele, False, vis_va, ct, phc, [])
    stM = []
    VM = RS2.build_rows(raw, sorted(val), dele, True, vis_vm, ct, phc, stM)
    assert VM.row_id.is_unique and set(VM.row_id) == set(VA.row_id)
    S = pd.read_pickle(os.path.join(RS2.CK, "rs2b_fold%d.pkl" % j)).set_index("row_id")
    assert set(S.index) == set(VA.row_id)
    w = TR.row_id.map(wmap).values.astype(float); assert np.isfinite(w).all()
    out = VM[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True); out["fold"] = j
    rep = {}
    for s in SEEDS:
        p = TF1.fit_base(TR, ct, phc, w, s)
        va = pd.Series(p(VA), index=VA.row_id.values)
        rep[s] = float(np.max(np.abs(va.loc[S.index].values - S["r_%d" % s].values)))
        out["m_%d" % s] = p(VM)
        out["va_%d" % s] = va.loc[out.row_id].values
    assert np.isfinite(out[["m_7", "m_101"]].values).all()
    out.to_pickle(path)
    SM = pd.DataFrame(stM); SM.assign(fold=j).to_csv(os.path.join(CK, "rs3_chains_vm_fold%d.csv" % j), index=False)
    log("fold %d: TR rows %d, VM rows %d (records %d; depth mean %.2f, c1 != prev number %d) | refit reproduction of stored RS2 val "
        "preds max|diff| s7 %.2e s101 %.2e | MASK vs VA BASE RMS diff s7 %.4f s101 %.4f | %.0fs"
        % (j, len(TR), len(out), len(SM), SM.depth.mean(), int((~SM.c1_is_prevnum.astype(bool)).sum()), rep[7], rep[101],
           rm(out.m_7 - out.va_7), rm(out.m_101 - out.va_101), time.time() - t0))


def run_mask(folds):
    os.makedirs(CK, exist_ok=True)
    L0 = pd.read_pickle(os.path.join(EVP.AUD, "lab_feats.pkl")); C = json.load(open(os.path.join(EVP.AUD, "cols.json")))
    ct, phc = C["ct"], C["phc"]; wmap = dict(zip(L0.row_id, L0.w))
    F = pd.read_csv(EVF); raw = RS2.Raw()
    log("restitch linker links p >= %.1f: %d (RS2 file)" % (RS2.P_MIN, len(raw.link)))
    for j in folds:
        mask_fold(j, F, raw, ct, phc, wmap, L0)


# ------------------------------------------------------------------ part 2 helpers
def in23_eve(X):
    X = X.assign(rec=X.farm + "_" + X.day.map("{:03d}".format))
    return X[X.hour == 23].drop_duplicates("rec").set_index("rec").in_temp, X[X.hour >= 18].groupby("rec").in_temp.mean()


def b4_table(CT, picks, ok, IN23, EVE, recs):
    """CT rows of records in recs: ch_in23 / ch_eve <- pick content when p0 >= .5 and ok(rec, pick) and the pick has an hour-23 row."""
    CT = CT.copy().reset_index(drop=True)
    ch = CT.ch_in23.values.astype(float).copy(); ce = CT.ch_eve.values.astype(float).copy()
    n_cand = n_rep = n_block = 0
    for i, (f, d) in enumerate(zip(CT.farm.values, CT.day.values)):
        r = (f, int(d))
        if r not in recs or rec_s(r) not in picks:
            continue
        pv, p0 = picks[rec_s(r)]
        if not isinstance(pv, str) or not p0 >= P_B4:
            continue
        n_cand += 1
        if not ok(r, rec_t(pv)) or pv not in IN23.index:
            n_block += 1; continue
        ch[i] = IN23[pv]; ce[i] = EVE.get(pv, np.nan); n_rep += 1
    CT["ch_in23"] = ch; CT["ch_eve"] = ce
    return CT, n_cand, n_rep, n_block


def report():
    F = pd.read_csv(EVF); folds = sorted(F.fold.unique())
    R = pd.read_pickle(os.path.join(EVP.CK, "ev_all.pkl"))
    RS = pd.concat([pd.read_pickle(os.path.join(RS2.CK, "rs2b_fold%d.pkl" % k)) for k in folds], ignore_index=True)
    MK = pd.concat([pd.read_pickle(os.path.join(CK, "rs3m_fold%d.pkl" % k)) for k in folds], ignore_index=True)
    assert RS.row_id.is_unique and MK.row_id.is_unique and len(RS) == len(R) == len(MK)
    # refit reproduction gate (pre-fixed): refit models must reproduce the stored RS2 val predictions
    chk = MK.merge(RS[["row_id", "r_7", "r_101"]], on="row_id")
    rep = max(float(np.max(np.abs(chk["va_%d" % s] - chk["r_%d" % s]))) for s in SEEDS)
    log("RS2 BASE refit reproduction (all folds): max |diff| %.2e (tolerance %.0e)" % (rep, REPRO_TOL))
    if not rep < REPRO_TOL:
        log("STOP: refitted RS2 BASE does not reproduce the stored RS2 predictions -> MASK OOF would come from a different BASE"); return
    R = R.merge(RS[["row_id", "r_7", "r_101"]], on="row_id", how="left").merge(MK[["row_id", "m_7", "m_101"]], on="row_id", how="left")
    assert R[["r_7", "r_101", "m_7", "m_101"]].notna().all().all()
    # corrector training sources
    st = pd.concat([pd.read_csv(os.path.join(EVP.OOF, "DIAG10_%d.csv" % k)) for k in range(10)], ignore_index=True)
    assert set(st.row_id) == set(R.row_id)
    MKi = MK.set_index("row_id")
    stm = st.copy()
    for s in SEEDS:
        stm["base_REF_%d" % s] = MKi.loc[st.row_id, "m_%d" % s].values
    log("corrector source: DIAG10 OOF base vs RS2 MASK OOF: RMS diff s7 %.4f s101 %.4f | RMSE (all 9600 rows) DIAG10 %.4f / %.4f, "
        "RS2-MASK %.4f / %.4f, RS2-VA(stored val preds) %.4f / %.4f"
        % (rm(st.base_REF_7 - stm.base_REF_7), rm(st.base_REF_101 - stm.base_REF_101), rm(st.base_REF_7 - st.sub_temp),
           rm(st.base_REF_101 - st.sub_temp), rm(stm.base_REF_7 - st.sub_temp), rm(stm.base_REF_101 - st.sub_temp),
           rm(R.r_7 - R.sub_temp), rm(R.r_101 - R.sub_temp)))
    # inputs / chain
    D0 = os.path.join(env.ROOT, u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
    tX = V18.ident(pd.read_csv(os.path.join(D0, "train_X.csv"))); sX = V18.ident(pd.read_csv(os.path.join(D0, "test_X.csv")))
    recs_all = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
    trainrecs = set((f, int(d)) for f, d in zip(tX.farm, tX.day)); testrecs = set((f, int(d)) for f, d in zip(sX.farm, sX.day))
    labrecs = set((f, int(d)) for f, d in zip(st.farm, st.day))
    # v22 anchors (v7, as EVP)
    Lk = pd.read_csv(EVP.LK)
    Lk = Lk[Lk.prev_is_labelled.astype(bool) & (Lk.p >= .8) & (Lk.steps_to_anchor == 1) & Lk.anchor_st23.notna() & (Lk.dT_air.abs() <= 1.5)]
    AN_all = dict(zip(Lk.rec, Lk.anchor_st23)); PREV = dict(zip(Lk.rec, Lk.prev))
    # TN8 (tb1r EV definition)
    L7 = pd.read_csv(L7_F, encoding="utf-8-sig")
    v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
    PE = pd.read_csv(PE_F); PM = pd.read_csv(PM_F)
    TN8 = {r: pv for r, pv, p0, lb in zip(PE.rec, PE.prev0, PE.p0, PE.lab0) if r not in v7set and bool(lb) and isinstance(pv, str) and p0 >= P_TN8}
    BP = pickle.load(open(BASEPKL, "rb")); rb = BP["recs"].reset_index(drop=True)
    ST23 = {"%s_%03d" % (f, int(d)): float(BP["ST"][i, 23]) for i, (f, d) in enumerate(zip(rb.farm, rb.day))}
    PEd = {r: (pv, p0) for r, pv, p0 in zip(PE.rec, PE.prev0, PE.p0)}
    PMd = {r: (pv, p0) for r, pv, p0 in zip(PM.rec, PM.prev0, PM.p0)}
    # diagnostic: mirror picks (p0 >= .5) whose top-6 candidate list contains a test record
    ntc = 0
    for _, row in PM.iterrows():
        if isinstance(row.prev0, str) and row.p0 >= P_B4 and any(isinstance(row["prev%d" % i], str) and rec_t(row["prev%d" % i]) in testrecs for i in range(6)):
            ntc += 1
    log("v7 anchor list %d | v7 set %d | TN8 candidates (EV file, before fold drop) %d | mirror picks p0 >= .5: %d, of which top-6 contains a "
        "test record %d (prev0 itself test: %d)" % (len(AN_all), len(v7set), len(TN8), int(((PM.p0 >= P_B4) & PM.prev0.notna()).sum()), ntc,
                                                      int(sum(isinstance(p, str) and q >= P_B4 and rec_t(p) in testrecs for p, q in zip(PM.prev0, PM.p0)))))
    Rk = []; tot = dict(an=0, an_drop=0, tn=0, tn_drop=0, trB_c=0, trB_r=0, trB_b=0, vaB_c=0, vaB_r=0, vaB_b=0)
    for k in folds:
        val, dele = fold_sets(F, k)
        excl = val | dele; exk = {rec_s(x) for x in excl}
        Q = R[R.fold == k].reset_index(drop=True).copy()
        key = (Q.farm + "_" + Q.day.map("{:03d}".format)).values; h = Q.hour.values; early = h <= 7
        AN = {}
        for r in set(key):
            if r in AN_all:
                if PREV[r] in exk:
                    tot["an_drop"] += 1
                else:
                    AN[r] = AN_all[r]
        TNk = {}
        for r in set(key):
            if r in TN8:
                if rec_t(TN8[r]) in excl:
                    tot["tn_drop"] += 1
                else:
                    TNk[r] = ST23[TN8[r]]
        assert not (set(AN) & set(TNk)) and all(np.isfinite(v) for v in TNk.values())
        tot["an"] += len(AN); tot["tn"] += len(TNk)
        CTm, CTv = EVP.chain_tables(tX, sX, recs_all, dele, val, False)
        # B4 tables
        Xm = tX[[(f, d) not in excl for f, d in zip(tX.farm, tX.day)]]
        Xv = pd.concat([tX[[(f, d) not in dele for f, d in zip(tX.farm, tX.day)]], sX])
        I23m, EVm = in23_eve(Xm); I23v, EVv = in23_eve(Xv)
        ok_tr = lambda r, x: x in trainrecs and x not in excl
        ok_va = lambda r, x: x not in dele and ((x in trainrecs and x not in val) or (x[0] == r[0] and x[1] < r[1]))
        trlab = {r for r in labrecs if r not in excl}
        CTmB, c1, r1, b1 = b4_table(CTm, PMd, ok_tr, I23m, EVm, trlab)
        CTvB, c2, r2, b2 = b4_table(CTv, PEd, ok_va, I23v, EVv, val)
        for a_, b_ in (("trB_c", c1), ("trB_r", r1), ("trB_b", b1), ("vaB_c", c2), ("vaB_r", r2), ("vaB_b", b2)):
            tot[a_] += b_
        Qc = Q.merge(CTv, on=["farm", "day"], how="left"); Qc["jump"] = Qc.in_temp - Qc.ch_in23
        Qb = Q.merge(CTvB, on=["farm", "day"], how="left"); Qb["jump"] = Qb.in_temp - Qb.ch_in23
        assert (Qc.row_id.values == Q.row_id.values).all() and (Qb.row_id.values == Q.row_id.values).all()
        ntr = int(((st.hour <= 7).values & np.array([(f, d) not in excl for f, d in zip(st.farm, st.day)])).sum())
        M_cur = EVP.corrector_preds(st, CTm, excl, Qc)
        M_r3 = EVP.corrector_preds(stm, CTm, excl, Qc)
        M_r3b = EVP.corrector_preds(stm, CTmB, excl, Qb)
        spec = {"CUR": ("a", M_cur, Qc, False), "R2": ("r", M_cur, Qc, False), "R3": ("r", M_r3, Qc, False),
                "R3B": ("r", M_r3b, Qb, False), "R3BT": ("r", M_r3b, Qb, True)}
        for arm, (t, mods, QQ, tn) in spec.items():
            for s, fam in CELLS:
                p = EVP.w40(Q["%s_%d" % (t, s)].values, Q.st_codex_REF.values, Q["st_pfn_" + fam].values, Q.in_temp.values)
                Qk = QQ.assign(p=p, p_air=p - QQ.in_temp.values)
                q = p.copy(); q[early] = p[early] + .5 * mods[(s, fam)].predict(Qk.loc[early, V18.FEAT])
                c = "%d%s" % (s, fam)
                Q["blend_%s_%s" % (arm, c)] = p; Q["v18_%s_%s" % (arm, c)] = q
                Q["fin_%s_%s" % (arm, c)] = EVP.anchor_apply(key, h, q, {**AN, **TNk} if tn else AN, "v22")
        Q["anch_v7"] = [x in AN for x in key]; Q["anch_tn8"] = [x in TNk for x in key]
        Rk.append(Q)
        log("fold %d: corrector training rows %d (hours 0-7, records outside val/del) | v7 anchors %d, TN8 %d | B4 training picks "
            "p0>=.5 %d replaced %d blocked %d | B4 val picks %d replaced %d blocked %d"
            % (k, ntr, len(AN), len(TNk), c1, r1, b1, c2, r2, b2))
    R = pd.concat(Rk, ignore_index=True)
    log("totals: %s" % tot)
    # reproduction checks
    d_cur = max(float(np.max(np.abs(R["fin_CUR_%d%s" % c] - R["v22_a_%d%s" % c]))) for c in CELLS)
    R2s = pd.read_pickle(os.path.join(RS2.CK, "rs2_ev_all.pkl")).set_index("row_id").loc[R.row_id]
    d_r2 = max(float(np.max(np.abs(R["fin_R2_%d%s" % c].values - R2s["xv22_r_%d%s" % c].values))) for c in CELLS)
    log("reproduction: CUR vs stored ev_all v22_a cells max|diff| %.2e | R2 vs RS2 rs2_ev_all xv22_r cells max|diff| %.2e" % (d_cur, d_r2))
    for arm in ARMS:
        for stg in ("blend", "v18", "fin"):
            R["%s_%s" % (stg, arm)] = np.mean([R["%s_%s_%d%s" % (stg, arm, s, f)] for s, f in CELLS], 0)
    R["base_CUR"] = np.mean([R.a_7, R.a_101], 0); R["base_RS2"] = np.mean([R.r_7, R.r_101], 0); R["base_RS2mask"] = np.mean([R.m_7, R.m_101], 0)
    R.to_pickle(os.path.join(CK, "rs3_ev_all.pkl"))
    # scoring
    key = (R.farm + "_" + R.day.map("{:03d}".format)).values; y = R.sub_temp.values
    rough = np.array([(f, d) in TF1.RD for f, d in zip(R.farm, R.day)]); p2 = (R.day >= 179).values
    cold = np.nan_to_num(R.in_temp.values, nan=99.0) <= 8; base = ~rough
    scopes = [("pass2", base & p2), ("all", base)]
    for f in ("F13", "F47"):
        scopes += [(f + "_pass2", base & p2 & (R.farm == f).values), (f + "_all", base & (R.farm == f).values)]
    scopes += [("cold_pass2", base & p2 & cold), ("cold_all", base & cold), ("pass2_withrough", p2), ("all_withrough", np.ones(len(R), bool))]
    log("empty scopes (skipped): %s" % [sc for sc, m in scopes if not m.any()])
    scopes = [(sc, m) for sc, m in scopes if m.any()]
    SC = dict(scopes)
    early = R.hour.values <= 7
    # correction magnitude on val early rows
    for arm in ARMS[:4]:
        cc = (R["v18_" + arm] - R["blend_" + arm]).values[early & base]
        log("  mean corrector change on val rows h0-7 (cell mean) %-5s: mean %+.4f RMS %.4f" % (arm, cc.mean(), rm(cc)))
    pairs = [("R2", "CUR"), ("R3", "CUR"), ("R3B", "CUR"), ("R3BT", "CUR"), ("R3", "R2"), ("R3B", "R3"), ("R3BT", "R3B")]
    rows = []
    for new, ref in pairs:
        for sc, m in scopes:
            a_, b_ = rm((R["fin_" + ref] - y)[m]), rm((R["fin_" + new] - y)[m])
            cl = [100 * (rm((R["fin_%s_%d%s" % (new, s, f)] - y)[m]) / rm((R["fin_%s_%d%s" % (ref, s, f)] - y)[m]) - 1) for s, f in CELLS]
            P, n = boot(key, y, R["fin_" + ref].values, R["fin_" + new].values, m)
            rows.append(dict(new=new, ref=ref, scope=sc, n_rec=n, n_rows=int(m.sum()), rmse_ref=a_, rmse_new=b_, d_pct=100 * (b_ / a_ - 1),
                             c7A=cl[0], c7B=cl[1], c101A=cl[2], c101B=cl[3], cells_better=sum(x < 0 for x in cl), P_worse=P))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "rs3_arms_v1.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    log("\n================ final stage (after anchors): new vs ref, RMSE of cell-mean, cell %% changes, P(new worse) ================")
    log(T.round(4).to_string(index=False))
    # stage table (context): absolute RMSE per arm / stage
    srow = []
    for sc, m in scopes:
        d = dict(scope=sc, n_rec=len(set(key[m])), BASE_CUR=rm((R.base_CUR - y)[m]), BASE_RS2=rm((R.base_RS2 - y)[m]), BASE_RS2mask=rm((R.base_RS2mask - y)[m]))
        for arm in ARMS:
            if arm != "R3BT":
                d["v18_" + arm] = rm((R["v18_" + arm] - y)[m])
            d["fin_" + arm] = rm((R["fin_" + arm] - y)[m])
        srow.append(d)
    TS = pd.DataFrame(srow); TS.to_csv(os.path.join(OUT, "rs3_stages_v1.csv"), index=False)
    log("\n================ RMSE by stage (cell-mean; BASE = seed mean) ================"); log(TS.round(4).to_string(index=False))
    # pass-2 records improved / worsened vs CUR
    m = SC["pass2"]; recrow = []
    for arm in ARMS[1:]:
        dS = pd.Series(((R["fin_" + arm] - y) ** 2 - (R.fin_CUR - y) ** 2)[m]).groupby(key[m]).sum()
        recrow.append(dict(arm=arm, records=len(dS), improved=int((dS < -1e-9).sum()), worsened=int((dS > 1e-9).sum()),
                           unchanged=int((dS.abs() <= 1e-9).sum()), dSSE_sum=float(dS.sum()),
                           top_gains=str((-dS).sort_values(ascending=False).head(3).round(2).to_dict()),
                           top_losses=str(dS.sort_values(ascending=False).head(3).round(2).to_dict())))
        if arm == "R3BT":
            dS.rename("dSSE_R3BT_vs_CUR").to_csv(os.path.join(OUT, "rs3_pass2_records_v1.csv"))
    TR_ = pd.DataFrame(recrow); log("\n================ pass-2 records vs CUR (cell-mean SSE) ================"); log(TR_.to_string(index=False))
    # cells table for the primary verdict
    crow = []; ok = True
    for sc in ("pass2", "all"):
        mm = SC[sc]
        for s, f in CELLS:
            c = "%d%s" % (s, f); a_ = rm((R["fin_CUR_" + c] - y)[mm]); b_ = rm((R["fin_R3BT_" + c] - y)[mm])
            crow.append(dict(scope=sc, cell=c, CUR=a_, R3BT=b_, d_pct=100 * (b_ / a_ - 1), better=b_ < a_)); ok &= b_ < a_
    TC = pd.DataFrame(crow); TC.to_csv(os.path.join(OUT, "rs3_cells_v1.csv"), index=False)
    log("\n================ PRIMARY: R3BT vs CUR per cell (final stage) ================"); log(TC.round(4).to_string(index=False))
    P2 = boot(key, y, R.fin_CUR.values, R.fin_R3BT.values, SC["pass2"])[0]; Pa = boot(key, y, R.fin_CUR.values, R.fin_R3BT.values, SC["all"])[0]
    log("P(R3BT worse than CUR): pass-2 %.4f, all %.4f" % (P2, Pa))
    log("VERDICT RS3 (R3BT vs CUR): %s (all 4 cells better on pass-2 and all: %s; P pass-2 < %.3f: %s)"
        % ("PASS" if ok and P2 < ALPHA else "FAIL", ok, ALPHA, P2 < ALPHA))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--mask":
        run_mask([int(x) for x in a[1].split(",")])
    elif a and a[0] == "--report":
        report()
    else:
        run_mask(range(8)); report()
