# -*- coding: utf-8 -*-
"""TF1: chunk-start (deleted previous record) feature fill for the temperature BASE member (2026-10-11 집 클로드).

Background (audit_temp a2-a7, catalog 6.613): every evaluation chunk is preceded by a DELETED record, so in the 8 chunk-start
evaluation records (F13_185/200/220/235, F47_183/198/218/233) BASE cross-day features are NaN or odd at early hours
(*_r24m/_r24s NaN at 0-4 h, in_temp_lag1-3 / d1 / d3 / out_temp_lag1 / rad_eff_lag1 NaN at 0-2 h, act_*_hsince inflated).
DIAG10 never reproduces this (neighbour-day inputs are kept).  Separately the v18 chain corrector gives chunk-start records a
lit1_in23 taken from the record TWO numbers earlier, while in training (MASK) such a case is NaN.

Everything below is FIXED BEFORE RUNNING.

Validator DS (deleted start):
  folds = DIAG10 (anal_q1_errors.diag_folds, 10 folds of 5 sorted training days, split_mask +-1 day buffer).  Training rows,
  weights and features = the production MASK world (audit_temp/lab_feats.pkl, w = train_flags_v6 .2/.2) - identical to DIAG10.
  Validation rows: for every maximal run of consecutive validation days (per farm) the record right before the run is blanked
  (all train_X inputs NaN, as the deleted record in evaluation; it is a buffer day so never a training row); validation-row
  BASE and physics features are recomputed in that world (test_X inputs NaN as in MASK).  "First records" = the first record of
  every run (its previous record is unavailable: blanked, a test day, or absent).
  BASE = production recipe (0.65 (physics linear + LGB huber residual) + 0.25 Ridge + 0.10 Nystroem; make_submission_v3
  factories), BASE seeds 7 and 101 (one fit per seed per fold, the three feature versions predicted by the same model).
  Blend cells: W40G-S formula with the stored DIAG10 out-of-fold CODEX_REF and TabPFN families A / B (tt1_ckpt), BASE replaced by
  this script's BASE -> 4 cells = BASE seed {7, 101} x PFN family {A, B}.  (BASE-only RMSE also reported.)
  Scoring: rows of first records; pass 2 = record day >= 179; the 30 strict CO2-rough days (rw1_rough_days_v1) are EXCLUDED from
  scoring (temperature protocol 10-10); with-rough numbers reported for context.  All validation rows reported for context.

Treatments on first-record rows (test-time only; training features untouched):
  (o) oracle: standard DIAG10 features (neighbour record kept) - context only, not available in evaluation.
  (a) current: features as computed with the previous record deleted.
  (b) within-record causal fill, only on first-record rows:
      in_temp_lag{1,2,3}, out_temp_lag1, rad_eff_lag1 where NaN -> the same input at max(h - k, 0) of the same record
        (i.e. hour-0 value carried);  in_temp_d1 / in_temp_d3 where NaN -> in_temp - (filled lag1 / lag3);
      every *_r24m / *_r24s where NaN -> expanding mean / std (ddof 1, single value -> 0) of the same input over hours 0..h
        of the record (for h >= 5 the computed value already equals this);
      every act_*_hsince_on / _chg -> min(value, h + 1) (clock restarted at the record boundary; same as audit a5).
  (c) NaN-augmented training: out of scope, skipped.
  (d) v18 chain corrector: lit1_in23 / lit2_in23 = NaN unless the literal previous / second-previous existing record is
      exactly day-1 / day-2 (training-consistent) vs current.  Consistency fix: effect reported only, no verdict.

PASS rule for (b) over (a) (pre-fixed):
  first-record rows (rough days excluded): blend RMSE (b) < (a) in ALL 4 cells, separately for pass-2 rows AND all-pass rows;
  AND record-level bootstrap (resample first records pooled over folds with replacement, 4000 draws, seed 0, cell-mean
  predictions) P(RMSE_b >= RMSE_a) < .025 for pass-2 rows AND for all-pass rows.  Single candidate (no Bonferroni).
  Exploratory (no verdict): (b) restricted to one feature group (lag / r24 / event).

Evaluation-row change (approximation of the final pipeline): BASE refit on the full MASK training frame with production seeds
  7 / 101 / 2024; delta BASE (b - a) on the chunk-start records -> P9' = temp_candidate_v13 + 0.4 * delta BASE (other members
  kept) -> v18 corrector re-run (reproduces temp_candidate_v18 when unchanged; checked) -> v22 anchor re-run (reproduces
  temp_candidate_v22; checked).  Same for (d) and (b)+(d).  Per-record change reported.  No submission file is produced.

Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u tf1_chunkstart/tf1_chunkstart_fill_v1.py
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time
import numpy as np, pandas as pd

PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도"
sys.path.insert(0, os.path.join(PK, "code"))
import common  # noqa: E402  (identical file in 연구실/클로드/code and the package)
import harness  # noqa: E402
from temp_mask_v1 import build_world  # noqa: E402
from anal_q1_errors import diag_folds  # noqa: E402
from make_submission_v3 import m_lgbh, m_ridge, m_nys  # noqa: E402
import make_temp_v18_chain as V18  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

OUT = os.path.join(env.ROOT, u"집", u"클로드", "research", "tf1_chunkstart")
AUD = os.path.join(env.ROOT, u"집", u"클로드", "research", "audit_temp")
OOF = os.path.join(env.LOCAL, "tt1_ckpt")
ANCH = os.path.join(env.LOCAL, "linker_v7", "eval60_confident_1step_anchor_v7.csv")
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
SEEDS_V = (7, 101); SEEDS_E = (7, 101, 2024); FAMS = ("A", "B")
NBOOT, ALPHA = 4000, .025
INPUTS = [c for c in common.USABLE]
ORIG = common.load_raw
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- worlds
def ds_loader(blank):
    def ld():
        tX, ty, sX = ORIG()
        tX = tX.copy(); sX = sX.copy()
        sX[INPUTS] = np.nan                                  # MASK world
        m = np.array([(f, d) in blank for f, d in zip(tX.farm, tX.day)])
        tX.loc[m, [c for c in tX.columns if c not in ("row_id", "farm", "day", "hour", "t")]] = np.nan
        return tX, ty, sX
    return ld


def world(loader):
    common.load_raw = loader; harness._CACHE.clear()
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG; harness._CACHE.clear()
    return lab, ct, phc


def runs(days):
    days = sorted(days); out = []; cur = [days[0]]
    for d in days[1:]:
        if d == cur[-1] + 1:
            cur.append(d)
        else:
            out.append(cur); cur = [d]
    out.append(cur)
    return out


# ---------------------------------------------------------------- fill (b)
def raw_panel():
    tX, _, sX = ORIG()
    a = pd.concat([tX, sX], ignore_index=True)
    a = a[a.farm.isin(common.TARGET_FARMS)].copy()
    a = common.add_derived(a)
    return a.set_index(["farm", "day", "hour"])


def fill_b(F, recs, ct, RAW, groups=("lag", "r24", "event")):
    """F: frame with farm/day/hour + ct; recs: set of (farm, day) to fill.  Returns a copy."""
    F = F.copy()
    lagspec = [("in_temp_lag1", "in_temp", 1), ("in_temp_lag2", "in_temp", 2), ("in_temp_lag3", "in_temp", 3),
               ("out_temp_lag1", "out_temp", 1), ("rad_eff_lag1", "rad_eff", 1)]
    r24 = [c for c in ct if c.endswith("_r24m") or c.endswith("_r24s")]
    ev = [c for c in ct if "_hsince" in c]
    sel = np.array([(f, d) in recs for f, d in zip(F.farm, F.day)])
    for (f, d), g in F[sel].groupby(["farm", "day"]):
        ix = g.index; h = g.hour.values.astype(int)
        R = RAW.loc[(f, d)].reindex(range(24))
        if "lag" in groups:
            for c, src, k in lagspec:
                if c in ct:
                    x = R[src].values; fillv = x[np.maximum(h - k, 0)]
                    cur = F.loc[ix, c].values; F.loc[ix, c] = np.where(np.isnan(cur), fillv, cur)
            x = R["in_temp"].values
            for c, k in (("in_temp_d1", 1), ("in_temp_d3", 3)):
                if c in ct:
                    cur = F.loc[ix, c].values
                    F.loc[ix, c] = np.where(np.isnan(cur), x[h] - x[np.maximum(h - k, 0)], cur)
        if "r24" in groups:
            for c in r24:
                src = c[:-5]; x = R[src]
                st = x.expanding().mean() if c.endswith("m") else x.expanding().std().fillna(0.0)
                st = st.values[h]
                cur = F.loc[ix, c].values; F.loc[ix, c] = np.where(np.isnan(cur), st, cur)
        if "event" in groups:
            for c in ev:
                cur = F.loc[ix, c].values
                F.loc[ix, c] = np.where(np.isnan(cur), cur, np.minimum(cur, h + 1.0))
    return F


# ---------------------------------------------------------------- BASE
def fit_base(tr, ct, phc, w, seed):
    y = tr.sub_temp.values
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    lin = LinearRegression().fit(imp.transform(tr[phc]), y, sample_weight=w)
    btr = lin.predict(imp.transform(tr[phc]))
    g = m_lgbh(seed).fit(tr[ct], y - btr, sample_weight=w)
    r = m_ridge(seed).fit(tr[ct], y, ridge__sample_weight=w)
    n = m_nys(seed).fit(tr[ct], y, ridge__sample_weight=w)

    def pred(X):
        return .65 * (lin.predict(imp.transform(X[phc])) + g.predict(X[ct])) + .25 * r.predict(X[ct]) + .10 * n.predict(X[ct])
    return pred


def w40(base, codex, pfn, it):
    g = np.where(np.isnan(it), 1.0, np.clip((it - 8.0) / 2.0, 0.0, 1.0))
    return .4 * base + (.2 + .4 * (1 - g)) * codex + .4 * g * pfn


# ---------------------------------------------------------------- part 1: DS validator
def part1(L0, ct, phc, RAW):
    w = L0.w.values.astype(float)
    folds = diag_folds(L0)
    oof = pd.concat([pd.read_csv(os.path.join(OOF, "DIAG10_%d.csv" % k)) for k in range(10)], ignore_index=True).set_index("row_id")
    rows = []
    t0 = time.time()
    allrecs = set(zip(L0.farm, L0.day))
    for k, fd in enumerate(folds):
        path = os.path.join(OUT, "local_ckpt", "ds_fold%d.pkl" % k)
        if os.path.exists(path):
            rows.append(pd.read_pickle(path)); log("fold %d from checkpoint" % k); continue
        trm, vam = common.split_mask(L0, fd)
        blank, first = set(), set()
        for f, ds in fd.items():
            if not ds:
                continue
            for r in runs(ds):
                first.add((f, r[0]))
                if (f, r[0] - 1) in allrecs:
                    blank.add((f, r[0] - 1))
        assert not any(trm[np.array([(f, d) in blank for f, d in zip(L0.farm, L0.day)])])
        LD, ct2, phc2 = world(ds_loader(blank))
        assert ct2 == ct and phc2 == phc
        LD = LD.set_index("row_id").loc[L0.row_id[vam]].reset_index()
        VO = L0[vam].reset_index(drop=True)
        assert (LD.row_id.values == VO.row_id.values).all()
        VA = VO.copy(); VA[ct + phc] = LD[ct + phc].values           # (a) deleted-start features
        VB = fill_b(VA, first, ct, RAW)
        VG = {g: fill_b(VA, first, ct, RAW, (g,)) for g in ("lag", "r24", "event")}
        out = VO[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy()
        out["fold"] = k
        out["first"] = [(f, d) in first for f, d in zip(out.farm, out.day)]
        out["newblank"] = [(f, d - 1) in blank for f, d in zip(out.farm, out.day)]
        # feature diagnostics on first-record rows
        fr = out["first"].values
        nanO = VO.loc[fr, ct].isna().sum(); nanA = VA.loc[fr, ct].isna().sum(); nanB = VB.loc[fr, ct].isna().sum()
        dif = (~np.isclose(VO.loc[~fr, ct].astype(float).values, VA.loc[~fr, ct].astype(float).values, equal_nan=True)).any(1).sum()
        out.attrs["diag"] = dict(nan_extra_A=(nanA - nanO)[(nanA - nanO) > 0].to_dict(), nan_extra_B=(nanB - nanO)[(nanB - nanO) > 0].to_dict(),
                                 nonfirst_rows_changed=int(dif), n_first=len(first), n_blank=len(blank))
        for s in SEEDS_V:
            p = fit_base(L0[trm], ct, phc, w[trm], s)
            out["o_%d" % s] = p(VO); out["a_%d" % s] = p(VA); out["b_%d" % s] = p(VB)
            for g, VGg in VG.items():
                out["b%s_%d" % (g, s)] = p(VGg)
        st = oof.loc[out.row_id]
        for c in ("codex_REF", "pfn_A", "pfn_B", "base_REF_7", "base_REF_101", "fold"):
            out["st_" + c] = st[c].values
        os.makedirs(os.path.dirname(path), exist_ok=True); out.to_pickle(path)
        rows.append(out)
        log("fold %d: val rows %d, first records %d (newly blanked %d), non-first val rows with changed features %d | %.0fs"
            % (k, len(out), len(first), len(blank), dif, time.time() - t0))
        log("   extra NaN cells in first-record rows (a): %s" % json.dumps({c: int(v) for c, v in out.attrs["diag"]["nan_extra_A"].items()}))
        log("   extra NaN cells after fill (b): %s" % json.dumps({c: int(v) for c, v in out.attrs["diag"]["nan_extra_B"].items()}))
    return pd.concat(rows, ignore_index=True)


def boot(D, ca, cb, nb=NBOOT):
    D = D.assign(sa=(D[ca] - D.sub_temp) ** 2, sb=(D[cb] - D.sub_temp) ** 2)
    G = D.groupby(["farm", "day"]).agg(sa=("sa", "sum"), sb=("sb", "sum"), n=("sa", "size"))
    rng = np.random.default_rng(0); idx = rng.integers(0, len(G), (nb, len(G)))
    sa, sb, n = G.sa.values[idx].sum(1), G.sb.values[idx].sum(1), G.n.values[idx].sum(1)
    return float(np.mean(np.sqrt(sb / n) >= np.sqrt(sa / n))), len(G)


def report1(R):
    log("\n================ PART 1: DS validator ================")
    log("stored-vs-refit BASE (standard features) correlation: seed7 %.4f, seed101 %.4f; RMS diff %.4f / %.4f"
        % (np.corrcoef(R.o_7, R.st_base_REF_7)[0, 1], np.corrcoef(R.o_101, R.st_base_REF_101)[0, 1],
           rm(R.o_7 - R.st_base_REF_7), rm(R.o_101 - R.st_base_REF_101)))
    log("fold numbering equals stored DIAG10 oof folds: %s" % bool((R.fold.values == R.st_fold.values).all()))
    R = R.copy()
    for s in SEEDS_V:
        for fam in FAMS:
            for t in ["o", "a", "b", "blag", "br24", "bevent"]:
                R["W_%s_%d%s" % (t, s, fam)] = w40(R["%s_%d" % (t, s)].values, R.st_codex_REF.values, R["st_pfn_" + fam].values, R.in_temp.values)
    cells = [(s, fam) for s in SEEDS_V for fam in FAMS]
    for t in ["o", "a", "b", "blag", "br24", "bevent"]:
        R["W_%s_mean" % t] = np.mean([R["W_%s_%d%s" % (t, s, fam)] for s, fam in cells], 0)
        R["B_%s_mean" % t] = np.mean([R["%s_%d" % (t, s)] for s in SEEDS_V], 0)
    rough = np.array([(f, d) in RD for f, d in zip(R.farm, R.day)])
    p2 = (R.day >= 179).values; fr = R["first"].values
    scopes = {"first_pass2": fr & p2 & ~rough, "first_all": fr & ~rough, "first_pass2_withrough": fr & p2,
              "first_all_withrough": fr, "first_newblank_all": fr & R.newblank.values & ~rough,
              "allval_all": ~rough, "allval_pass2": p2 & ~rough}
    tab = []
    for sc, m in scopes.items():
        D = R[m]
        for s, fam in cells:
            e = {t: rm(D["W_%s_%d%s" % (t, s, fam)] - D.sub_temp) for t in ["o", "a", "b", "blag", "br24", "bevent"]}
            eb = {t: rm(D["%s_%d" % (t, s)] - D.sub_temp) for t in ["o", "a", "b"]}
            tab.append(dict(scope=sc, cell="%d%s" % (s, fam), n_rows=int(m.sum()), n_rec=D.groupby(["farm", "day"]).ngroups,
                            W_o=e["o"], W_a=e["a"], W_b=e["b"], d_b_vs_a_pct=100 * (e["b"] / e["a"] - 1),
                            W_blag=e["blag"], W_br24=e["br24"], W_bevent=e["bevent"], BASE_o=eb["o"], BASE_a=eb["a"], BASE_b=eb["b"],
                            d_a_vs_o_pct=100 * (e["a"] / e["o"] - 1)))
    T = pd.DataFrame(tab); T.to_csv(os.path.join(OUT, "tf1_ds_cells.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    log(T.round(4).to_string(index=False))
    P = {}
    for sc in ("first_pass2", "first_all", "first_newblank_all", "first_pass2_withrough", "first_all_withrough"):
        D = R[scopes[sc]]
        P[sc], nrec = boot(D, "W_a_mean", "W_b_mean")
        pg = {g: boot(D, "W_a_mean", "W_b%s_mean" % g)[0] for g in ("lag", "r24", "event")}
        log("bootstrap %-22s records %3d  P(b worse than a) = %.4f | single groups lag %.3f r24 %.3f event %.3f | P(a worse than o) = %.4f"
            % (sc, nrec, P[sc], pg["lag"], pg["r24"], pg["event"], boot(D, "W_o_mean", "W_a_mean")[0]))
    c1 = all(T[(T.scope == sc)].W_b.lt(T[(T.scope == sc)].W_a).all() for sc in ("first_pass2", "first_all"))
    c2 = P["first_pass2"] < ALPHA and P["first_all"] < ALPHA
    verdict = "PASS" if (c1 and c2) else "FAIL"
    log("\nVERDICT (b) over (a): %s  [all 4 cells better pass-2 & all-pass: %s; P pass-2 %.4f, all-pass %.4f (< %.3f both: %s)]"
        % (verdict, c1, P["first_pass2"], P["first_all"], ALPHA, c2))
    # by hour (first records, rough excluded, cell mean)
    D = R[scopes["first_all"]]
    H = D.groupby("hour").apply(lambda g: pd.Series(dict(n=len(g), o=rm(g.W_o_mean - g.sub_temp), a=rm(g.W_a_mean - g.sub_temp),
                                                          b=rm(g.W_b_mean - g.sub_temp), dBASE_b_a=float((g.B_b_mean - g.B_a_mean).abs().mean()))))
    log("\nfirst-record rows by hour (cell-mean blend RMSE):"); log(H.round(3).T.to_string())
    PR = D.groupby(["farm", "day"]).apply(lambda g: pd.Series(dict(o=rm(g.W_o_mean - g.sub_temp), a=rm(g.W_a_mean - g.sub_temp),
                                                                   b=rm(g.W_b_mean - g.sub_temp))))
    PR.to_csv(os.path.join(OUT, "tf1_ds_first_records.csv"))
    log("first records: b better than a in %d of %d records" % (int((PR.b < PR.a).sum()), len(PR)))
    R.to_pickle(os.path.join(OUT, "local_ckpt", "ds_all.pkl"))
    return verdict, P


# ---------------------------------------------------------------- part 2: evaluation changes
def lit_fix(CT, records):
    CT = CT.copy()
    pos = {}
    for f in V18.FARMS:
        days = [d for ff, d in records if ff == f]
        for i, d in enumerate(days):
            pos[(f, d)] = (days[i - 1] if i >= 1 else None, days[i - 2] if i >= 2 else None)
    p1 = np.array([pos[(f, d)][0] for f, d in zip(CT.farm, CT.day)], dtype=object)
    p2 = np.array([pos[(f, d)][1] for f, d in zip(CT.farm, CT.day)], dtype=object)
    bad1 = np.array([p is None or p != d - 1 for p, d in zip(p1, CT.day)])
    bad2 = np.array([p is None or p != d - 2 for p, d in zip(p2, CT.day)])
    CT.loc[bad1, "lit1_in23"] = np.nan; CT.loc[bad2, "lit2_in23"] = np.nan
    return CT


def run_v18(P9, fixd):
    D0 = os.path.join(env.ROOT, u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
    tX = V18.ident(pd.read_csv(os.path.join(D0, "train_X.csv"))); sX = V18.ident(pd.read_csv(os.path.join(D0, "test_X.csv")))
    records = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
    sd = tX[V18.V].std()
    CTm = V18.chain_table(tX, records, sd); CTf = V18.chain_table(pd.concat([tX, sX]), records, sd)
    if fixd:
        m0 = CTm.copy(); CTm = lit_fix(CTm, records); CTf2 = lit_fix(CTf, records)
        trs = set(zip(tX.farm, tX.day)); ev = set(zip(sX.farm, sX.day))
        it = np.array([(f, d) in trs for f, d in zip(CTm.farm, CTm.day)]); ie = np.array([(f, d) in ev for f, d in zip(CTf.farm, CTf.day)])
        ch_tr = int(((m0.lit1_in23.notna() & CTm.lit1_in23.isna()) | (m0.lit2_in23.notna() & CTm.lit2_in23.isna()))[it].sum())
        ch_ev = CTf[ie & ((CTf.lit1_in23.notna() & CTf2.lit1_in23.isna()) | (CTf.lit2_in23.notna() & CTf2.lit2_in23.isna())).values]
        log("   (d) lit fix: training records changed %d | evaluation records changed %d: %s"
            % (ch_tr, len(ch_ev), ", ".join("%s_%03d" % (f, d) for f, d in zip(ch_ev.farm, ch_ev.day))))
        CTf = CTf2
    D = pd.concat([pd.read_csv(os.path.join(OOF, f)) for f in sorted(os.listdir(OOF)) if f.startswith("DIAG10_")], ignore_index=True)
    D = D.merge(CTm, on=["farm", "day"], how="left"); D["jump"] = D.in_temp - D.ch_in23
    D = D[D.hour <= 7].reset_index(drop=True)
    P9 = P9[["row_id", "sub_temp"]].rename(columns={"sub_temp": "p"})
    Q = V18.ident(sX[["row_id"] + V18.V]).merge(P9, on="row_id").merge(CTf, on=["farm", "day"], how="left")
    Q["jump"] = Q.in_temp - Q.ch_in23; Q["p_air"] = Q.p - Q.in_temp
    early = (Q.hour <= 7).values; corr = []
    import lightgbm as lgb
    for b in ("base_REF_7", "base_REF_101"):
        for fam in "AB":
            p = V18.w40(D, b, fam); Dk = D.assign(p=p, p_air=p - D.in_temp.values)
            m = lgb.LGBMRegressor(n_estimators=100, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
            m.fit(Dk[V18.FEAT], Dk.sub_temp.values - p); corr.append(m.predict(Q.loc[early, V18.FEAT]))
    c = np.mean(corr, 0)
    Q["pred"] = Q.p.values; Q.loc[early, "pred"] = Q.p.values[early] + .5 * c
    S = P9[["row_id"]].merge(Q[["row_id", "pred"]], on="row_id", how="left")
    return S.rename(columns={"pred": "sub_temp"})


def run_v22(T):
    C, CLIP = .36, 2.0
    BETA = np.array([0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566,
                     0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355])
    A = pd.read_csv(ANCH, encoding="utf-8-sig"); A = A[A.dT_air.abs() <= 1.5]; AN = dict(zip(A.row, A.anchor_st23))
    k = T.row_id.str[:7]; h = T.row_id.str[8:10].astype(int)
    q0 = T[h == 0].set_index(k[h == 0]).sub_temp
    new = T.sub_temp.values.copy()
    for i, (kk, hh) in enumerate(zip(k, h)):
        if kk in AN:
            new[i] = T.sub_temp.values[i] + BETA[hh] * np.clip(AN[kk] - C - q0[kk], -CLIP, CLIP)
    S = T[["row_id"]].copy(); S["sub_temp"] = new
    return S, set(AN)


def part2(L0, T0, ct, phc, RAW):
    log("\n================ PART 2: evaluation-row changes ================")
    tX, _, sX = ORIG()
    avail = set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day))
    starts = {(f, d) for f, d in zip(T0.farm, T0.day) if (f, d - 1) not in avail}
    log("chunk-start evaluation records:", sorted(starts))
    TB = fill_b(T0, starts, ct, RAW)
    chg = (~np.isclose(TB[ct].astype(float).values, T0[ct].astype(float).values, equal_nan=True)).any(1)
    log("evaluation rows with any filled feature: %d (all in chunk-start records: %s)"
        % (chg.sum(), bool(np.all([(f, d) in starts for f, d in zip(T0.farm[chg], T0.day[chg])]))))
    w = L0.w.values.astype(float); dB = []
    for s in SEEDS_E:
        t0 = time.time(); p = fit_base(L0, ct, phc, w, s); dB.append(p(TB) - p(T0)); log("  full BASE seed %d (%.0fs)" % (s, time.time() - t0))
    dB = np.mean(dB, 0)
    v13 = pd.read_csv(os.path.join(PK, "temp_candidate_v13.csv")); v18 = pd.read_csv(os.path.join(PK, "temp_candidate_v18.csv"))
    v22 = pd.read_csv(os.path.join(PK, "temp_candidate_v22.csv"))
    assert (v13.row_id.values == T0.row_id.values).all()
    # reproduction checks
    r18 = run_v18(v13, False); r22, anch = run_v22(r18)
    e18 = np.abs(r18.sub_temp.values - v18.sub_temp.values).max(); e22 = np.abs(r22.sub_temp.values - v22.sub_temp.values).max()
    log("reproduction: v18 max|diff| %.2e, v22 max|diff| %.2e" % (e18, e22))
    P9b = v13.copy(); P9b["sub_temp"] = v13.sub_temp.values + .4 * dB
    out = {"W40_b": P9b.sub_temp.values - v13.sub_temp.values}
    b18 = run_v18(P9b, False); out["v18_b"] = b18.sub_temp.values - r18.sub_temp.values
    out["v22_b"] = run_v22(b18)[0].sub_temp.values - r22.sub_temp.values
    d18 = run_v18(v13, True); out["v18_d"] = d18.sub_temp.values - r18.sub_temp.values
    out["v22_d"] = run_v22(d18)[0].sub_temp.values - r22.sub_temp.values
    bd18 = run_v18(P9b, True); out["v18_bd"] = bd18.sub_temp.values - r18.sub_temp.values
    out["v22_bd"] = run_v22(bd18)[0].sub_temp.values - r22.sub_temp.values
    E = T0[["row_id", "farm", "day", "hour"]].copy(); E["dBASE_b"] = dB
    for k, v in out.items():
        E[k] = v
    E["anchored"] = [r[:7] in anch for r in E.row_id]
    E.to_csv(os.path.join(OUT, "tf1_eval_changes_rows.csv"), index=False)
    cols = ["dBASE_b"] + list(out)
    g = E.groupby(["farm", "day"])
    PR = pd.concat({c: g[c].apply(lambda x: np.sqrt((x ** 2).mean())) for c in cols}, axis=1)
    PR["v22_bd_h0"] = g.apply(lambda x: float(x.v22_bd.values[x.hour.values == 0][0]))
    PR["v22_bd_mean"] = g.v22_bd.mean(); PR["anchored"] = g.anchored.first(); PR["start"] = [k in starts for k in PR.index]
    PR = PR[(PR[cols].abs() > 1e-9).any(axis=1)]
    PR.to_csv(os.path.join(OUT, "tf1_eval_changes_records.csv"))
    log("per-record RMS change (records with any change):"); log(PR.round(4).to_string())
    for c in cols:
        x = E[c].values; m = np.abs(x) > 1e-9
        log("  %-8s rows changed %4d | RMS over changed %.4f | mean %+.4f | max|.| %.3f | RMS over all 1440 %.4f"
            % (c, m.sum(), np.sqrt((x[m] ** 2).mean()) if m.any() else 0, x[m].mean() if m.any() else 0, np.abs(x).max(), np.sqrt((x ** 2).mean())))
    H = E[E.dBASE_b.abs() > 1e-9].groupby("hour")[["dBASE_b", "W40_b", "v22_b"]].mean()
    log("mean change by hour (changed rows):"); log(H.round(3).T.to_string())


def main():
    os.makedirs(os.path.join(OUT, "local_ckpt"), exist_ok=True)
    L0 = pd.read_pickle(os.path.join(AUD, "lab_feats.pkl")); T0 = pd.read_pickle(os.path.join(AUD, "test_feats.pkl"))
    C = json.load(open(os.path.join(AUD, "cols.json"))); ct, phc = C["ct"], C["phc"]
    RAW = raw_panel()
    # sanity: the MASK world rebuilt here equals the audit frame
    LM, ctM, phcM = world(ds_loader(set()))
    LM = LM.set_index("row_id").loc[L0.row_id].reset_index()
    assert ctM == ct and phcM == phc
    ok = np.allclose(LM[ct + phc].astype(float).values, L0[ct + phc].astype(float).values, equal_nan=True)
    log("MASK world rebuilt == audit lab_feats: %s (rows %d, ct %d, phc %d)" % (ok, len(L0), len(ct), len(phc)))
    assert ok
    R = part1(L0, ct, phc, RAW)
    report1(R)
    part2(L0, T0, ct, phc, RAW)


if __name__ == "__main__":
    main()
