# -*- coding: utf-8 -*-
"""RS1 (2026-10-11 집 클로드): re-stitched cross-day BASE features, scored on the EV temperature harness.

Motivation (measured just before, DIAG10 OOF, hour 0): when the previous record NUMBER is truly the same house (25 records)
BASE .46 / final blend .45; when it is another house (148 records) BASE .98 / blend .84.  BASE (common.py and the other
production builders) computes lags / rolling 3..168 h / ewm / previous-day aggregates / hours-since on a per-farm axis ordered by
record number, so at early hours these features mostly carry ANOTHER house's history ("seam contamination").

Everything below is FIXED BEFORE RUNNING.

RS1 features.  For a target record r = (farm f, num d) its history days d-1 .. d-7 on the farm-f axis are REPLACED by its re-stitched
predecessor chain c1 .. c7:
  step(x) = linker prev of x if x is in cv_all400_choice_v7m.csv with p >= .8 (fixed), else the previous number (farm(x), num(x) - 1).
  c1 = step(r), c2 = step(c1), ... up to 7 steps.  The chain STOPS at the first link that is missing (record has no inputs anywhere),
  not visible (below), or a cycle; positions beyond the stop are NaN (all inputs blank).  Content = the predecessor's raw inputs
  (all train_X / test_X input columns) hour by hour.  The target's own record is untouched.  Then the unchanged production
  feature builders (temp_mask_v1.build_world: common / features_v2 / feat_new / features_v4) run on that axis and only the
  target's rows are taken.
Implementation: targets are processed in "worlds"; one world = all targets of one fold with num % 8 == j (8 worlds), so the 7-day
  windows of targets in the same world never overlap another target.  A window position with no record on the axis gets synthetic
  input rows (no labels); positions before the farm's first number are dropped (= NaN).
Visibility:
  TR world of EV fold k (training rows): world = TF1/EV MASK world (test_X inputs NaN, val_k U del_k blanked); a predecessor is
    visible iff it is a train_X record that is neither in val_k U del_k nor a test record.
  VA world of fold k (val rows, both IDs at once - every builder works per farm axis, and the target windows are rewritten):
    world = del_k blanked, test_X kept; for a target of ID F a predecessor is NOT visible if it is a del_k record, a val_k record
    of the other ID, a same-ID val_k record with a number >= the target's (only EARLIER same-ID val records are allowed), or a
    test_X record of the other ID (rule safety: other-ID evaluation inputs are banned).  Training records of either ID are visible.
  Labels are never used to build features (the linker file is consumed as given: rec, prev, p).
Model / chain (identical to ev_temp_pipeline_v1.py config 'a'): BASE production recipe retrained per EV fold on TR_k RS1 features,
  seeds 7 and 101, production row weights (lab_feats.pkl by row_id); stored DIAG10 CODEX_REF and TabPFN A/B; W40G-S blend;
  v18 corrector (built from stored OOF, fold geometry, unchanged); round-12 and v22 anchors (same anchor list / fold drops).
  Current = stored ev_all.pkl config 'a' (recomputed here as a reproduction check).
Scoring: val rows, 30 strict CO2-rough days excluded (temperature protocol); scopes all val rows / pass-2 (num >= 179); per farm;
  hours: 0, 0-5, whole day; split by whether the previous NUMBER is truly the same house (linker_v3 truth.pkl, cert links:
  'same' = cert and a1 == (farm, num-1); 'other' = cert and a1 != previous number; 'unk' = not cert).
  Members: BASE alone (seed mean) and the final chain v22 (cell mean of 4 cells = seed {7,101} x PFN {A,B}).
  Bootstrap: record-level, 4000 draws, seed 0, P(RMSE_RS1 >= RMSE_current).
PRE-FIXED PASS: final-chain (v22) RMSE of RS1 < current in EVERY one of the 4 cells on pass-2 rows AND on all val rows, AND
  P(pass-2) < .025.  Single candidate (no Bonferroni).  Everything else is context.
Approximations (known before running): content older than 7 days (beyond the window) still reaches ewm tails (<= ~9 % weight for
  half-life 48 h), hours-since counters and season cumulative sums; the linker map covers only the 400 labelled records (other
  records step to the previous number); weights not recomputed per fold; CODEX/TabPFN not retrained (stored DIAG10 OOF).
Run:  cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u rs1_restitch/rs1_restitch_v1.py --folds 0,1
      (BASE checkpoints per fold; any split over processes), then  ... rs1_restitch_v1.py --report
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env  # noqa: E401,E702
import os, json, time, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

RES = os.path.join(env.ROOT, u"집", u"클로드", "research")
OUT = os.path.join(RES, "rs1_restitch"); CK = os.path.join(OUT, "local_ckpt")


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


EVP = _load("evp", os.path.join(RES, "ev_validator", "temp", "ev_temp_pipeline_v1.py"))
TF1 = EVP.TF1; V18 = EVP.V18
ORIG = TF1.ORIG
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
TRUTH = os.path.join(env.LOCAL, "linker_v3", "truth.pkl")
EVF = os.path.join(RES, "ev_validator", "ev_folds_v1.csv")
P_MIN = .8; DEPTH = 7; NCOL = 8
SEEDS = EVP.SEEDS; FAMS = EVP.FAMS; CELLS = EVP.CELLS
rm = EVP.rm; log = EVP.log
IDC = ("row_id", "farm", "day", "hour", "t")
FARMS = ("F13", "F47")


# ------------------------------------------------------------------ raw content / links
class Raw:
    def __init__(self):
        tX, ty, sX = ORIG()
        self.cols = [c for c in tX.columns if c not in IDC]
        self.cont = {}
        self.test = set()
        self.train = set()
        for src, D in (("t", tX), ("s", sX)):
            D = D[D.farm.isin(FARMS)]
            for (f, d), g in D.groupby(["farm", "day"]):
                a = np.full((24, len(self.cols)), np.nan); a[g.hour.values] = g[self.cols].values
                self.cont[(f, int(d))] = a
                (self.test if src == "s" else self.train).add((f, int(d)))
        self.fmin = {f: min(d for ff, d in self.cont if ff == f) for f in FARMS}
        L = pd.read_csv(LK)
        L = L[L.p >= P_MIN]
        self.link = {(r[:3], int(r[4:7])): (p[:3], int(p[4:7])) for r, p in zip(L.rec, L.prev)}

    def step(self, x):
        return self.link.get(x, (x[0], x[1] - 1))

    def chain(self, r, visible):
        out, cur, seen = [], r, {r}
        for _ in range(DEPTH):
            nx = self.step(cur)
            if nx not in self.cont or nx in seen or not visible(nx):
                break
            out.append(nx); seen.add(nx); cur = nx
        return out


def make_loader(raw, base_blank, mask_sx, targets, vis_fn, stats):
    """targets: list of (farm, num); vis_fn(target) -> visible(rec) predicate.  Built once, copies returned on every call."""
    cache = {}

    def ld():
        if "v" not in cache:
            cache["v"] = _ld()
        tX, ty, sX = cache["v"]
        return tX.copy(), ty.copy(), sX.copy()

    def _ld():
        tX, ty, sX = ORIG(); tX = tX.copy(); sX = sX.copy()
        cols = raw.cols
        if mask_sx:
            sX[cols] = np.nan
        keyT = list(zip(tX.farm, tX.day))
        m = np.array([k in base_blank for k in keyT]); tX.loc[m, cols] = np.nan
        writes = {}
        for r in targets:
            ch = raw.chain(r, vis_fn(r))
            stats.append(dict(rec="%s_%03d" % r, depth=len(ch), c1="" if not ch else "%s_%03d" % ch[0],
                              c1_is_prevnum=bool(ch) and ch[0] == (r[0], r[1] - 1),
                              default=ch == [(r[0], r[1] - k) for k in range(1, len(ch) + 1)] and len(ch) == DEPTH))
            for k in range(1, DEPTH + 1):
                pos = (r[0], r[1] - k)
                if pos[1] < raw.fmin[pos[0]]:
                    continue
                assert pos not in writes, "overlapping windows"
                writes[pos] = raw.cont[ch[k - 1]] if k <= len(ch) else np.full((24, len(cols)), np.nan)
        for D in (tX, sX):
            sel = D.farm.isin(FARMS).values
            sub = D.loc[sel, ["farm", "day", "hour"]]
            hit = np.array([(f, d) in writes for f, d in zip(sub.farm, sub.day)])
            if hit.any():
                ix = sub.index[hit]; ss = sub[hit]
                vals = np.stack([writes[(f, d)][h] for f, d, h in zip(ss.farm, ss.day, ss.hour)])
                D.loc[ix, cols] = vals
        have = set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day))
        add = []
        for (f, d), a in writes.items():
            if (f, d) in have or np.isnan(a).all():
                continue
            df = pd.DataFrame(a, columns=cols)
            df["row_id"] = ["%s_%03d_%02d" % (f, d, h) for h in range(24)]
            df["farm"] = f; df["day"] = d; df["hour"] = np.arange(24); df["t"] = d * 24 + df.hour
            add.append(df[tX.columns])
        if add:
            tX = pd.concat([tX] + add, ignore_index=True)
        return tX, ty, sX
    return ld


def build_rows(raw, targets, base_blank, mask_sx, vis_fn, ct, phc, stats):
    parts = []
    for j in range(NCOL):
        tj = [r for r in targets if r[1] % NCOL == j]
        if not tj:
            continue
        L, c2, p2 = TF1.world(make_loader(raw, base_blank, mask_sx, tj, vis_fn, stats)); assert c2 == ct and p2 == phc
        st = set(tj)
        parts.append(L[[(f, d) in st for f, d in zip(L.farm, L.day)]])
    return pd.concat(parts, ignore_index=True)


def base_fold(k, F, raw, ct, phc, wmap, L0):
    path = os.path.join(CK, "rs1b_fold%d.pkl" % k)
    if os.path.exists(path):
        return pd.read_pickle(path)
    t0 = time.time()
    val = set(map(tuple, F[(F.fold == k) & (F.role == "val")][["farm", "num"]].values))
    dele = set(map(tuple, F[(F.fold == k) & (F.role == "del")][["farm", "num"]].values))
    labrec = set(zip(L0.farm, L0.day))
    trT = sorted(r for r in labrec if r not in val and r not in dele)
    vis_tr = lambda r: (lambda x: x in raw.train and x not in val and x not in dele)
    stT = []
    TR = build_rows(raw, trT, val | dele, True, vis_tr, ct, phc, stT)

    def vis_va(r):
        def v(x):
            if x in dele:
                return False
            if x in val and (x[0] != r[0] or x[1] >= r[1]):
                return False
            if x in raw.test and x[0] != r[0]:
                return False
            return True
        return v
    stV = []
    VA = build_rows(raw, sorted(val), dele, False, vis_va, ct, phc, stV)
    w = TR.row_id.map(wmap).values.astype(float); assert np.isfinite(w).all()
    out = VA[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy(); out["fold"] = k
    for s in SEEDS:
        p = TF1.fit_base(TR, ct, phc, w, s); out["r_%d" % s] = p(VA)
    # sanity: hour-0 in_temp_lag1 of val rows == hour-23 in_temp of c1
    chk = []
    SV = pd.DataFrame(stV).set_index("rec")
    for rid, lag1 in zip(VA.row_id[VA.hour == 0], VA.in_temp_lag1[VA.hour == 0]):
        c1 = SV.loc[rid[:7], "c1"]
        exp = raw.cont[(c1[:3], int(c1[4:7]))][23, raw.cols.index("in_temp")] if c1 else np.nan
        chk.append((np.isnan(exp) and np.isnan(lag1)) or np.isclose(exp, lag1))
    out.attrs["stats_tr"] = stT; out.attrs["stats_va"] = stV
    out.to_pickle(path)
    pd.DataFrame(stV).assign(fold=k).to_csv(os.path.join(CK, "rs1_chains_val_fold%d.csv" % k), index=False)
    log("fold %d: TR rows %d (records %d, chain != default %d), VA rows %d (records %d; depth mean %.2f, c1 != prev number %d, "
        "depth 0 %d) | lag1 check %d/%d | %.0fs"
        % (k, len(TR), len(stT), sum(not s["default"] for s in stT), len(out), len(stV), np.mean([s["depth"] for s in stV]),
           sum(not s["c1_is_prevnum"] for s in stV), sum(s["depth"] == 0 for s in stV), sum(chk), len(chk), time.time() - t0))
    return out


def run_folds(folds):
    os.makedirs(CK, exist_ok=True)
    L0 = pd.read_pickle(os.path.join(EVP.AUD, "lab_feats.pkl")); C = json.load(open(os.path.join(EVP.AUD, "cols.json")))
    ct, phc = C["ct"], C["phc"]; wmap = dict(zip(L0.row_id, L0.w))
    F = pd.read_csv(EVF); raw = Raw()
    log("linker links p >= %.1f: %d" % (P_MIN, len(raw.link)))
    for k in folds:
        base_fold(k, F, raw, ct, phc, wmap, L0)


# ------------------------------------------------------------------ report
def report():
    F = pd.read_csv(EVF); folds = sorted(F.fold.unique())
    R = pd.read_pickle(os.path.join(EVP.CK, "ev_all.pkl"))
    RS = pd.concat([pd.read_pickle(os.path.join(CK, "rs1b_fold%d.pkl" % k)) for k in folds], ignore_index=True)
    assert RS.row_id.is_unique and len(RS) == len(R)
    R = R.merge(RS[["row_id", "r_7", "r_101"]], on="row_id", how="left"); assert R.r_7.notna().all()
    log("RS1 vs current BASE: corr s7 %.4f s101 %.4f" % (np.corrcoef(R.a_7, R.r_7)[0, 1], np.corrcoef(R.a_101, R.r_101)[0, 1]))
    D0 = os.path.join(env.ROOT, u"공용", u"대회자료", u"정형데이터", u"참가자_배포")
    tX = V18.ident(pd.read_csv(os.path.join(D0, "train_X.csv"))); sX = V18.ident(pd.read_csv(os.path.join(D0, "test_X.csv")))
    recs_all = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
    Lk = pd.read_csv(LK)
    Lk = Lk[Lk.prev_is_labelled.astype(bool) & (Lk.p >= .8) & (Lk.steps_to_anchor == 1) & Lk.anchor_st23.notna() & (Lk.dT_air.abs() <= 1.5)]
    AN_all = dict(zip(Lk.rec, Lk.anchor_st23)); PREV = dict(zip(Lk.rec, Lk.prev))
    st = pd.concat([pd.read_csv(os.path.join(EVP.OOF, "DIAG10_%d.csv" % k)) for k in range(10)], ignore_index=True)
    Rk = []
    for k in folds:
        val = set(map(tuple, F[(F.fold == k) & (F.role == "val")][["farm", "num"]].values))
        dele = set(map(tuple, F[(F.fold == k) & (F.role == "del")][["farm", "num"]].values))
        excl = val | dele; exk = {"%s_%03d" % x for x in excl}
        Q = R[R.fold == k].reset_index(drop=True).copy()
        key = (Q.farm + "_" + Q.day.map("{:03d}".format)).values; h = Q.hour.values
        AN = {r: AN_all[r] for r in set(key) if r in AN_all and PREV[r] not in exk}
        CTm, CTv = EVP.chain_tables(tX, sX, recs_all, dele, val, False)
        Qc = Q.merge(CTv, on=["farm", "day"], how="left"); Qc["jump"] = Qc.in_temp - Qc.ch_in23
        assert (Qc.row_id.values == Q.row_id.values).all()
        mods = EVP.corrector_preds(st, CTm, excl, Qc); early = h <= 7
        for t in ("a", "r"):
            for s, fam in CELLS:
                p = EVP.w40(Q["%s_%d" % (t, s)].values, Q.st_codex_REF.values, Q["st_pfn_" + fam].values, Q.in_temp.values)
                Qk = Qc.assign(p=p, p_air=p - Qc.in_temp.values)
                q = p.copy(); q[early] = p[early] + .5 * mods[(s, fam)].predict(Qk.loc[early, V18.FEAT])
                c = "%d%s" % (s, fam)
                Q["xblend_%s_%s" % (t, c)] = p; Q["xv18_%s_%s" % (t, c)] = q
                Q["xv22_%s_%s" % (t, c)] = EVP.anchor_apply(key, h, q, AN, "v22")
        Rk.append(Q)
    R = pd.concat(Rk, ignore_index=True)
    rep = max(float(np.max(np.abs(R["xv22_a_%d%s" % c] - R["v22_a_%d%s" % c]))) for c in CELLS)
    log("reproduction of stored current v22 cells: max |diff| %.2e" % rep)
    for t in ("a", "r"):
        R["base_%s" % t] = np.mean([R["%s_%d" % (t, s)] for s in SEEDS], 0)
        for st_ in ("xblend", "xv18", "xv22"):
            R["%s_%s" % (st_, t)] = np.mean([R["%s_%s_%d%s" % (st_, t, s, f)] for s, f in CELLS], 0)
    # truth classes
    T = pd.read_pickle(TRUTH)
    cls = {}
    for fb, nb, fa, na, ce in zip(T.fb, T.nb, T.fa, T.na, T.cert):
        cls["%s_%03d" % (fb, nb)] = "unk" if not ce else ("same" if (fa == fb and na == nb - 1) else "other")
    key = (R.farm + "_" + R.day.map("{:03d}".format)).values; y = R.sub_temp.values; hr = R.hour.values
    R["cls"] = [cls.get(x, "unk") for x in key]
    chains = pd.concat([pd.read_csv(os.path.join(CK, "rs1_chains_val_fold%d.csv" % k)) for k in folds], ignore_index=True)
    R = R.merge(chains[["rec", "depth", "c1_is_prevnum"]].rename(columns={"rec": "key"}), left_on=key, right_on="key", how="left")
    rough = np.array([(f, d) in TF1.RD for f, d in zip(R.farm, R.day)]); p2 = (R.day >= 179).values
    R.drop(columns=["key"]).to_pickle(os.path.join(CK, "rs1_ev_all.pkl"))
    base = ~rough
    scopes = [("all", base), ("pass2", base & p2)]
    for f in FARMS:
        scopes += [(f + "_all", base & (R.farm == f).values), (f + "_pass2", base & p2 & (R.farm == f).values)]
    for c in ("same", "other", "unk"):
        scopes += [("%s_all" % c, base & (R.cls == c).values), ("%s_pass2" % c, base & p2 & (R.cls == c).values)]
    scopes += [("c1!=prevnum_all", base & ~R.c1_is_prevnum.astype(bool).values), ("c1==prevnum_all", base & R.c1_is_prevnum.astype(bool).values)]
    rows = []
    for sc, m in scopes:
        for hn, hm in (("h0", hr == 0), ("h0-5", hr <= 5), ("day", np.ones(len(R), bool))):
            mm = m & hm
            d = dict(scope=sc, hours=hn, n_rec=len(set(key[mm])), n_rows=int(mm.sum()))
            for nm, col in (("BASE", "base"), ("blend", "xblend"), ("v18", "xv18"), ("final", "xv22")):
                a_, r_ = rm((R[col + "_a"] - y)[mm]), rm((R[col + "_r"] - y)[mm])
                d[nm + "_cur"] = a_; d[nm + "_rs1"] = r_; d[nm + "_d%"] = 100 * (r_ / a_ - 1)
            if mm.sum() and (hn == "day" or sc in ("all", "pass2")):
                d["P_final"] = EVP.boot(key, y, R.xv22_a.values, R.xv22_r.values, mm)[0]
            rows.append(d)
    TB = pd.DataFrame(rows); TB.to_csv(os.path.join(OUT, "rs1_scores_v1.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    cols = ["scope", "hours", "n_rec", "BASE_cur", "BASE_rs1", "BASE_d%", "final_cur", "final_rs1", "final_d%", "P_final"]
    log("\n================ RS1 vs current (EV, rough excluded; cell-mean predictions) ================")
    log(TB[cols].round(4).to_string(index=False))
    # cells + verdict
    crow = []; ok = True
    for sc, m in scopes[:2]:
        for s, f in CELLS:
            c = "%d%s" % (s, f); a_ = rm((R["xv22_a_" + c] - y)[m]); r_ = rm((R["xv22_r_" + c] - y)[m])
            crow.append(dict(scope=sc, cell=c, final_cur=a_, final_rs1=r_, better=r_ < a_)); ok &= r_ < a_
    TC = pd.DataFrame(crow); TC.to_csv(os.path.join(OUT, "rs1_cells_v1.csv"), index=False)
    log("\nper cell (final chain, whole day):"); log(TC.round(4).to_string(index=False))
    P2 = EVP.boot(key, y, R.xv22_a.values, R.xv22_r.values, scopes[1][1])[0]
    Pa = EVP.boot(key, y, R.xv22_a.values, R.xv22_r.values, scopes[0][1])[0]
    log("\nP(RS1 worse) final chain: pass-2 %.4f, all %.4f" % (P2, Pa))
    log("VERDICT RS1: %s (all cells better on pass-2 and all: %s; P pass-2 < .025: %s)" % ("PASS" if ok and P2 < .025 else "FAIL", ok, P2 < .025))
    ch = chains
    log("val chains: records %d, depth mean %.2f, depth 7 %d, depth 0 %d, c1 != previous number %d"
        % (len(ch), ch.depth.mean(), (ch.depth == 7).sum(), (ch.depth == 0).sum(), (~ch.c1_is_prevnum.astype(bool)).sum()))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--folds":
        run_folds([int(x) for x in a[1].split(",")])
    elif a and a[0] == "--report":
        report()
    else:
        run_folds(range(8)); report()
