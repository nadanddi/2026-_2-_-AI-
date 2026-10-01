# -*- coding: utf-8 -*-
"""LS1: how much of the G_C2 day-offset oracle can PUBLIC train_y labels recover?
(lab Claude, 2026-10-01)  -- a MEASUREMENT (diagnostic), not a model candidate.

Background
  57% of G_C2 DIAG10 squared error is the day-level offset e_day = mean_h(pred - y).
  Knowing e_day for the worst 10% of days alone gives DIAG10 -23.7% (6b.32~33).
  The user allowed (2026-10-01) train_y labels as features for evaluation rows
  (default: earlier days; later-day use is flagged separately).
  Real test layout (checked today): every test block has labelled train days on
  both sides at record distance 2 (one empty day between), and via the weather-only
  calendar map (deep_cal_9, no labels used) 27/60 test days have a labelled
  same-farm sibling of the same calendar date, 44/60 one in either farm.

Question
  Do offsets of OTHER labelled days (cross-fitted G_C2 residuals) predict e_day(d)?

Offsets  e_day from G_C2 DIAG10 OOF rebuilt from saved members (as re17).
Eligibility of a source day d' for target day (f, d)
  labelled, d' != d, DIAG10 fold(d') != fold(d)  (a real cross-fitted residual is
  never from the same model as the test prediction), and if same farm |d - d'| >= 2
  (the test gap).
Sources (mean of eligible days' e_day; 0 if none)
  P set (default, earlier/concurrent):
    rec_prev   same farm, latest eligible earlier record day with d - d' in 2..11
    sib_early  same farm, same calendar date, record-earlier
    oth_same   other farm, same calendar date
    cal_prev   same farm, calendar date c-1 (any record position)
  F set (flagged, adds future direction):
    P + rec_next (earliest later record day, 2..11), sib_late (same date, record-later),
        cal_next (same farm, c+1)
Estimator  ridge (alpha 1, no intercept) on source values, fitted on days of the
  other 9 DIAG10 folds, applied to the held-out fold (cross-fitted).  Correction
  pred_new = pred - e_hat(day) on every hour of the day.
Metrics  row RMSE base / corrected / oracle (e_day removed); recovery fraction
  = (base - corr) / (base - oracle); day-block bootstrap (farm x DIAG 5-day chunk,
  2000 reps, seed 0) of the relative change and P(worse); subsets: all 400 days,
  2nd segment (day >= 179, the test-like part), worst-10% |e_day| days.
  Availability of each source for the REAL 60 test days (label-free bookkeeping).

PRE-SET INTERPRETATION RULE (fixed before running):
  PROCEED to a separately pre-registered model hypothesis (H-LS1) only if, for the
  P set: all-400 relative change < 0 with P(worse) < 0.025, recovery fraction
  >= 0.10, AND relative change < 0 on the 2nd-segment subset.
  Otherwise CLOSE the "label structure recovers the day offset" direction (P set).
  The F set is descriptive only (flagged), whatever it shows.
  Nothing here is a submission or adoption.

Output  logs/ls1_label_structure_recovery_v1.log
Run     PYTHONPATH="" python -u ls1_label_structure_recovery_v1.py  (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import load  # noqa: E402
from anal_q1_errors import diag_folds  # noqa: E402

LOG = os.path.join(HERE, "..", "logs", "ls1_label_structure_recovery_v1.log")
P_SET = ["rec_prev", "sib_early", "oth_same", "cal_prev"]
F_SET = P_SET + ["rec_next", "sib_late", "cal_next"]
out = open(LOG, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


def gc2_diag(lab):
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    base = np.mean([z["DIAG10__MASK__7"], z["DIAG10__MASK__101"]], axis=0)
    cx = np.mean([z["DIAG10__CODEX__726"], z["DIAG10__CODEX__727"]], axis=0)
    pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_DIAG10.npy").mean(0)
    return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn


def sources(D, cal, labelled, fold_of, targets):
    """targets: list of (farm, day).  Source values from eligible labelled days in D."""
    ed = D.e_day.to_dict()
    bycal = {}
    for (f, d) in labelled:
        bycal.setdefault((f, cal[(f, d)]), []).append(d)
    days_of = {f: sorted(d for (ff, d) in labelled if ff == f) for f in ("F13", "F47")}
    rows = []
    for (f, d) in targets:
        c = cal[(f, d)]
        fo = fold_of.get((f, d), -1)
        other = "F47" if f == "F13" else "F13"

        def ok(ff, dd):
            if (ff, dd) not in ed or (ff == f and dd == d):
                return False
            if fold_of[(ff, dd)] == fo:
                return False
            return not (ff == f and abs(dd - d) < 2)

        r = {}
        prev = [x for x in days_of[f] if 2 <= d - x <= 11 and ok(f, x)]
        nxt = [x for x in days_of[f] if 2 <= x - d <= 11 and ok(f, x)]
        r["rec_prev"] = [ed[(f, max(prev))]] if prev else []
        r["rec_next"] = [ed[(f, min(nxt))]] if nxt else []
        sib = [x for x in bycal.get((f, c), []) if ok(f, x)]
        r["sib_early"] = [ed[(f, x)] for x in sib if x < d]
        r["sib_late"] = [ed[(f, x)] for x in sib if x > d]
        r["oth_same"] = [ed[(other, x)] for x in bycal.get((other, c), []) if ok(other, x)]
        r["cal_prev"] = [ed[(f, x)] for x in bycal.get((f, c - 1), []) if ok(f, x)]
        r["cal_next"] = [ed[(f, x)] for x in bycal.get((f, c + 1), []) if ok(f, x)]
        row = {"farm": f, "day": d}
        for k, v in r.items():
            row[k] = float(np.mean(v)) if v else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(["farm", "day"])


def ridge_fit(X, y, alpha=1.0):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def main():
    _, lab, _ = load()
    lab = lab.copy()
    lab["pred"] = gc2_diag(lab)
    lab["e"] = lab.pred - lab.sub_temp
    D = lab.groupby(["farm", "day"]).agg(e_day=("e", "mean")).reset_index().set_index(["farm", "day"])
    k = pd.read_csv(env.LOCAL + "/deep_cal_9_days.csv")
    cal = {(f, int(d)): int(c) for f, d, c in zip(k.farm, k.day, k.cal)}
    folds = diag_folds(lab)
    fold_of = {(f, int(d)): i for i, fd in enumerate(folds) for f in fd for d in fd[f]}
    labelled = list(D.index)
    p("labelled days %d, rows %d; G_C2 DIAG10 RMSE %.5f" % (len(D), len(lab), np.sqrt((lab.e ** 2).mean())))

    S = sources(D, cal, labelled, fold_of, labelled)
    S = S.join(D)
    S["fold"] = [fold_of[i] for i in S.index]
    S["seg2"] = S.index.get_level_values("day") >= 179
    q90 = S.e_day.abs().quantile(0.9)
    S["worst10"] = S.e_day.abs() >= q90

    p("\n1. per-source availability and correlation with e_day (labelled days)")
    p("%-10s %5s %8s %8s | %5s %8s" % ("source", "n", "r", "slope", "n2nd", "r2nd"))
    for s in F_SET:
        m = S[s].notna()
        m2 = m & S.seg2
        r = np.corrcoef(S.loc[m, s], S.loc[m, "e_day"])[0, 1] if m.sum() > 3 else np.nan
        sl = np.polyfit(S.loc[m, s], S.loc[m, "e_day"], 1)[0] if m.sum() > 3 else np.nan
        r2 = np.corrcoef(S.loc[m2, s], S.loc[m2, "e_day"])[0, 1] if m2.sum() > 3 else np.nan
        p("%-10s %5d %+8.3f %+8.3f | %5d %+8.3f" % (s, m.sum(), r, sl, m2.sum(), r2))

    # real test-day availability (label-free bookkeeping: cal map + which days have labels)
    test_days = sorted(set(cal) - set(labelled))
    test_days = [x for x in test_days if x[0] in ("F13", "F47")]
    fo_none = {**fold_of}
    St = sources(D, cal, labelled, {**fo_none, **{x: -1 for x in test_days}}, test_days)
    p("\n   real test days %d: available per source  " % len(St)
      + "  ".join("%s %d" % (s, St[s].notna().sum()) for s in F_SET))
    p("   test days with >=1 P source: %d, >=1 F source: %d"
      % (St[P_SET].notna().any(axis=1).sum(), St[F_SET].notna().any(axis=1).sum()))

    lab = lab.join(S, on=["farm", "day"], rsuffix="_d")
    chunk = {}
    for f in ("F13", "F47"):
        ds = sorted(D.loc[f].index)
        for i, d in enumerate(ds):
            chunk[(f, d)] = (f, i // 5)
    blocks = np.array([hash(chunk[(f, d)]) for f, d in zip(lab.farm, lab.day)])
    ub = np.unique(blocks)
    bidx = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(0)
    draws = [rng.choice(ub, len(ub)) for _ in range(2000)]

    def rm(e, idx=None):
        return np.sqrt(np.mean(e ** 2)) if idx is None else np.sqrt(np.mean(e[idx] ** 2))

    for name, cols in (("P (default)", P_SET), ("F (flagged, future incl.)", F_SET)):
        X = np.column_stack([S[c].fillna(0).values for c in cols])
        y = S.e_day.values
        ehat = np.zeros(len(S))
        coefs = []
        for fo in range(len(folds)):
            tr = (S.fold != fo).values
            b = ridge_fit(X[tr], y[tr])
            coefs.append(b)
            ehat[~tr] = X[~tr] @ b
        S["ehat"] = ehat
        eh = lab[["farm", "day"]].join(S.ehat, on=["farm", "day"]).ehat.values
        e0 = lab.e.values
        e1 = e0 - eh
        eo = e0 - lab.e_day.values
        p("\n2. set %s  sources %s" % (name, cols))
        p("   mean ridge coef: " + "  ".join("%s %+.3f" % (c, v) for c, v in zip(cols, np.mean(coefs, 0))))
        for sub, mask in (("all 400", np.ones(len(lab), bool)), ("2nd seg", lab.seg2.values),
                          ("worst10%", lab.worst10.values)):
            idx = np.where(mask)[0]
            b0, b1, bo = rm(e0, idx), rm(e1, idx), rm(eo, idx)
            rec = (b0 - b1) / (b0 - bo)
            p("   %-9s rows %5d  base %.4f  corr %.4f (%+.2f%%)  oracle %.4f (%+.1f%%)  recovery %.3f"
              % (sub, len(idx), b0, b1, 100 * (b1 / b0 - 1), bo, 100 * (bo / b0 - 1), rec))
        rel = []
        for dr in draws:
            ii = np.concatenate([bidx[b] for b in dr])
            rel.append(rm(e1, ii) / rm(e0, ii) - 1)
        rel = np.array(rel)
        p("   all-400 bootstrap: rel %+.2f%%  95%% CI [%+.2f, %+.2f]%%  P(worse) %.4f"
          % (100 * (rm(e1) / rm(e0) - 1), 100 * np.percentile(rel, 2.5), 100 * np.percentile(rel, 97.5),
             (rel >= 0).mean()))
        d = S[["e_day", "ehat", "worst10"]]
        p("   day level: corr(ehat, e_day) %+.3f;  worst10 AUC of |ehat| %.3f"
          % (np.corrcoef(d.ehat, d.e_day)[0, 1], auc(np.abs(d.ehat.values), d.worst10.values)))
        if name.startswith("P"):
            rel_all = rm(e1) / rm(e0) - 1
            idx2 = np.where(lab.seg2.values)[0]
            rel2 = rm(e1, idx2) / rm(e0, idx2) - 1
            rec_all = (rm(e0) - rm(e1)) / (rm(e0) - rm(eo))
            pw = (rel >= 0).mean()
            ok = rel_all < 0 and pw < 0.025 and rec_all >= 0.10 and rel2 < 0
            p("\n   PRE-SET RULE (P set): rel<0 %s, P(worse)<0.025 %s, recovery>=0.10 %s, 2nd seg<0 %s -> %s"
              % (rel_all < 0, pw < 0.025, rec_all >= 0.10, rel2 < 0, "PROCEED to H-LS1" if ok else "CLOSE"))


def auc(score, lab01):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = lab01.sum()
    n0 = len(lab01) - n1
    return (r[lab01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    main()
