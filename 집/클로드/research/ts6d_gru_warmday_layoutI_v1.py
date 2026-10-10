# -*- coding: utf-8 -*-
"""TS6d: GRU member gated by row warmth AND by 'was this day cold so far' - judged on layout I (never used for the GRU) plus
cold-validator safety (2026-10-10 집 클로드).  TS6b: 0.1g GRU passed on F/G/H (all rows -0.71..-0.72%, pass-2 -1.6..-2.0%,
pooled P .0025) but on the old validators EXT12 +2.09% (cold days' warm hours).  Fixed before running.
Candidates (k = 2, alpha .0125):
  GW : blend = W40G-S + 0.1 * g * (GRU - W40G-S)                       (TS6b form, 4th-layout check)
  GWD: blend = W40G-S + 0.1 * g * h * (GRU - W40G-S),  h = clip((m_t - 8) / 2, 0, 1),
       m_t = minimum of in_temp over hours 0..t of the same record day (causal; NaN -> h = 1)
PASS iff (a) layout I (members from TT23: BASE 808/909, PFN 41-48, CODEX R; GRU refit here with the TS6 recipe), all rows:
  all 4 cells better and farm x 5-day block bootstrap P(worse) < .0125;  AND (b) safety on the old cold validators EXT10 and
  EXT12 (TT1 members + TS6 GRU OOF): fail iff seed-mean worse and share(better) < .0125.  DIAG10/EL1 reported.
  Rough days excluded from scoring.
Run (after TT23):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts6d_gru_warmday_layoutI_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S6 = load("s6", "ts6_gru_member_v1.py"); T23 = load("t23", "tt23_pw1_confirm_v1.py")
W = S6.W; RD = S6.RD; ALPHA = .0125
CK = os.path.join(env.LOCAL, "ts6d_ckpt"); CK23 = os.path.join(env.LOCAL, "tt23_ckpt")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True)
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float)
    S = S6.sequences(lab); y = lab.sub_temp.values
    for k, fd in enumerate(T23.fresh_folds(lab)):
        path = os.path.join(CK, "I_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = T23.masks(lab, fd)
        p = np.mean([S6.fit_predict(S[tm], y[tm], wb[tm], S[vm], sd, dev) for sd in (1, 2)], 0)
        pd.DataFrame({"row_id": lab.row_id[vm].values, "GRU": p}).to_csv(path, index=False)
        print("I/%d done (GRU alone %.3f)" % (k, r(p - y[vm])), flush=True)


def gates(X):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    m = X.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(X.index)
    h = np.where(m.isna(), 1, np.clip((m - 8) / 2, 0, 1))
    return g, h


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261028)
    I = pd.concat([pd.read_csv(os.path.join(CK23, f)) for f in sorted(os.listdir(CK23)) if f.startswith("I_")], ignore_index=True)
    I = I.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))]), on="row_id")
    I = I[[(f, d) not in RD for f, d in zip(I.farm, I.day)]].reset_index(drop=True)
    O = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "tt1_ckpt", f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, "tt1_ckpt")))], ignore_index=True)
    O = O.merge(pd.concat([pd.read_csv(os.path.join(env.LOCAL, "ts6_ckpt", f)) for f in sorted(os.listdir(os.path.join(env.LOCAL, "ts6_ckpt")))]), on=["validator", "row_id"])
    O = O[[(f, d) not in RD for f, d in zip(O.farm, O.day)]].reset_index(drop=True)
    for cand in ("GW", "GWD"):
        print("\n==== %s" % cand); ok = True
        for v, X, seeds, bc, rc, role in (("layout I", I, (808, 909), "base_R_%d", "codex_R", "primary"),
                                           ("EXT10", O[O.validator == "EXT10"].reset_index(drop=True), (7, 101), "base_REF_%d", "codex_REF", "safety"),
                                           ("EXT12", O[O.validator == "EXT12"].reset_index(drop=True), (7, 101), "base_REF_%d", "codex_REF", "safety"),
                                           ("DIAG10", O[O.validator == "DIAG10"].reset_index(drop=True), (7, 101), "base_REF_%d", "codex_REF", "report"),
                                           ("EL1", O[O.validator == "EL1"].reset_index(drop=True), (7, 101), "base_REF_%d", "codex_REF", "report")):
            g, h = gates(X); wgt = .1 * g * (h if cand == "GWD" else 1); y = X.sub_temp.values; rel, pw, r2 = [], [], []
            for s in seeds:
                for f in "AB":
                    a = (.4 * X[bc % s] + (.2 + .4 * (1 - g)) * X[rc] + .4 * g * X["pfn_%s" % f]).values; b = a + wgt * (X.GRU.values - a)
                    rel.append(r(b - y) / r(a - y) - 1); pw.append(boot(X, a, b, rng)); L2 = X.day.values >= 179; r2.append(r((b - y)[L2]) / r((a - y)[L2]) - 1)
            m = np.mean(rel)
            if role == "primary":
                o = all(x < 0 for x in rel) and max(pw) < ALPHA; ok &= o
            elif role == "safety":
                o = not (m > 0 and (1 - min(pw)) < ALPHA); ok &= o
            else:
                o = True
            print("  %-8s %-7s all %+.2f%% [%+.2f, %+.2f] P max %.4f | 2차 %+.2f%% %s" % (v, role, 100 * m, 100 * min(rel), 100 * max(rel), max(pw), 100 * np.mean(r2), "" if role == "report" else ("ok" if o else "FAIL")))
        print("VERDICT %s: %s" % (cand, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS6D_SUM") != "1":
        run()
    summarize()
