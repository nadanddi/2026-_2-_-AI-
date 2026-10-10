# -*- coding: utf-8 -*-
"""TS7b: cold-validator safety of V3 (2-layer GRU) with the day gate T=10 on the OLD validators (2026-10-11 집 클로드).
TT26 passed V3 on layout K.  Safety rule (fixed before running, as for the 1-layer GRU in TS6e): EXT10 and EXT12 each fail iff
seed-mean worse and share(better) < .025; DIAG10/EL1 reported.  V3 refit on the old folds (TS7 recipe, seeds 1/2).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts7b_v3_cold_safety_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s7", os.path.join(HERE, "ts7_sequence_variants_v1.py")); S7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S7)
S6 = S7.S6; W = S6.W; RD = S6.RD; common = S6.common
CK = os.path.join(env.LOCAL, "ts7b_ckpt"); CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True)
    import env_extra_gpu  # noqa: F401
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds(); wb = np.asarray(wb, float); y = lab.sub_temp.values
    S = S6.sequences(lab)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            p = np.mean([S7.fit_predict(S[tm], y[tm], wb[tm], S[vm], sd, dev, "V3") for sd in (1, 2)], 0)
            pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values, "V3": p}).to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def summarize():
    rng = np.random.default_rng(20261103)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True).merge(
        pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))]), on=["validator", "row_id"])
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].reset_index(drop=True)
    ok = True
    for v in ("EXT10", "EXT12", "DIAG10", "EL1"):
        X = G[G.validator == v].reset_index(drop=True); g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
        mm = X.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(X.index)
        h = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1)); y = X.sub_temp.values; rel, pw = [], []
        cl = (X.farm + "_" + (X.day // 5).astype(str)).values
        for s in (7, 101):
            for f in "AB":
                a = (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * X.codex_REF + .4 * g * X["pfn_%s" % f]).values; b = a + .1 * g * h * (X.V3.values - a)
                dd = pd.Series((b - y) ** 2 - (a - y) ** 2).groupby(cl).agg(["sum", "count"]); idx = rng.integers(0, len(dd), (20000, len(dd)))
                pw.append(float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())); rel.append(r(b - y) / r(a - y) - 1)
        m = np.mean(rel); o = not (m > 0 and (1 - min(pw)) < .025) if v in ("EXT10", "EXT12") else True; ok &= o
        print("  %-6s %+.2f%% [%+.2f, %+.2f] P max %.4f | V3 alone %.3f %s" % (v, 100 * m, 100 * min(rel), 100 * max(rel), max(pw), r(X.V3 - X.sub_temp), ("ok" if o else "FAIL") if v in ("EXT10", "EXT12") else "(report)"))
    print("VERDICT V3 cold safety: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TS7B_SUM") != "1":
        run()
    summarize()
