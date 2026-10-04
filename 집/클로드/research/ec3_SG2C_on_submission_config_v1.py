# -*- coding: utf-8 -*-
"""SG2C (check, fixed before running; 2026-10-05 집 클로드).  SG2 (6.310) passed on the
R3S baseline.  Does it also help the SUBMISSION configuration (submission_13: final =
clip(shrink(0.8 R3_DP1 + 0.2 TabPFN)), shrink = 0.5 current + 0.5 today-to-date mean)?
Stored OOF, no refit:
  R3_DP1   local/ec3_DP1_all.csv dp_7 / dp_101 / dp_2024 (DIAG10, EL1)
  TabPFN   Codex ec_dc4_integration v2_integration_oof.csv season_pfn (DIAG10 only;
           same DIAG10 folds and seeds 7 / 101 / 2024)
Sets: DIAG10 pass-2 rows (full configuration); EL1 pass-2 rows (no stored TabPFN ->
R3_DP1 with shrink only, descriptive).  SG2 correction = ec3_SG2 functions unchanged
(reference = labelled records outside the fold, anchors either side, guard .30, 0.5),
applied after the final prediction, then the same clip.
Reading (fixed): 'helps the configuration' iff all 3 seeds better on DIAG10 pass-2
rows; seed-mean cluster bootstrap P(worse) reported."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
SEEDS = (7, 101, 2024)


def shrink(p, fr):
    d = fr[["farm", "day", "hour"]].copy(); d["p"] = p
    d = d.sort_values(["farm", "day", "hour"])
    avg = d.groupby(["farm", "day"]).p.transform(lambda s: s.expanding().mean())
    out = pd.Series(.5 * d.p + .5 * avg, index=d.index)
    return out.reindex(fr.index).values


def main():
    raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    lo, hi = lab.sub_ec.min(), lab.sub_ec.max()
    DP = pd.read_csv(os.path.join(env.LOCAL, "ec3_DP1_all.csv"))
    V = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_dc4_integration_20261002_v1", "v2_integration_oof.csv"))
    V = V[V.validator == "DIAG10"]
    pfn = V.groupby("row_id").season_pfn.mean()
    rr = V.groupby("row_id").season_r3.std().max()
    print("TabPFN identical across seed rows (max SD of season_pfn):", float(V.groupby("row_id").season_pfn.std().max()), " season_r3 SD across seeds", float(rr))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    res = {}
    for vname in ("DIAG10", "EL1"):
        F = DP[DP.validator == vname].reset_index(drop=True)
        if vname == "DIAG10":
            F["pfn"] = F.row_id.map(pfn)
            assert F.pfn.notna().all()
        for s in SEEDS:
            rawp = .8 * F["dp_%d" % s] + .2 * F.pfn if vname == "DIAG10" else F["dp_%d" % s]
            F["cfg_%d" % s] = np.clip(shrink(rawp.values, F), lo, hi)
            F["sg_%d" % s] = np.nan
        for k, G in F.groupby("validation_fold"):
            if not (G.day >= 179).any():
                continue
            vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}
            cal = sg2.ref_calendar(R, WV, ref)
            for s in SEEDS:
                F.loc[G.index, "sg_%d" % s] = np.clip(sg2.correction(G, "cfg_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        P2 = F[F.day >= 179].copy()
        P2["dm"] = P2.groupby(["farm", "day"]).sub_ec.transform("mean")
        cells = []
        for s in SEEDS:
            a, b = r(P2["cfg_%d" % s] - P2.sub_ec), r(P2["sg_%d" % s] - P2.sub_ec)
            cells.append((a, b))
        tag = "full config (0.8 R3_DP1 + 0.2 TabPFN, shrink)" if vname == "DIAG10" else "R3_DP1 + shrink (no stored TabPFN)"
        print("\n%s pass-2 rows, %s:" % (vname, tag))
        print("  " + "  ".join("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)) for s, (a, b) in zip(SEEDS, cells)))
        for nm, m in (("normal days", P2.dm < 1), ("high days", P2.dm >= 1)):
            G = P2[m]
            print("  %-11s " % nm + "  ".join("s%d %.4f->%.4f" % (s, r(G["cfg_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec)) for s in SEEDS))
        P2["cl"] = P2.farm + "_" + (P2.day // 5).astype(str)
        bm = P2[["cfg_%d" % s for s in SEEDS]].mean(axis=1); cm = P2[["sg_%d" % s for s in SEEDS]].mean(axis=1)
        dd = ((cm - P2.sub_ec) ** 2 - (bm - P2.sub_ec) ** 2).groupby(P2.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values
        idx = np.random.default_rng(20261005).integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        print("  seed-mean %.4f -> %.4f  P(worse) %.4f" % (r(bm - P2.sub_ec), r(cm - P2.sub_ec), p))
        res[vname] = all(b < a for a, b in cells)
    print("\nSG2C: helps the submission configuration on DIAG10 pass-2 (all seeds):", res["DIAG10"], "| EL1 descriptive all seeds:", res["EL1"])


if __name__ == "__main__":
    main()
