# -*- coding: utf-8 -*-
"""BV1 (2026-10-11 집 클로드, descriptive): does SG2's validated gain survive the REAL evaluation geometry?
Catalog 6.536: the organisers deleted exactly one record between every labelled stretch and every evaluation chunk, so at
test time the nearest labelled neighbour of an evaluation record is >= 2 records away.  Our EL1 / P2LOO validators let SG2
use the directly adjacent labelled records.  BV1 recomputes SG2 on the stored RV1 OOF (CUR, seeds 47/1414/6464) with the
reference set additionally missing the labelled record immediately before and after each hidden chunk (EL1: 5-record chunks;
P2LOO: single days) - i.e. a one-record buffer on both sides, like the evaluation set.  Reports pass-2 RMSE of
no-SG2 / SG2 (as validated) / SG2 with buffer, overall, normal days, high days.  No adoption decision here.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_bv1_buffered_sg2_check_v1.py
"""
import os, sys, json, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    raw, full, lab, lock, signatures, fds = hg3.p3.prepare()
    Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[(O.tag == "CUR") & O.validator.isin(["EL1", "P2LOO"])]
    lohi = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"), usecols=["row_id", "lo", "hi"]).drop_duplicates("row_id").set_index("row_id")
    out = []
    for (vn, k), G0 in O.groupby(["validator", "fold"]):
        vd = set(zip(G0.farm, G0.day))
        buf = set()
        for f in {x[0] for x in vd}:
            labd = sorted(d for (ff, d) in labset if ff == f); hid = sorted(d for (ff, d) in vd if ff == f)
            before = [d for d in labd if d < hid[0] and (f, d) not in vd]; after = [d for d in labd if d > hid[-1] and (f, d) not in vd]
            if before: buf.add((f, before[-1]))
            if after: buf.add((f, after[0]))
        refs = {"std": {x for x in labset if x not in vd}, "buf": {x for x in labset if x not in vd and x not in buf}}
        for s, G in G0.groupby("seed"):
            G = G.reset_index(drop=True); lo, hi = lohi.lo.reindex(G.row_id).values, lohi.hi.reindex(G.row_id).values
            res = {}
            for tag, ref in refs.items():
                cal = hg3.ref_calendar(Rr, WV, ref)
                o, _ = hg3.correction(G, "nosg", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal, DM, hg3.s_low_table(DM, ref))
                res[tag] = np.clip(o, lo, hi)
            out.append(G.assign(sg_std=res["std"], sg_buf=res["buf"]))
        print("  %s/%d done" % (vn, k), flush=True)
    P = pd.concat(out, ignore_index=True); P = P[P.day >= 179]
    P["dm"] = P.groupby(["validator", "fold", "seed", "farm", "day"]).sub_ec.transform("mean")
    print("check: recomputed std SG2 vs stored sg max |diff| %.2e" % np.abs(P.sg_std - P.sg).max())
    for v, X in P.groupby("validator"):
        for nm, m in (("all", X.dm > -1), ("normal", X.dm < 1), ("high", X.dm >= 1)):
            Z = X[m]
            print("  %-6s %-6s no-SG2 %.4f | SG2 validated %.4f (%+.1f%%) | SG2 with buffer %.4f (%+.1f%%)" % (
                v, nm, rm(Z.nosg - Z.sub_ec), rm(Z.sg_std - Z.sub_ec), 100 * (rm(Z.sg_std - Z.sub_ec) / rm(Z.nosg - Z.sub_ec) - 1),
                rm(Z.sg_buf - Z.sub_ec), 100 * (rm(Z.sg_buf - Z.sub_ec) / rm(Z.nosg - Z.sub_ec) - 1)))


if __name__ == "__main__":
    main()
