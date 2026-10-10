# -*- coding: utf-8 -*-
"""EB1 (2026-10-11 집 클로드, user: "1~3 다 해봐"): re-judge the pending EC candidates under the EVALUATION GEOMETRY.
Catalog 6.536/6.538: evaluation chunks are separated from labelled stretches by one deleted record on each side, so neighbour
references directly adjacent to a hidden chunk do not exist at test time.  'Buffered' validators = EL1 (5-record pass-2 chunks)
and P2LOO (single pass-2 days) with the labelled record immediately before and after the hidden chunk removed from the
SG2/HG3 reference set (the stored base OOF already excluded same-farm +-1 days from training).
Stored RV1 OOF (no refit): CUR = 0.8 R3_DP1 + 0.2 PFN, B4 = 0.6 / 0.4, seeds 47/1414/6464.
Baseline: CUR + SG2(buffered).  Candidates (k = 3, alpha .025/3 = .0083), each with buffered references:
  C1 B4 + SG2      C2 CUR + HG3      C3 B4 + HG3
Fixed before running.  PASS iff all 3 seeds better on BOTH buffered EL1 and buffered P2LOO pass-2 rows AND seed-mean farm x
5-day block bootstrap P(worse) < .0083 on buffered EL1 (the chunk validator closest to the evaluation geometry).
Reported: normal / high days, per farm.  Caveat: same OOF that selected B4/HG3 before -> re-check, not fresh evidence.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_eb1_buffered_rejudge_v1.py
"""
import os, sys, json, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
CK = os.path.join(env.LOCAL, "eb1_ckpt"); ALPHA = .025 / 3
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = hg3.p3.prepare()
    Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[O.tag.isin(["CUR", "B4"]) & O.validator.isin(["EL1", "P2LOO"])]
    lohi = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"), usecols=["row_id", "lo", "hi"]).drop_duplicates("row_id").set_index("row_id")
    for (vn, k), G0 in O.groupby(["validator", "fold"]):
        path = os.path.join(CK, "%s_%d.csv" % (vn, k))
        if os.path.exists(path):
            continue
        vd = set(zip(G0.farm, G0.day)); buf = set()
        for f in {x[0] for x in vd}:
            labd = sorted(d for (ff, d) in labset if ff == f); hid = sorted(d for (ff, d) in vd if ff == f)
            b = [d for d in labd if d < hid[0] and (f, d) not in vd]; a = [d for d in labd if d > hid[-1] and (f, d) not in vd]
            if b: buf.add((f, b[-1]))
            if a: buf.add((f, a[0]))
        ref = {x for x in labset if x not in vd and x not in buf}
        cal = hg3.ref_calendar(Rr, WV, ref); ST = hg3.s_low_table(DM, ref); part = []
        for (tag, s), G in G0.groupby(["tag", "seed"]):
            G = G.reset_index(drop=True); lo, hi = lohi.lo.reindex(G.row_id).values, lohi.hi.reindex(G.row_id).values
            sg, hg = hg3.correction(G, "nosg", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal, DM, ST)
            part.append(G[["validator", "fold", "row_id", "tag", "seed", "farm", "day", "hour", "sub_ec"]].assign(sgb=np.clip(sg, lo, hi), hgb=np.clip(hg, lo, hi)))
        pd.concat(part).to_csv(path, index=False); print("  %s/%d done" % (vn, k), flush=True)


def summarize():
    rng = np.random.default_rng(20261113)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    P = P[P.day >= 179]
    W = P.pivot_table(index=["validator", "row_id", "seed", "farm", "day", "sub_ec"], columns="tag", values=["sgb", "hgb"]).reset_index()
    W.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in W.columns]
    W["dm"] = W.groupby(["validator", "seed", "farm", "day"]).sub_ec.transform("mean")
    base = "sgb_CUR"; cands = {"C1 B4+SG2": "sgb_B4", "C2 CUR+HG3": "hgb_CUR", "C3 B4+HG3": "hgb_B4"}
    for name, c in cands.items():
        ok = True; print("\n==== %s (buffered)" % name)
        for v in ("EL1", "P2LOO"):
            X = W[W.validator == v]; cells = [rm(Z[c] - Z.sub_ec) / rm(Z[base] - Z.sub_ec) - 1 for _, Z in X.groupby("seed")]
            ok &= all(x < 0 for x in cells)
            M = X.groupby("row_id").agg(y=("sub_ec", "first"), b=(base, "mean"), c=(c, "mean"), farm=("farm", "first"), day=("day", "first"), dm=("dm", "first"))
            nm = M.dm < 1
            line = "  %-5s cells %s | base %.4f -> %.4f | normal %.4f -> %.4f | high %.4f -> %.4f | F13 %+.2f%% F47 %+.2f%%" % (
                v, " ".join("%+.2f%%" % (100 * x) for x in cells), rm(M.b - M.y), rm(M.c - M.y), rm((M.b - M.y)[nm]), rm((M.c - M.y)[nm]),
                rm((M.b - M.y)[~nm]), rm((M.c - M.y)[~nm]), *(100 * (rm((M.c - M.y)[M.farm == f]) / rm((M.b - M.y)[M.farm == f]) - 1) for f in ("F13", "F47")))
            if v == "EL1":
                cl = (M.farm + "_" + (M.day // 5).astype(str)).values
                dd = pd.Series((M.c - M.y) ** 2 - (M.b - M.y) ** 2).groupby(cl).agg(["sum", "count"])
                idx = rng.integers(0, len(dd), (20000, len(dd))); p = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
                ok &= p < ALPHA; line += " | P(worse) %.4f" % p
            print(line)
        print("VERDICT %s: %s" % (name, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("EB1_SUM") != "1":
        run()
    summarize()
