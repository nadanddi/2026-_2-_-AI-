# -*- coding: utf-8 -*-
"""NX1 (2026-10-11 집 클로드, user: "EC쪽 뭐라도 해봐"; 집 클로드 now owns EC): evaluation-set column for the
locally passed but never-LB-tested B4 (TabPFN share 0.4 instead of 0.2, catalog 6.355 ADOPT, 6.420 conditional pass)
on top of the exact submission_14 pipeline.  Uses the stored submission_14 member predictions (run1/prediction_details.npz)
-> 0.6*R3 + 0.4*PFN -> same shrink/clip -> same SG2.  Descriptive only: reports how much the evaluation column moves
vs EC14 (score identity: a column whose change RMS exceeds 2x0.1384 is a guaranteed loss).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_nx1_b4_column_v1.py
"""
import os, sys
import env, env_extra  # noqa: F401  research environment (packages); pkg model reuses these modules
from pathlib import Path
PKG = Path(r"C:\work\farmai\집\클로드\local\submission14_ec_sg2_20261005\pkg")
RUN = PKG.parent / "run1"
sys.path.insert(0, str(PKG)); os.chdir(PKG)
import model as M  # noqa: E402  (pkg env/env_extra)
import numpy as np, pandas as pd  # noqa: E402

OUT = Path(r"C:\work\farmai\집\클로드\research\local\nx1"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    x, tx, tr, q, ids, ts, table, notes = M.load(PKG / "data")
    z = np.load(RUN / "prediction_details.npz")
    assert (z["row_id"] == q.row_id.to_numpy(str)).all()
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    y = pd.read_csv(PKG / "data" / "train_y.csv", usecols=["row_id", "sub_ec"]); yi = M.identify(y)
    ec = yi.groupby(["farm", "day"]).sub_ec.mean(); ref = set(ec.index)
    S = M.sg2post.prepare(pd.concat([x, tx], ignore_index=True)); cal = M.sg2post.ref_calendar(S, ref)
    lo, hi = float(y.sub_ec.min()), float(y.sub_ec.max())
    res = {}
    for name, w in (("EC14", .2), ("B4", .4)):
        p13 = np.clip(M.shrink((1 - w) * r3 + w * pfn, q), lo, hi)
        res[name] = np.clip(M.sg2post.correct(q[["row_id"]], p13, S, ec, ref, cal), lo, hi)
    s14 = pd.read_csv(RUN / "submission_14.csv", encoding="utf-8-sig").set_index("row_id").sub_ec
    chk = np.abs(res["EC14"] - s14.loc[q.row_id].values).max(); print("re-derived EC14 vs submission_14 max |diff| %.2e" % chk)
    d = res["B4"] - res["EC14"]
    print("B4 - EC14: RMS %.4f mean %+.4f min %+.3f max %+.3f" % (np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))
    dd = pd.DataFrame({"farm": q.farm, "day": q.day, "ec14": res["EC14"], "b4": res["B4"]}).groupby(["farm", "day"]).mean()
    dd["diff"] = dd.b4 - dd.ec14
    print(dd.sort_values("ec14", ascending=False).round(3).head(12).to_string())
    print("daily change: mean %+.4f, |diff|>.05 days %d of 60" % (dd["diff"].mean(), (dd["diff"].abs() > .05).sum()))
    pd.DataFrame({"row_id": q.row_id, "ec14": res["EC14"], "b4": res["B4"]}).to_csv(OUT / "nx1_b4_eval_column.csv", index=False)


if __name__ == "__main__":
    main()
