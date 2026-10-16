# -*- coding: utf-8 -*-
"""submission_20 EC (2026-10-11 집 클로드; rule-clean replacement of W2): 9th-round EC (submission_14, reproduced exactly from
output_14) + EA1 linker anchors.  For evaluation records whose same-house previous date was found by linker v5 with confidence
(온도/linker_v5 output eval60_confident_1step_anchor_v5.csv: hour-0 decision, 1 step, previous record labelled, p >= .8; each
record decides on its own from its hour-0 inputs, earlier records and training data), pred_h = p_h + (A - pm_h), A = previous
record's 23 h EC (training label), pm_h = running mean of p over hours 0..h.  Local evidence: catalog 6.574 (EA1 PASS).
Run:  python ea1_post.py <eval60_confident_1step_anchor_v5.csv>   (writes output_20/ec_ea1.csv)
"""
import os, sys
import numpy as np, pandas as pd
import c18post as C


def main():
    anchors = sys.argv[1]
    out_dir = os.path.join(C.HERE, "output_20"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_ea1.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    z = np.load(os.path.join(C.HERE, "output_14", "prediction_details.npz")); rid = z["row_id"]
    s2 = np.load(os.path.join(C.HERE, "output_14", "sg2_details.npz")); assert (s2["row_id"] == rid).all()
    F = pd.DataFrame({"row_id": rid, "farm": [r[:3] for r in rid], "day": [int(r[4:7]) for r in rid], "hour": [int(r[8:10]) for r in rid]})
    ty = pd.read_csv(os.path.join(C.DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    ty["farm"], ty["day"] = ty.row_id.str[:3], ty.row_id.str[4:7].astype(int)
    ec = ty.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    F["p14pre"] = np.clip(C.shrink(.8 * r3 + .2 * pfn, F), lo, hi)
    Rr, WV, hrs, SIG, DM = C.prepare_structure()
    vd = set(zip(F.farm, F.day)); cal = C.ref_calendar(Rr, WV, labset); ST = C.s_low_table(DM, labset)
    sg14, _ = C.correction(F, "p14pre", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    F["ec14"] = np.clip(sg14, lo, hi)
    rep = bool(np.abs(F.ec14.values - s2["pred_final"]).max() < 1e-12)
    A = pd.read_csv(anchors); AN = dict(zip(A.row, A.anchor_ec23))
    F = F.sort_values(["farm", "day", "hour"]); F["rec"] = F.row_id.str[:7]
    cm = F.groupby("rec").ec14.transform(lambda s: s.expanding().mean())
    a = F.rec.map(AN)
    F["sub_ec"] = np.clip(np.where(a.notna(), F.ec14 + (a - cm), F.ec14), lo, hi)
    F = F.sort_index()
    F[["row_id", "sub_ec"]].to_csv(out, index=False, float_format="%.6f")
    d = F.sub_ec - s2["pred_final"]
    print("reproduces submission_14 EC %s (must be True on the original data); anchored records %d" % (rep, len(AN)))
    print("wrote %s | change vs submission_14 EC: RMS %.4f, rows changed %d" % (out, np.sqrt((d ** 2).mean()), int((d.abs() > 1e-9).sum())))


if __name__ == "__main__":
    main()
