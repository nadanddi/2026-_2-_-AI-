# -*- coding: utf-8 -*-
"""submission_19 EC (2026-10-11 집 클로드, LEADERBOARD HYPOTHESIS TEST): 9th-round EC (submission_14, reproduced exactly from
output_14) + W2 structural rule ONLY (no B4, no HG3).  W2 = pure pass-2 window 'same ID record number - 2 = same house previous
date' (catalog 6.549 / 6.550; functions from c18post.py, unchanged).  Separates W2 from B4 + HG3 (round 11 had all three).
Run:  python w2only_post.py   (writes output_19/ec_w2only.csv)
"""
import os
import numpy as np, pandas as pd
import c18post as C


def main():
    out_dir = os.path.join(C.HERE, "output_19"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_w2only.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    z = np.load(os.path.join(C.HERE, "output_14", "prediction_details.npz")); rid = z["row_id"]
    s2 = np.load(os.path.join(C.HERE, "output_14", "sg2_details.npz")); assert (s2["row_id"] == rid).all()
    F = pd.DataFrame({"row_id": rid, "farm": [r[:3] for r in rid], "day": [int(r[4:7]) for r in rid], "hour": [int(r[8:10]) for r in rid]})
    ty = pd.read_csv(os.path.join(C.DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    ty["farm"], ty["day"], ty["hour"] = ty.row_id.str[:3], ty.row_id.str[4:7].astype(int), ty.row_id.str[8:10].astype(int)
    ec = ty.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index); ec23 = ty[ty.hour == 23].set_index(["farm", "day"]).sub_ec
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    F["p14pre"] = np.clip(C.shrink(.8 * r3 + .2 * pfn, F), lo, hi)
    Rr, WV, hrs, SIG, DM = C.prepare_structure()
    vd = set(zip(F.farm, F.day)); cal = C.ref_calendar(Rr, WV, labset); ST = C.s_low_table(DM, labset)
    sg14, _ = C.correction(F, "p14pre", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    F["ec14"] = np.clip(sg14, lo, hi)
    rep = bool(np.abs(F.ec14.values - s2["pred_final"]).max() < 1e-12)
    ids = pd.concat([pd.read_csv(os.path.join(C.DATA, f), usecols=["row_id"]) for f in ("train_X.csv", "test_X.csv")]).row_id
    ids = ids[ids.str[:3].isin(["F13", "F47"])]
    exist = {(a, int(b)) for a, b in zip(ids.str[:3], ids.str[4:7])}
    w, fired = C.w2(F, "ec14", WV, hrs, ec23, labset, exist)
    F["sub_ec"] = np.clip(w, lo, hi)
    F[["row_id", "sub_ec"]].to_csv(out, index=False, float_format="%.6f")
    d = F.sub_ec - s2["pred_final"]
    print("reproduces submission_14 EC %s (must be True on the original data); W2 fired on %s" % (rep, sorted({(f, d_) for f, d_, h in fired})))
    print("wrote %s | change vs submission_14 EC: RMS %.4f, rows changed %d" % (out, np.sqrt((d ** 2).mean()), int((d.abs() > 1e-9).sum())))


if __name__ == "__main__":
    main()
