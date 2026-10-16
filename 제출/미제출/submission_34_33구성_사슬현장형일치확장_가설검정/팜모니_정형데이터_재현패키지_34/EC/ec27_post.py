# -*- coding: utf-8 -*-
"""submission_27 EC (2026-10-15 집 클로드, LEADERBOARD HYPOTHESIS TEST; '5-point model' with the FIELD linker FL; catalog 6.923-6.926, 6.931):
    EC = NH + FL yesterday-EC anchor,   NH = pre + 0.5 * HR5(pre)   (EV3L removed: FL4 6.923 FAIL)
  pre, HR5: exactly f5e_post.py stage_pre / stage_hr5 (round-9 EC members before SG2 + day-level TabPFN corrector, no SG2/W2/B5).
  Anchor (EA form, catalog EA4): for an evaluation record whose FL hour-0 previous day is a labelled training record with p >= .8 and
    |indoor-air jump 23h -> 0h| <= 1.5 C (온도/linker_v7/ap_fl.py eval -> fl_eval_anchor.csv):  pred_h = q_h + (A - mean(q_0..q_h)),
    A = that record's 23 h EC label.  Widening to p >= .5 was tested and rejected (AC1/AC2, 6.925/6.926).
Run:  python ec27_post.py <fl_eval_anchor.csv>    (writes output_27/ec_pre.csv, ec_nh.csv, ec_nh_fl.csv)
"""
import os, sys, subprocess
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=dict(os.environ, PYTHONHASHSEED="0")))
import numpy as np, pandas as pd
import f5e_post as F5
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def main():
    anchors = sys.argv[1]
    out_dir = os.path.join(HERE, "output_27"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_nh_fl.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    pre = F5.stage_pre(out_dir); nh = F5.stage_hr5(pre, out_dir)
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])]
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    A = pd.read_csv(anchors, encoding="utf-8-sig")
    A = A[(A.steps_to_anchor == 1) & A.prev_is_labelled.astype(bool) & (A.p_h0 >= .8) & (A.dT_air.abs() <= 1.5) & A.anchor_ec23.notna()]
    AN = dict(zip(A.row, A.anchor_ec23))
    F = nh.copy(); F["rec"] = F.row_id.str[:7]; F["h"] = F.row_id.str[8:10].astype(int); F = F.sort_values(["rec", "h"])
    cm = F.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()); a = F.rec.map(AN)
    F["new"] = np.clip(np.where(a.notna(), F.sub_ec + (a - cm), F.sub_ec), lo, hi); F = F.sort_index()
    F[["row_id", "new"]].rename(columns={"new": "sub_ec"}).to_csv(out, index=False, float_format="%.6f")
    d1 = nh.sub_ec.values - pre.sub_ec.values; d = (F.new - F.sub_ec).values
    print("wrote %s | HR5 change RMS %.4f | anchored records %d %s | anchor change RMS %.4f, rows changed %d"
          % (out, np.sqrt((d1 ** 2).mean()), len(AN), sorted(AN), np.sqrt((d ** 2).mean()), int((np.abs(d) > 1e-9).sum())))


if __name__ == "__main__":
    main()
