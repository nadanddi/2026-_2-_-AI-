"""submission_36 EC = ec29_post.py with the anchor list extended by the TAG2 field two-view agreement anchors (catalog 7.044 ECT2: P2LOO/EL1 pass-2 6/6 better, P .0026, 0 harmed;
DIAG10 pass-2 -.22%; DIAG10 all-rows guard failed on F47_132 pass-1).  Same strength .8 x p0.  Original description follows.
"""
# -*- coding: utf-8 -*-
"""submission_29 EC (2026-10-16 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.956 CB4-EC, 6.988 DGX4):
    EC = NH + 0.8 * p0 * (FL yesterday-EC anchor),   NH = pre + 0.5 * HR5(pre)   (same NH as round 27, ec27_post.py)
  Round 27 used the full anchor  pred_h = q_h + (A - mean(q_0..q_h)).  Here the anchor term is scaled by 0.8 x the linker's hour-0
  probability p0 (CB4-EC selection on DIAG10: strength .8, constant over the day, probability-proportional ON).  Same anchor records
  (p0 >= .8, labelled training previous day, |dT_air| <= 1.5) so the effective strength is .64-.80.
  Local evidence (CB4-EC, vs round-27 rule): P2LOO pass-2 -3.2 %, EL1 -4.5 %, 6/6 cells better, but P2LOO block P .35 (anchor-only changes
  touch few records) -> FAIL on the local rule; submitted as a hypothesis test.
Run:  python ec29_post.py <fl_eval_anchor.csv>    (writes output_36/ec_pre.csv, ec_nh.csv, ec_nh_fl.csv)
"""
import os, sys, subprocess
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=dict(os.environ, PYTHONHASHSEED="0")))
import numpy as np, pandas as pd
import f5e_post as F5
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
STRENGTH = 0.8


def main():
    anchors = sys.argv[1]
    out_dir = os.path.join(HERE, "output_36"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_nh_fl.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    pre = F5.stage_pre(out_dir); nh = F5.stage_hr5(pre, out_dir)
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])]
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    A = pd.read_csv(anchors, encoding="utf-8-sig")
    A = A[(A.steps_to_anchor == 1) & A.prev_is_labelled.astype(bool) & (A.p_h0 >= .5) & (A.dT_air.abs() <= 1.5) & A.anchor_ec23.notna()]   # ECT2: input = fl_eval_anchor_tag2.csv (p0 >= .8 anchors + TAG2 field-agreement anchors .5 <= p0 < .8)
    AN = dict(zip(A.row, A.anchor_ec23)); PP = dict(zip(A.row, A.p0))
    F = nh.copy(); F["rec"] = F.row_id.str[:7]; F["h"] = F.row_id.str[8:10].astype(int); F = F.sort_values(["rec", "h"])
    cm = F.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()); a = F.rec.map(AN); wgt = STRENGTH * F.rec.map(PP)
    F["new"] = np.clip(np.where(a.notna(), F.sub_ec + wgt * (a - cm), F.sub_ec), lo, hi); F = F.sort_index()
    F[["row_id", "new"]].rename(columns={"new": "sub_ec"}).to_csv(out, index=False, float_format="%.6f")
    d = (F.new - F.sub_ec).values
    print("wrote %s | anchored records %d | effective strength %s | anchor change RMS %.4f, rows changed %d"
          % (out, len(AN), {k: round(STRENGTH * v, 3) for k, v in sorted(PP.items())}, np.sqrt((d ** 2).mean()), int((np.abs(d) > 1e-9).sum())))


if __name__ == "__main__":
    main()
