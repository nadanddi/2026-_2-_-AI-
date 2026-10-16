# -*- coding: utf-8 -*-
"""submission_26 EC final stage (2026-10-15 집 클로드, '5-point / field-information' build; EA4):
    pred_h = F5E_h + (A - running mean of F5E over hours 0..h of the record)      (shape kept, level moved to yesterday's measurement)
  F5E = output_25/ec_f5e.csv (f5e_post.py: pre-SG2 stage + HR5 + EV3L 10%).
  A   = the 23 h EC label of the record's previous day found by linker v7 (온도/linker_v7 ap_v7.py -> eval60_confident_1step_anchor_v7.csv:
        each evaluation record decides INDEPENDENTLY at hour 0 from its own hour-0 inputs, training records (labels allowed) and same-farm
        earlier-day records; no other-farm evaluation inputs, no later inputs, no decisions carried between evaluation records),
        used only when: previous day is a labelled TRAINING record, one step, p >= .8, |in_temp(0 h) - in_temp(prev 23 h)| <= 1.5 (EA1/EA2/EA4 rule).
  In the field this is 'yesterday's measured EC of the same bed'; here the linker restores which record is yesterday (adapter).
  Clipped to the training EC range.  Uses only the record's own predictions up to hour h -> causal.
Local evidence (EA4, SZ2 independent bases): vs F5E pass-2 DIAG10 -7.0..-7.4%, EL1 -25.9..-26.3%, P2LOO -8.8..-9.2% (P .0000, DIAG10 harm guard pass).
Run:  python ea26_post.py <eval60_confident_1step_anchor_v7.csv>     (reads output_25/ec_f5e.csv; writes output_26/ec_f5e_ea.csv)
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def main():
    anchors = sys.argv[1]
    out_dir = os.path.join(HERE, "output_26"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_f5e_ea.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    F = pd.read_csv(os.path.join(HERE, "output_25", "ec_f5e.csv"))
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])]
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    A = pd.read_csv(anchors, encoding="utf-8-sig")
    A = A[(A.steps_to_anchor == 1) & A.prev_is_labelled.astype(bool) & (A.p_h0 >= .8) & (A.dT_air.abs() <= 1.5) & A.anchor_ec23.notna()]
    AN = dict(zip(A.row, A.anchor_ec23))
    F["rec"] = F.row_id.str[:7]; F["h"] = F.row_id.str[8:10].astype(int); F = F.sort_values(["rec", "h"])
    cm = F.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean()); a = F.rec.map(AN)
    F["new"] = np.clip(np.where(a.notna(), F.sub_ec + (a - cm), F.sub_ec), lo, hi); F = F.sort_index()
    F[["row_id", "new"]].rename(columns={"new": "sub_ec"}).to_csv(out, index=False, float_format="%.6f")
    d = F.new - F.sub_ec
    print("wrote %s | anchored records %d %s | change vs F5E RMS %.4f, rows changed %d" % (out, len(AN), sorted(AN), np.sqrt((d ** 2).mean()), int((d.abs() > 1e-9).sum())))


if __name__ == "__main__":
    main()
