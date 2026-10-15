# -*- coding: utf-8 -*-
"""submission_28 draft EC (2026-10-15 집 클로드; catalog 6.933 KE1 KEA).
EC = NH + FL yesterday-EC anchor (exactly ec27_post.py: K8 = FL p >= .8, labelled training previous day, |dT_air| <= 1.5)
     + for records WITHOUT a K8 anchor: FLN state chain (KEA):
       r -> p1 -> p2 -> p3 via the FLN evaluation picks (ap_fln.py eval: each evaluation record decided independently at hour 0, 'no previous day' option),
       at most 3 steps, cumulative P = prod p0 >= .5; the chain must end at a LABELLED training record (A23 = its 23 h EC label);
       intermediate records must be SAME-FARM, EARLIER-NUMBERED evaluation records (their own NH prediction is used: est23 <- NH_m23 + (est23 - mean NH_m)).
       pred_h = NH_h + P * (est23 - mean(NH_0..h)).
  Local evidence (KE1, EC field rule): P2LOO pass-2 -2.0..-2.2 % (P .0037), DIAG10 -0.3 %, EL1 ~0; the gain comes mostly from weak 1-step FLN anchors.
Run:  python ec28_post.py <fl_eval_anchor.csv> <fln_eval_links.csv>     (needs output_27/ec_nh.csv from ec27_post.py; writes output_28/ec_nh_fl_kea.csv)
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def main():
    k8p, flnp = sys.argv[1:3]
    out_dir = os.path.join(HERE, "output_28"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_nh_fl_kea.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    nh = pd.read_csv(os.path.join(HERE, "output_27", "ec_nh.csv"))
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])]
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    ty["rec"] = ty.row_id.str[:7]; ty["h"] = ty.row_id.str[8:10].astype(int); EC23 = ty[ty.h == 23].set_index("rec").sub_ec.to_dict()
    A = pd.read_csv(k8p, encoding="utf-8-sig")
    A = A[(A.steps_to_anchor == 1) & A.prev_is_labelled.astype(bool) & (A.p_h0 >= .8) & (A.dT_air.abs() <= 1.5) & A.anchor_ec23.notna()]
    K8 = dict(zip(A.row, A.anchor_ec23))
    L = pd.read_csv(flnp); LINK = {r: (p, q) for r, p, q in zip(L.rec, L.prev0, L.p0) if isinstance(p, str)}
    F = nh.copy(); F["rec"] = F.row_id.str[:7]; F["h"] = F.row_id.str[8:10].astype(int); F = F.sort_values(["rec", "h"])
    NH = {r: g.sub_ec.values for r, g in F.groupby("rec")}; evr = set(NH)
    new = F.sub_ec.values.copy(); pos = {r: np.where(F.rec.values == r)[0] for r in evr}; log = []
    for r in sorted(evr):
        x = NH[r]; cm = np.cumsum(x) / np.arange(1, len(x) + 1); ix = pos[r]
        if r in K8: new[ix] = x + (K8[r] - cm); continue
        chain, P, cur = [], 1.0, r
        for step in range(3):
            if cur not in LINK: chain = None; break
            pv, q = LINK[cur]; P *= q
            if P < .5: chain = None; break
            chain.append(pv)
            if pv in EC23: break
            if not (pv in evr and pv[:3] == r[:3] and int(pv[4:7]) < int(r[4:7])): chain = None; break
            cur = pv
        if not chain or chain[-1] not in EC23: continue
        est = float(EC23[chain[-1]])
        for m in reversed(chain[:-1]): est = NH[m][-1] + (est - NH[m].mean())
        new[ix] = x + P * (est - cm); log.append((r, len(chain), round(P, 3), chain[-1]))
    F["new"] = np.clip(new, lo, hi); F = F.sort_index()
    F[["row_id", "new"]].rename(columns={"new": "sub_ec"}).to_csv(out, index=False, float_format="%.6f")
    d = F.new - F.sub_ec
    print("wrote %s | K8 anchors %d | KEA records %d %s | change vs NH RMS %.4f" % (out, len(K8), len(log), log, np.sqrt((d ** 2).mean())))


if __name__ == "__main__":
    main()
