# -*- coding: utf-8 -*-
"""MK1: were some 'unexplained' high-EC days actually explainable but MASKED by the other unexplained days?
(2026-10-09 집 클로드, user: "설명 가능한 데이터도 있는데, 생성된 데이터에 가려져서 같이 지워버린 거 아니야?").
Fixed before running.  Diagnostic.

Day sets (hx2_day_sets.json): HIGH 26 days (day-mean EC >= 1.2), UNEX2 17 of them (leak-free OOF residual >= .3),
EXPL = HIGH - UNEX2 (9 days).
For every high day d (26 runs), validation = day d only; exclusions: same farm d+-1, other farm d-3..d+3, lock-40 +-1.
  FULLTR  train on all remaining days                                    (= current way)
  CLEAN   train on all remaining days EXCEPT the other UNEX2 days         (d itself is never in training)
Model: current R3 recipe (.6 ET + .3 LGB-tweedie + .1 MLP; season DC4 + DP1; shrink; clip to train range), seed 47.
Reading (fixed): an UNEX2 day is 'MASKED' iff CLEAN day-mean |error| < .3 AND <= half of FULLTR day-mean |error|.
Reported for all 26 days: label day mean, FULLTR and CLEAN day-mean prediction and error, row RMSE.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u mk1_masked_highec_days_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
OTHER = {"F13": "F47", "F47": "F13"}
SEED = 47
W = (.6, .3, .1)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    sets = json.load(open(os.path.join(env.LOCAL, "hx2_day_sets.json"), encoding="utf-8"))
    high = {(f, int(d)) for f, d in sets["HIGH"]}; unex = {(f, int(d)) for f, d in sets["UNEX2"]}
    assert len(high) == 26 and len(unex) == 17
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    keys = list(zip(lab.farm, lab.day.astype(int)))
    out = []
    for (f, d) in sorted(high):
        va_m = np.array([k == (f, d) for k in keys])
        forb = {(f, d + j) for j in (-1, 0, 1)} | {(OTHER[f], d + j) for j in range(-3, 4)} | \
               {(ff, dd + j) for ff, dd in lock for j in (-1, 0, 1)}
        base_tr = np.array([k not in forb for k in keys])
        res = {}
        for cfg, tr_m in (("FULLTR", base_tr), ("CLEAN", base_tr & np.array([k not in unex for k in keys]))):
            tr, va = lab[tr_m].copy(), lab[va_m].copy()
            tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
            season, vq = dc4.season_index(tdays, vdays, wv)
            tr["season"] = [season[(a, b)] for a, b in zip(tr.farm, tr.day)]; vdays["season"] = vq
            va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
            e, l, m = wt0.members(tr, va, SEED, FS, BS)
            p = np.clip(core.shrink(W[0] * e + W[1] * l + W[2] * m, va), tr.sub_ec.min(), tr.sub_ec.max())
            res[cfg] = p
        y = lab.loc[va_m, "sub_ec"].to_numpy(float)
        row = dict(farm=f, day=d, group="UNEX2" if (f, d) in unex else "EXPL", y_mean=y.mean(),
                   full_mean=res["FULLTR"].mean(), clean_mean=res["CLEAN"].mean(),
                   full_err=res["FULLTR"].mean() - y.mean(), clean_err=res["CLEAN"].mean() - y.mean(),
                   full_rmse=r(res["FULLTR"] - y), clean_rmse=r(res["CLEAN"] - y))
        row["masked"] = row["group"] == "UNEX2" and abs(row["clean_err"]) < .3 and abs(row["clean_err"]) <= .5 * abs(row["full_err"])
        out.append(row)
        print("%s %3d %-5s y %.3f | FULLTR %.3f (%+.3f) | CLEAN %.3f (%+.3f)%s" % (
            f, d, row["group"], row["y_mean"], row["full_mean"], row["full_err"], row["clean_mean"], row["clean_err"],
            "  <- MASKED" if row["masked"] else ""), flush=True)
    O = pd.DataFrame(out); O.to_csv(os.path.join(env.LOCAL, "mk1_days.csv"), index=False)
    for g, G in O.groupby("group"):
        print("\n%s %d days: mean |err| FULLTR %.3f -> CLEAN %.3f; row RMSE %.3f -> %.3f; CLEAN better on %d days"
              % (g, len(G), G.full_err.abs().mean(), G.clean_err.abs().mean(), np.sqrt((G.full_rmse ** 2).mean()),
                 np.sqrt((G.clean_rmse ** 2).mean()), int((G.clean_err.abs() < G.full_err.abs()).sum())))
    print("\nMASKED UNEX2 days: %d of 17 -> %s" % (int(O.masked.sum()), O[O.masked][["farm", "day"]].values.tolist()))


if __name__ == "__main__":
    main()
