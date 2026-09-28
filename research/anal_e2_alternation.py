# -*- coding: utf-8 -*-
"""EC gap analysis, step 2: the every-other-day structure and the test layout.

Daily-mean EC autocorrelation is weak at lag 1 and strong at lag 2 (step 1),
consistent with two source greenhouses alternating day by day.

  1. test layout: test blocks per greenhouse, labelled days around them,
     parity of the nearest labelled days
  2. how well does the daily mean of day d-2 / d-1 / d-4 (labelled) predict
     day d, against the round-3 OOF daily-mean error (analysis only: whether
     earlier labels may be used as features is a rules question, not decided)
  3. do the model inputs already carry the alternation? (daily-mean OOF
     prediction autocorrelation, fingerprint features)
  4. where are the high-EC days and the noisy days in the record

Run:  cd research && PYTHONPATH="" <python> -u anal_e2_alternation.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import TARGET_FARMS
from harness import load
import train_flags_v6 as TF


def blocks(days):
    days = sorted(days)
    out, s = [], days[0]
    for a, b in zip(days, days[1:] + [None]):
        if b is None or b != a + 1:
            out.append((s, a))
            if b is not None:
                s = b
    return out


def main():
    panel, _, lab = load()
    z = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
    oof = pd.Series(z["oof"], index=z["row_id"])
    lab = lab.copy()
    lab["p"] = oof.loc[lab.row_id].values
    test = panel[panel.is_test]

    print("== 1. layout ==")
    for f in TARGET_FARMS:
        td = sorted(test[test.farm == f].day.unique())
        ld = set(lab[lab.farm == f].day.unique())
        print("%s test blocks %s" % (f, blocks(td)))
        l2 = sorted(d for d in ld if d >= 170)
        print("   labelled days >=170: %s" % blocks(l2))

    print("\n== 2. daily-mean predictability ==")
    rows = []
    for f in TARGET_FARMS:
        g = lab[lab.farm == f].groupby("day").agg(y=("sub_ec", "mean"), p=("p", "mean"))
        for d in g.index:
            r = dict(farm=f, day=d, y=g.y[d], p=g.p[d], pass2=d >= 179)
            for k in (1, 2, 3, 4, 6):
                r["lag%d" % k] = g.y.get(d - k, np.nan)
            rows.append(r)
    D = pd.DataFrame(rows)
    for scope, m in (("all", np.ones(len(D), bool)), ("pass2", D.pass2.values)):
        print("-- %s --" % scope)
        base = D[m]
        print("   model OOF daily-mean RMSE %.3f (n=%d)" % (np.sqrt(((base.p - base.y) ** 2).mean()), len(base)))
        for k in (1, 2, 3, 4, 6):
            s = base[base["lag%d" % k].notna()]
            print("   lag%d: n=%3d  RMSE(label lag) %.3f  | model on same days %.3f | corr(y, lag) %.2f"
                  % (k, len(s), np.sqrt(((s["lag%d" % k] - s.y) ** 2).mean()),
                     np.sqrt(((s.p - s.y) ** 2).mean()), np.corrcoef(s.y, s["lag%d" % k])[0, 1]))

    print("\n== 3. does the model carry the alternation? ==")
    for f in TARGET_FARMS:
        g = D[D.farm == f].set_index("day").reindex(range(int(D.day.min()), int(D.day.max()) + 1))
        print("%s  autocorr lag1/lag2:  label %.2f / %.2f | OOF pred %.2f / %.2f | OOF error %.2f / %.2f"
              % (f, g.y.autocorr(1), g.y.autocorr(2), g.p.autocorr(1), g.p.autocorr(2),
                 (g.p - g.y).autocorr(1), (g.p - g.y).autocorr(2)))
    # parity of error: are errors of day d and d-2 correlated (same source) more than d-1?
    print("\n== 4. high-EC days and noisy days along the record ==")
    nd = TF.noisy_days()
    D = D.merge(nd[["farm", "day", "noisy", "ac1"]], on=["farm", "day"], how="left")
    D["dbin"] = pd.cut(D.day, [0, 60, 120, 150, 178, 210, 250])
    print(D.groupby(["farm", "dbin"], observed=True).agg(n=("y", "size"), high=("y", lambda s: int((s > 1.2).sum())),
                                                          noisy=("noisy", "sum"), y=("y", "mean")).to_string())
    print("\nnoisy vs clean days, daily-mean EC: noisy %.3f clean %.3f | corr(ac1, y) %.2f"
          % (D[D.noisy == True].y.mean(), D[D.noisy == False].y.mean(), D[["ac1", "y"]].corr().iloc[0, 1]))
    # within the same day-bin, does noisy matter for EC error?
    D["ae"] = (D.p - D.y).abs()
    print(D.groupby(["dbin", "noisy"], observed=True).ae.agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
