# H-HOUSE h3: who drives the EC pass-2 gain of chain info (excl1 / causal ew0.9_g2) while lag1 hurts?  controls + record bootstrap.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
OUT = r"C:\work\farmai\집\클로드\research\agentX_house\local"
EP = pd.read_csv(os.path.join(OUT, "ec_oof_diag10.csv")); EP["rid"] = EP.farm + "_" + EP.day.map("{:03d}".format)
day = EP.groupby("rid").agg(d=("sub_ec", "mean"), q=("sg", "mean"), y=("sub_ec", "mean"))
day["d"] = EP.assign(e=EP.sub_ec - EP.sg).groupby("rid").e.mean()
rng = np.random.default_rng(7)
for mode in ("cert", "loose"):
    PR = pd.read_csv(os.path.join(OUT, f"preds_ec{'_loose' if mode == 'loose' else ''}.csv"))
    CH = pd.read_csv(os.path.join(OUT, f"chains_{mode}.csv")).set_index("rid")
    p2 = PR[PR.p2].copy()
    allp2 = PR[PR.p2]
    p2["loo_p2mean"] = [allp2[allp2.rid != x].d.mean() for x in p2.rid]
    # chain mates in pass1 only (far, older/newer) vs pass2 mates
    def mates(x, which):
        c = CH.loc[x, "chain"]; m = CH[(CH.chain == c) & (CH.index != x)]
        m = m[[(int(i[4:]) >= 179) == (which == "p2") for i in m.index]]
        m = m[m.index.isin(day.index)]
        p = CH.loc[x, "pos"]; m = m[(m.pos - p).abs() > 1]
        return day.loc[m.index, "d"].mean() if len(m) else np.nan
    p2["mate_p1"] = [mates(x, "p1") for x in p2.rid]; p2["mate_p2"] = [mates(x, "p2") for x in p2.rid]
    print(f"\n#### {mode}: pass-2 EC records in chains {len(p2)}; mean d {p2.d.mean():+.3f}, sd {p2.d.std():.3f}")
    with pd.option_context("display.width", 220, "display.max_rows", 100, "display.float_format", "{:+.3f}".format):
        cols = ["rid", "d", "lag1", "ew0.9_g2", "excl1", "mate_p1", "mate_p2", "loo_p2mean"]
        print(p2[cols].sort_values("rid").to_string(index=False))
    for pc in ("lag1", "ew0.9_g2", "excl1", "mate_p1", "mate_p2", "loo_p2mean"):
        s = p2[p2[pc].notna()]
        if len(s) < 4: continue
        c = np.corrcoef(s[pc], s.d)[0, 1]
        # record-level: offset-MSE change using beta=.5 fixed (pre-set) ; bootstrap over records P(worse)
        dl = (s.d - .5 * s[pc]) ** 2 - s.d ** 2
        bs = np.array([dl.values[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)])
        top = dl.sort_values().values[:3].sum() / dl.sum() if dl.sum() < 0 else np.nan
        print(f"  {pc:11s} n{len(s):3d} corr {c:+.2f}  beta.5 offsetMSE {s.d.pow(2).mean():.4f}->{((s.d-.5*s[pc])**2).mean():.4f}  P(worse) {np.mean(bs>=0):.3f}  share of gain from best 3 recs {top:.2f}  improved {int((dl<0).sum())}/{len(dl)}")
