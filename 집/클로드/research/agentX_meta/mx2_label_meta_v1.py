# -*- coding: utf-8 -*-
"""MX2: TRAINING label meta (F13/F47): quantisation grid, constant runs, midnight jumps on truth links vs within-day,
level shifts (recalibration-like steps) of day-mean OOF residual along truth chains, residual vs chain position/gap."""
import os, sys, pickle
R0 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R0)
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.join(R0, "local", "agentX_meta")


def grid(v, q):
    return np.mean(np.abs(v / q - np.round(v / q)) < 1e-6)


def main():
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y["farm"] = Y.row_id.str[:3]; Y["day"] = Y.row_id.str[4:7].astype(int); Y["hour"] = Y.row_id.str[8:10].astype(int)
    print("rows %d, sub_temp NaN %d, sub_ec NaN %d" % (len(Y), Y.sub_temp.isna().sum(), Y.sub_ec.isna().sum()))
    for c, qs in (("sub_temp", [1, .5, .1, .05, .01, .001]), ("sub_ec", [.1, .01, .005, .001, .0001])):
        v = Y[c].dropna().values
        print("%s grid share: %s" % (c, ", ".join("%g:%.3f" % (q, grid(v, q)) for q in qs)))
        last = np.round(v / qs[-2]).astype(int) % 10; print("  last digit freq:", np.bincount(last, minlength=10) / len(last))
        for f in ("F13", "F47"):
            s = Y[Y.farm == f]; print("  %s min %.3f max %.3f, exact-zero %d, n unique %d" % (f, s[c].min(), s[c].max(), (s[c] == 0).sum(), s[c].nunique()))
    # constant runs within record
    def runs(v):
        best = cur = 1
        for i in range(1, len(v)):
            cur = cur + 1 if (v[i] == v[i - 1]) else 1; best = max(best, cur)
        return best
    G = Y.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"])
    rr = G.agg(rt=("sub_temp", lambda v: runs(v.values)), re=("sub_ec", lambda v: runs(v.values)))
    print("longest constant run per record: temp", rr.rt.value_counts().sort_index().to_dict(), " ec", rr.re.value_counts().sort_index().to_dict())
    # truth links
    Rb = pickle.load(open(os.path.join(R0, "local", "linker_v3", "base.pkl"), "rb")); rec = Rb["recs"].reset_index(drop=True)
    T = pd.read_pickle(os.path.join(R0, "local", "linker_v3", "truth.pkl"))
    key = {i: (rec.farm[i], int(rec.day[i])) for i in range(len(rec))}
    YI = Y.set_index(["farm", "day", "hour"])
    def yv(k, h, c):
        try: return YI.loc[(k[0], k[1], h), c]
        except KeyError: return np.nan
    J = []
    for _, t in T.iterrows():
        b, a = key[int(t.b)], key[int(t.a1)]
        J.append(dict(b=b, a=a, gap=b[1] - a[1] if a[0] == b[0] else np.nan, cross=a[0] != b[0], cert=t.cert,
                      jt=yv(b, 0, "sub_temp") - yv(a, 23, "sub_temp"), je=yv(b, 0, "sub_ec") - yv(a, 23, "sub_ec")))
    J = pd.DataFrame(J); print("truth links %d, cert dist" % len(J), J.cert.value_counts().to_dict() if J.cert.dtype == object or J.cert.nunique() < 10 else J.cert.describe().to_dict())
    Ys = Y.sort_values(["farm", "day", "hour"]); dts = Ys.groupby(["farm", "day"]).sub_temp.diff(); des = Ys.groupby(["farm", "day"]).sub_ec.diff()
    print("midnight jump |23->0| on truth links: temp median %.3f p90 %.3f | within-day |dh| median %.3f p90 %.3f (h0->1: %.3f)" % (
        J.jt.abs().median(), J.jt.abs().quantile(.9), dts.abs().median(), dts.abs().quantile(.9), dts[Ys.hour == 1].abs().median()))
    print("                                    EC   median %.4f p90 %.4f | within-day median %.4f p90 %.4f (h0->1: %.4f)" % (
        J.je.abs().median(), J.je.abs().quantile(.9), des.abs().median(), des.abs().quantile(.9), des[Ys.hour == 1].abs().median()))
    big = J[J.je.abs() > 4 * des.abs().quantile(.9)]; print("  EC midnight jumps > 4x within-day p90: %d of %d links; signed median %+.3f" % (len(big), J.je.notna().sum(), big.je.median() if len(big) else np.nan))
    # chains from truth links
    H = pd.read_pickle(os.path.join(OUT, "resid.pkl"))
    D = H.groupby(["farm", "day"]).agg(rt=("rt", "mean"), re=("re", "mean"), ye=("sub_ec_x" if "sub_ec_x" in H else "sub_ec", "mean")).reset_index()
    DI = D.set_index(["farm", "day"])
    prev = {r.b: r.a for r in J.itertuples()}
    nxt = {}
    for b, a in prev.items(): nxt.setdefault(a, []).append(b)
    starts = [k for k in set(prev) | set(nxt) if k not in prev]
    pos = {}
    for b in prev:  # position = steps back to start (cycle-safe)
        k, n, seen = b, 0, set()
        while k in prev and k not in seen: seen.add(k); k = prev[k]; n += 1
        pos[b] = n
    L = J.copy(); L["pos"] = [pos.get(b, 0) for b in L.b]
    for c in ("rt", "re"):
        L[c + "_b"] = [DI[c].get(b, np.nan) for b in L.b]; L[c + "_a"] = [DI[c].get(a, np.nan) for a in L.a]
    print("\n== residual persistence along truth links (day-mean OOF residual) ==")
    for c in ("rt", "re"):
        ok = L[c + "_b"].notna() & L[c + "_a"].notna(); r, p = spearmanr(L[c + "_a"][ok], L[c + "_b"][ok])
        print("  %s: rho(prev,cur) %+.3f p %.2g n %d" % (c, r, p, ok.sum()))
        # lag-1 autocorr of differences along chain triplets a2->a->b
        trip = [(prev[a], a, b) for b, a in prev.items() if a in prev]
        d1, d2 = [], []
        for x, a, b in trip:
            v = [DI[c].get(k, np.nan) for k in (x, a, b)]
            if not np.any(np.isnan(v)): d1.append(v[1] - v[0]); d2.append(v[2] - v[1])
        r2, p2 = spearmanr(d1, d2); print("     corr(delta_t, delta_t+1) %+.3f p %.2g n %d  (iid spikes -> -0.5, steps/random walk -> 0)" % (r2, p2, len(d1)))
        # step test: after a big change, does the level persist (step) or revert (spike)?
        trip4 = []
        for b, a in prev.items():
            if a in prev and b in nxt:
                for c2 in nxt[b]:
                    v = [DI[c].get(k, np.nan) for k in (prev[a], a, b, c2)]
                    if not np.any(np.isnan(v)): trip4.append(v)
        V = np.array(trip4)
        if len(V):
            base = (V[:, 0] + V[:, 1]) / 2; jump = V[:, 2] - base; keep = (V[:, 3] - base) / np.where(np.abs(jump) > 1e-9, jump, np.nan)
            thr = np.quantile(np.abs(jump), .85); m = np.abs(jump) >= thr
            print("     big jumps (top15%%, |j|>=%.3f) n %d: next-day retention median %.2f (1=step/recalibration, 0=spike)" % (thr, m.sum(), np.nanmedian(keep[m])))
        ok = L[c + "_b"].notna(); r3, p3 = spearmanr(L.pos[ok], L[c + "_b"][ok]); r4, p4 = spearmanr(L.gap[ok & L.gap.notna()], (L[c + "_b"] - L[c + "_a"])[ok & L.gap.notna()].abs())
        print("     residual vs chain position rho %+.3f p %.2g ; |change| vs day gap rho %+.3f p %.2g" % (r3, p3, r4, p4))
        for nm, mk in (("pass1", np.array([b[1] < 179 for b in L.b])), ("pass2", np.array([b[1] >= 179 for b in L.b]))):
            ok2 = ok & L[c + "_a"].notna() & mk
            if ok2.sum() > 10:
                r5, p5 = spearmanr(L[c + "_a"][ok2], L[c + "_b"][ok2]); print("     %s persistence rho %+.3f n %d" % (nm, r5, ok2.sum()))
    # cross-farm (house moved between IDs) links: is the residual jump larger?
    for c in ("rt", "re"):
        dd = (L[c + "_b"] - L[c + "_a"]).abs()
        print("  |delta %s| same-ID link median %.3f vs cross-ID link median %.3f (n %d)" % (c, dd[~L.cross].median(), dd[L.cross].median(), L.cross.sum()))
    L.to_pickle(os.path.join(OUT, "mx2_links.pkl"))


if __name__ == "__main__":
    main()
