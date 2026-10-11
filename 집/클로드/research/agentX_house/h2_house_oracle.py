# H-HOUSE (2026-10-11 agentX): is the day offset a persistent house property beyond lag-1?  Diagnostic, training labels only.
# Chains = connected components of certain truth links a1->b (linker_v3 truth.pkl, cert==True; cross-ID links kept).
# Offset d_r = record mean of (y - q) on DIAG10 OOF (temp: each of 4 cells; EC: CUR seed-mean sg).
# Fixed before looking: predictors (i) lag1, (ii) LOO chain mean, (iii) causal EW lambda {.5,.7,.9} with gap g {1,2,3,5},
# (iv) chain mean excluding |j-i|<=k, k {1,2,3,5}.  Shrink beta chosen on pass-1 records (LS), judged on pass-2; also beta=1.
import sys, os, pickle
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

LOOSE = len(sys.argv) > 1 and sys.argv[1] == "loose"
HERE = r"C:\work\farmai\집\클로드\research"; OUT = os.path.join(HERE, "agentX_house", "local")
R = pickle.load(open(os.path.join(HERE, "local", "linker_v3", "base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
rid = np.array([f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]); N = len(r)
T = pd.read_pickle(os.path.join(HERE, "local", "linker_v3", "truth.pkl"))
if LOOSE:   # all best links, resolve duplicate a1 by lowest c1 (sensitivity only)
    C = T.sort_values("c1").drop_duplicates("a1")
    C = C[C.c1 <= 1.0]
else:
    C = T[T.cert]
print("MODE", "loose" if LOOSE else "cert", "links", len(C))
nxt = {int(a): int(b) for a, b in zip(C.a1, C.b)}; prv = {b: a for a, b in nxt.items()}
assert len(nxt) == len(C) and len(prv) == len(C)
chain = -np.ones(N, int); pos = np.zeros(N, int); chains = []
for i in range(N):
    if i in prv or chain[i] >= 0 or i not in nxt:
        continue
    seq = [i]
    while seq[-1] in nxt:
        seq.append(nxt[seq[-1]])
    for k, j in enumerate(seq):
        chain[j] = len(chains); pos[j] = k
    chains.append(seq)
left = [i for i in prv if chain[i] < 0]; print("unassigned (cycles?)", len(left))
L = np.array([len(s) for s in chains])
print(f"chains {len(chains)}, records in chains {L.sum()}, length dist {pd.Series(L).value_counts().sort_index().to_dict()}")
xid = [sum(r.farm[s[k]] != r.farm[s[k + 1]] for k in range(len(s) - 1)) for s in chains]
print("cross-ID links in chains", sum(xid), "chains with >=1 cross-ID", sum(x > 0 for x in xid))
p2span = [(r.day.values[s] >= 179).mean() for s in chains]
print("chains: all pass1 %d, all pass2 %d, mixed %d" % (sum(p == 0 for p in p2span), sum(p == 1 for p in p2span), sum(0 < p < 1 for p in p2span)))
ntest = sum((r.split.values[s] != "train").sum() for s in chains); print("test records inside chains", ntest)
for s in sorted(chains, key=len)[-5:]:
    print("  long chain:", " ".join(rid[s]))


def offsets(df, ycol, qcols):
    key = df.farm + "_" + df.day.astype(int).map("{:03d}".format)
    return {qc: (df[ycol] - df[qc]).groupby(key).mean() for qc in qcols}


TP = pd.read_csv(os.path.join(OUT, "temp_oof_diag10.csv")); EP = pd.read_csv(os.path.join(OUT, "ec_oof_diag10.csv"))
TP["qm"] = TP[[f"q{k}" for k in range(4)]].mean(1)
TOFF = offsets(TP, "sub_temp", ["qm", "q0", "q1", "q2", "q3"]); EOFF = offsets(EP, "sub_ec", ["sg"])
id2i = {s: i for i, s in enumerate(rid)}
P2 = (r.day.values >= 179)


def dvec(ser):
    v = np.full(N, np.nan)
    for k, x in ser.items():
        v[id2i[k]] = x
    return v


def decomp(name, d):
    print(f"\n=== {name}: decomposition ===")
    for pn, pm in (("pass1", ~P2), ("pass2", P2), ("all", np.ones(N, bool))):
        m = pm & ~np.isnan(d) & (chain >= 0)
        ch = chain[m]; dv = d[m]; tot = np.var(dv)
        cnt = pd.Series(dv).groupby(ch).size(); keep = cnt[cnt >= 2].index
        mm = np.isin(ch, keep); dv2 = dv[mm]; ch2 = ch[mm]; g2 = pd.Series(dv2).groupby(ch2)
        k = len(keep); n = len(dv2); gm = dv2.mean()
        if k >= 2:
            msb = (g2.size() * (g2.mean() - gm) ** 2).sum() / (k - 1); msw = ((dv2 - g2.transform("mean").values) ** 2).sum() / max(n - k, 1)
            n0 = (n - (g2.size() ** 2).sum() / n) / (k - 1); icc = (msb - msw) / (msb + (n0 - 1) * msw)
        else:
            icc = np.nan
        print(f"  {pn}: offset-records in chains {m.sum()}, chains(>=2) {k} n {n}, sd {np.sqrt(tot):.3f}, ICC(between-chain) {icc:.3f}")
        row = []
        for lag in (1, 2, 3, 5, 10):
            a, b = [], []
            for s in chains:
                for i in range(len(s) - lag):
                    x, y = s[i], s[i + lag]
                    if pm[y] and not np.isnan(d[x]) and not np.isnan(d[y]):
                        a.append(d[x]); b.append(d[y])
            row.append(f"lag{lag} n{len(a)} r{pearsonr(a, b)[0]:+.2f} rho{spearmanr(a, b)[0]:+.2f}" if len(a) >= 5 else f"lag{lag} n{len(a)}")
        print("   ", " | ".join(row))
        for lag in (2, 3, 5):
            X = []
            for s in chains:
                for i in range(lag, len(s)):
                    t_, a1, ak = s[i], s[i - 1], s[i - lag]
                    if pm[t_] and not np.isnan(d[[t_, a1, ak]]).any():
                        X.append((d[t_], d[a1], d[ak]))
            if len(X) >= 8:
                X = np.array(X)
                e0 = X[:, 0] - np.polyval(np.polyfit(X[:, 1], X[:, 0], 1), X[:, 1]); ek = X[:, 2] - np.polyval(np.polyfit(X[:, 1], X[:, 2], 1), X[:, 1])
                pr = pearsonr(e0, ek); print(f"    partial corr(d_t, d_t-{lag} | d_t-1) n{len(X)} {pr[0]:+.2f} (p {pr[1]:.3f})")
        a, b = [], []
        idx = np.where(m)[0]
        for x in idx:
            for y in idx:
                if y <= x or chain[x] == chain[y] or r.farm[x] != r.farm[y] or abs(r.day[x] - r.day[y]) > 2:
                    continue
                a.append(d[x]); b.append(d[y])
        if len(a) > 5:
            print(f"    control other-chain same-ID |dno|<=2: n{len(a)} r{pearsonr(a, b)[0]:+.2f}")


def preds(d, i, avail):
    s = chains[chain[i]]; p = pos[i]

    def ok(j):
        return avail[j] and not np.isnan(d[j])
    out = {}
    j = s[p - 1] if p >= 1 else None
    out["lag1"] = d[j] if j is not None and ok(j) else np.nan
    oth = [d[x] for x in s if x != i and ok(x)]; out["loo_mean"] = np.mean(oth) if oth else np.nan
    for lam in (.5, .7, .9):
        for g in (1, 2, 3, 5):
            w = [(lam ** (p - q), d[s[q]]) for q in range(0, p - g + 1) if ok(s[q])]
            out[f"ew{lam}_g{g}"] = sum(a * b for a, b in w) / sum(a for a, _ in w) if w else np.nan
    for k in (1, 2, 3, 5):
        oth = [d[s[q]] for q in range(len(s)) if abs(q - p) > k and ok(s[q])]; out[f"excl{k}"] = np.mean(oth) if oth else np.nan
    dist = [p - q for q in range(p) if ok(s[q])]; out["dist_prev"] = min(dist) if dist else np.nan
    return out


def gains(name, d, rows_df, ycol, qcols):
    print(f"\n=== {name}: oracle gains (row RMSE on records having that predictor; beta LS on pass-1 records, judged pass-2; beta=1 too) ===")
    avail = ~np.isnan(d)
    recs = [i for i in range(N) if chain[i] >= 0 and not np.isnan(d[i])]
    PR = pd.DataFrame([dict(i=i, **preds(d, i, avail)) for i in recs]); PR["d"] = d[PR.i]; PR["p2"] = P2[PR.i]; PR["rid"] = rid[PR.i]
    key = (rows_df.farm + "_" + rows_df.day.astype(int).map("{:03d}".format)).values
    pcols = [c for c in PR.columns if c not in ("i", "d", "p2", "rid", "dist_prev")]
    res = []
    for qc in qcols:
        e = (rows_df[ycol] - rows_df[qc]).values
        for pc in pcols:
            sub = PR[PR[pc].notna()]
            s1 = sub[~sub.p2]; beta = float((s1[pc] * s1.d).sum() / (s1[pc] ** 2).sum()) if len(s1) > 3 else np.nan
            for pn, ss in (("p1", sub[~sub.p2]), ("p2", sub[sub.p2])):
                if len(ss) < 3:
                    continue
                mp = dict(zip(ss.rid, ss[pc])); rows = np.isin(key, list(mp.keys()))
                off = np.array([mp[k] for k in key[rows]]); e0 = e[rows]; b0 = np.sqrt(np.mean(e0 ** 2))
                res.append(dict(q=qc, pred=pc, pass_=pn, nrec=len(ss), beta=beta, base=b0,
                                g_beta=np.sqrt(np.mean((e0 - beta * off) ** 2)) / b0 - 1, g_1=np.sqrt(np.mean((e0 - off) ** 2)) / b0 - 1,
                                corr=np.corrcoef(ss[pc], ss.d)[0, 1]))
    D = pd.DataFrame(res)
    agg = D.groupby(["pred", "pass_"]).agg(nrec=("nrec", "first"), base=("base", "mean"), beta=("beta", "mean"), corr=("corr", "mean"),
                                            g_beta=("g_beta", "mean"), g_beta_worst=("g_beta", "max"), g_1=("g_1", "mean")).reset_index()
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:+.3f}".format):
        print(agg.to_string(index=False))
    tag = name + ("_loose" if LOOSE else "")
    D.to_csv(os.path.join(OUT, f"gains_{tag}.csv"), index=False); PR.to_csv(os.path.join(OUT, f"preds_{tag}.csv"), index=False)
    for pc in ("ew0.7_g2", "ew0.9_g2", "excl1", "loo_mean"):
        sub = PR[PR.lag1.notna() & PR[pc].notna()]
        s1, s2 = sub[~sub.p2], sub[sub.p2]
        if len(s1) < 5:
            continue
        b = np.linalg.lstsq(s1[["lag1", pc]].values, s1.d.values, rcond=None)[0]; b1 = (s1.lag1 * s1.d).sum() / (s1.lag1 ** 2).sum()

        def rr(ss, f):
            return np.sqrt(np.mean((ss.d.values - f) ** 2))
        out = [f"{pn}: n{len(ss)} none {rr(ss, 0):.3f} lag1 {rr(ss, b1 * ss.lag1.values):.3f} lag1+{pc} {rr(ss, ss[['lag1', pc]].values @ b):.3f}"
               for pn, ss in (("p1", s1), ("p2", s2)) if len(ss)]
        print(f"  [record offset RMSE] beta lag1 {b1:+.2f}, joint {b[0]:+.2f},{b[1]:+.2f} :: " + " ; ".join(out))
    print("  dist_prev (nearest earlier labelled chain member):", PR.dist_prev.value_counts(dropna=False).sort_index().to_dict())
    return PR


for name, OFF, df, yc, qcs in (("temp", TOFF, TP, "sub_temp", ["q0", "q1", "q2", "q3"]), ("ec", EOFF, EP, "sub_ec", ["sg"])):
    dq = dvec(OFF["qm" if name == "temp" else "sg"])
    print(f"\n##### {name}: records with offset {np.isfinite(dq).sum()}, in chains {(np.isfinite(dq) & (chain >= 0)).sum()}")
    decomp(name, dq)
    gains(name, dq, df, yc, qcs)
pd.DataFrame(dict(rid=rid, chain=chain, pos=pos)).to_csv(os.path.join(OUT, "chains_%s.csv" % ("loose" if LOOSE else "cert")), index=False)
