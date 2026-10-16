def huber(x, y, d=1.345):
    b = (x*y).sum()/(x*x).sum()
    for _ in range(50):
        r = y - b*x; s = np.median(np.abs(r))/.6745 + 1e-9; wt = np.minimum(1, d*s/np.maximum(np.abs(r), 1e-12))
        b = (wt*x*y).sum()/(wt*x*x).sum()
    return b
def ls(x, y): return (x*y).sum()/(x*x).sum()
def trim(x, y):
    o = np.argsort(-np.abs(x)); k = max(1, int(round(.1*len(x)))); keep = o[k:]; return ls(x[keep], y[keep])
def theil(x, y):
    m = np.abs(x) > .02; return float(np.median(y[m]/x[m]))
def evalw(A, mode, excl=()):
    T = pd.DataFrame([(t, p, V.at[t, "fold"], V.at[t, "chunk"]) for t, p in A.items()], columns=["t", "p", "fold", "chunk"])
    out = {}; base = {}; wl = []
    for sd, G in R.groupby("seed"):
        P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
        def xy(df): return P.loc[df.p, 23].values - P.loc[df.t, 0].values, Y.loc[df.t, 0].values - P.loc[df.t, 0].values
        wk = {}
        if mode[0] == "const": wk = {k: mode[1] for k in T.fold.unique()}
        elif mode[0] == "lofo_med":
            sl = []
            for k in sorted(T.fold.unique()):
                oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
                if len(oth) < 3: continue
                x, y = xy(oth); sl.append(ls(x, y))
            w = float(np.clip(np.median(sl), 0, 1)); wk = {k: w for k in T.fold.unique()}
            if sd == list(R.seed.unique())[0]: print("   lofo slopes", np.round(sl, 2))
        elif mode[0] == "nested_med":
            for k in sorted(T.fold.unique()):
                oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]; sl = []
                for j in sorted(oth.fold.unique()):
                    o2 = oth[oth.fold != j]
                    if len(o2) < 3: continue
                    x, y = xy(o2); sl.append(ls(x, y))
                wk[k] = float(np.clip(np.median(sl), 0, 1)) if sl else .5
        else:
            f = dict(ls=ls, huber=huber, trim=trim, theil=theil)[mode[0]]
            for k in sorted(T.fold.unique()):
                oth = T[(T.fold != k) & (T.t.str[4:7].astype(int) < 179)]
                if len(oth) < 3: wk[k] = .5; continue
                x, y = xy(oth); wk[k] = float(np.clip(f(x, y), 0, 1))
        wl.append(wk)
        for r in T.itertuples(): new.loc[r.t] = P.loc[r.t].values + c * wk[r.fold] * (P.at[r.p, 23] - P.at[r.t, 0])
        out[sd] = new
    recs = [r for r in T.t if int(r[4:7]) < 179 and r not in excl]
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(axis=1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(axis=1)).values for s in out], 0), index=recs)
    cl = per.groupby(T.set_index("t").chunk.reindex(recs).values).sum(); rg = np.random.default_rng(20261226); ii = rg.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    ws = sorted(set(round(v, 2) for d in wl for v in d.values()))
    return cells, pw, per, ws, T
import warnings; warnings.filterwarnings("ignore")
for nm, A in (("SAC1", A1), ("SA0", A0)):
    print("=====", nm)
    T0 = pd.DataFrame([(t, p, V.at[t, "fold"]) for t, p in A.items()], columns=["t", "p", "fold"]).set_index("t")
    print("fold of F47_137:", T0.fold.get("F47_137"), "| n per fold", T0.fold.value_counts().sort_index().to_dict())
    modes = [("ls",), ("lofo_med",), ("nested_med",), ("huber",), ("trim",), ("theil",), ("const", .3), ("const", .5), ("const", .7), ("const", 1.0)]
    for md in modes:
        for ex in ((), ("F47_137",)):
            cells, pw, per, ws, T = evalw(A, md, ex)
            print("%-12s excl%-8s cells %s | P %.3f | impr %d/%d | w %s" % (str(md), ",".join(ex) or "-", " ".join("%+.1f%%" % (100*x) for x in cells), pw, (per < 0).sum(), len(per), ws[:8]))
    cells, pw, per, ws, T = evalw(A, ("lofo_med",))
    tr = {t: truth.get(t) == p for t, p in A.items()}
    print("SSE delta total %.3f ; wrong-link recs sum %.3f (n %d)" % (per.sum(), per[[r for r in per.index if not tr[r]]].sum(), sum(not tr[r] for r in per.index)))
    df = pd.DataFrame(dict(d=per, ok=[tr[r] for r in per.index], p=[A[r] for r in per.index])).sort_values("d")
    print(df.head(6).to_string()); print(df.tail(4).to_string())
