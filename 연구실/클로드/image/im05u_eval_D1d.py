"""IM05u: 후보 D1d(D1c + OOF 오탐 맞춤 임계값) — IM_부분생성_D1d_v1.md. α=.0125."""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05t_eval_D1c as T  # noqa
S, P, V3, D, bb, lookfeats = T.S, T.P, T.V3, T.D, T.bb, T.lookfeats
SEEDS = (0, 1, 2); NB = 2000; ALPHA = .025 * 2 / 4

def bal(p, y, t, thr):
    r = (p[y == 0] < thr).mean(); rec = [(p[t == x] >= thr).mean() for x in V3.TYPES6 if (t == x).any()]; return .5 * r + .5 * np.mean(rec)

def main():
    rows, X, isred = D.load(T.SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"
    test = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v6 = sets == "im_set_v6"
    area = np.array([float(re.search(r"area=([\d.]+)", r["info"]).group(1)) if s == "im_set_v6" else 0 for r, s in zip(R, sets)])
    fz = {}
    for sn in T.SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    pg = test & (t == "partial_gen"); ps = test & (t == "partial_style"); ev = ~test & ~v6
    out = {}
    for arm, m in {"A": ~test & ~v6, "D1c": ~test & (~v6 | (area >= .10))}.items():
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); P_ = []
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            P_.append(P.rule(o, Lk))
        out[arm] = np.array(P_)
    A, G = out["A"], out["D1c"]
    fpA = np.mean([(a[realm] >= .5).mean() for a in A])
    taus = [np.quantile(g[realm], 1 - fpA) for g in G]; tau = float(np.mean(taus))
    print(f"A 시드평균 실제오탐 {fpA:.4f} → τ(시드별) {[round(x, 4) for x in taus]} 평균 {tau:.4f}", flush=True)
    rows_ = []
    for i, s in enumerate(SEEDS):
        a, g = A[i], G[i]
        r = dict(balA=bal(a[ev], y[ev], t[ev], .5), balG=bal(g[ev], y[ev], t[ev], tau), pgA=(a[pg] >= .5).mean(), pgG=(g[pg] >= tau).mean(),
                 fpA=(a[realm] >= .5).mean(), fpG=(g[realm] >= tau).mean(), psA=(a[ps] >= .5).mean(), psG=(g[ps] >= tau).mean())
        rows_.append(r)
        print(f"  s{s} 균형 A {r['balA']:.4f} D1d {r['balG']:.4f} | GPT부분생성 탐지 A {r['pgA']:.3f} D1d {r['pgG']:.3f} | GPT부분스타일 A {r['psA']:.3f} D1d {r['psG']:.3f}"
              f" | 실제오탐 A {r['fpA']:.4f} D1d {r['fpG']:.4f} | 유형(D1d) " + " ".join(f"{x[:6]}={(g[ev & (t == x)] >= tau).mean():.2f}" for x in V3.TYPES6), flush=True)
    Am, Gm = A.mean(0), G.mean(0); gi = np.where(pg)[0]; ri = np.where(realm)[0]
    farms = np.array(sorted(set(grp[gi]) | set(grp[ri]))); fi = {f: (gi[grp[gi] == f], ri[grp[ri] == f]) for f in farms}; rng = np.random.default_rng(7); bs = []
    for _ in range(NB):
        pick = rng.choice(farms, len(farms)); g_ = np.concatenate([fi[f][0] for f in pick]); r_ = np.concatenate([fi[f][1] for f in pick])
        if len(g_) == 0 or len(r_) == 0: continue
        bs.append(((Gm[g_] >= tau).mean() - (Gm[r_] >= tau).mean()) - ((Am[g_] >= .5).mean() - (Am[r_] >= .5).mean()))
    pw = float((np.array(bs) <= 0).mean())
    c = [all(r["balG"] > r["balA"] for r in rows_), all(r["pgG"] > r["pgA"] for r in rows_), all(r["fpG"] <= r["fpA"] + .005 for r in rows_), pw < ALPHA]
    print(f"\n== 판정(IM_부분생성_D1d_v1.md) ==\n  D1d: 균형↑ {c[0]} · 부분생성↑ {c[1]} · 오탐 {c[2]} · p_worse {pw:.4f} (<{ALPHA:.4f} {c[3]}) → {'후보' if all(c) else '불채택'}  (τ={tau:.4f})")

if __name__ == "__main__":
    main()
