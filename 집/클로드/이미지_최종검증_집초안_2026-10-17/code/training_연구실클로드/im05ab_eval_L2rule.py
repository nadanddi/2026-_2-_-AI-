"""IM05ab: L2 규칙 개선 — IM_L2규칙개선_v1.md. 기준 모델 A / G3c1, 규칙 R0~R3, 시드 3/4/5, α=.0083."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05z_eval_shortcut as Z  # noqa
P, V3, D, bb, lookfeats = Z.P, Z.V3, Z.D, Z.bb, Z.lookfeats
LOCAL = D.LOCAL; SETS = Z.SETS; SEEDS = (3, 4, 5); NB = 2000; ALPHA = .025 / 3

def rule(o, L, k):
    d1, gap = L[:, 0], L[:, 2]; inr = (d1 > 3.5) & (d1 < 20)
    hit = {"R0": inr & (gap > 8), "R1": inr & (gap > 8) & (o >= .25), "R2": inr & (gap > 12), "R3": inr & (gap > 12) & (o >= .25)}[k]
    return np.where(hit, np.maximum(o, .9), o), hit

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"
    test = np.isin(sets, ["im_set_gpt3", "im_set_gpt3r", "im_set_realn", "im_set_realg"]); ev = ~gtr & ~test
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    def balh(h): return .5 * (1 - h[ev & (y == 0)].mean()) + .5 * np.mean([h[ev & (t == x)].mean() for x in V3.TYPES6 if (ev & (t == x)).any()])
    O = {}
    for arm in ("A", "G3c1"):
        m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m])
        if arm == "G3c1":
            for x in P.GT:
                g, o_ = gtr & (t == x), ev & (t == x); tot = w[g].sum() + w[o_].sum()
                if w[o_].sum() == 0 or g.sum() == 0: continue
                w[g] = tot * .10 / g.sum(); w[o_] *= tot * .90 / w[o_].sum()
        O[arm] = []
        for s in SEEDS:
            o = np.zeros(N); fold = np.zeros(N, int)
            for f, (tr, te) in enumerate(V3.folds(grp, s)):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2; fold[te] = f
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            po = (D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2
            O[arm].append((o, fold, po)); print(f"  학습 완료 {arm} s{s}", flush=True)
    res = {}
    for k in ("R0", "R1", "R2", "R3"):
        for arm in ("A", "G3c1"):
            rr = []
            for i, s in enumerate(SEEDS):
                o, fold, po = O[arm][i]; p, hit = rule(o, Lk, k); pA0 = rule(O["A"][i][0], Lk, "R0")[0]
                if arm == "A": th = np.full(N, .5); tp = .5
                else:
                    ths = [float(np.quantile(p[realm & (fold != f)], 1 - (pA0[realm & (fold != f)] >= .5).mean())) for f in range(5)]; th = np.array(ths)[fold]; tp = float(np.mean(ths))
                h = p >= th; pp, _ = rule(po, PL, k)
                r = dict(fp=h[realm].mean(), bal=balh(h), det=h[v3].mean(), pub=int(((pp >= tp) == py).sum()), hr=int((hit & realm).sum()), hf=int((hit & ev & (y == 1)).sum()), h=h)
                rr.append(r)
                print(f"  [{k}/{arm}] s{s} 진짜오탐 {r['fp']:.4f} 균형 {r['bal']:.4f} v3 {r['det']:.3f} | 규칙발동 진짜 {r['hr']} 가짜 {r['hf']} | 공개 {r['pub']}/21 | 유형 "
                      + " ".join(f"{x[:6]}={h[ev & (t == x)].mean():.2f}" for x in V3.TYPES6) + " | 공개규칙대상 " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, pp) if f[:4] in ("uexm", "ttmn", "xeeo", "bpem", "yuag")), flush=True)
            res[(k, arm)] = rr
    print("\n== 판정(IM_L2규칙개선_v1.md) ==")
    use = np.where(ev)[0]; farms = np.array(sorted(set(grp[use]))); fi = {f: use[grp[use] == f] for f in farms}; ok = {}
    def balw(h, idx):
        wt = np.bincount(idx, minlength=N).astype(float); rate = lambda sel: (h[sel] * wt[sel]).sum() / max(wt[sel].sum(), 1e-9)
        return .5 * (1 - rate(ev & (y == 0))) + .5 * np.mean([rate(ev & (t == x)) for x in V3.TYPES6])
    for k in ("R1", "R2", "R3"):
        c1 = all(all(g["fp"] < a["fp"] for g, a in zip(res[(k, arm)], res[("R0", arm)])) for arm in ("A", "G3c1"))
        c2 = all(all(g["bal"] > a["bal"] for g, a in zip(res[(k, arm)], res[("R0", arm)])) for arm in ("A", "G3c1"))
        c3 = all(g["det"] >= a["det"] - .02 for g, a in zip(res[(k, "G3c1")], res[("R0", "G3c1")]))
        Hk = np.mean([r["h"] for r in res[(k, "A")]], 0) >= .5; H0 = np.mean([r["h"] for r in res[("R0", "A")]], 0) >= .5
        rng = np.random.default_rng(7); bs = [balw(Hk.astype(float), idx) - balw(H0.astype(float), idx) for idx in (np.concatenate([fi[f] for f in rng.choice(farms, len(farms))]) for _ in range(NB))]
        pw = float((np.array(bs) <= 0).mean()); ok[k] = c1 and c2 and c3 and pw < ALPHA
        print(f"  {k}: 진짜오탐↓ {c1} · 균형↑ {c2} · v3유지 {c3} · p_worse {pw:.4f} ({pw < ALPHA}) → {'후보' if ok[k] else '불채택'}")
    cands = [k for k in ok if ok[k]]
    if len(cands) > 1: print("  후보 여럿 → A 균형 최고:", max(cands, key=lambda k: np.mean([r["bal"] for r in res[(k, "A")]])))

if __name__ == "__main__":
    main()
