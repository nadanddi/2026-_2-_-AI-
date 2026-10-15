"""IM05ac: L2 규칙 개선 v2 — IM_L2규칙개선_v2.md. 모델 OOF를 한 번 계산해 저장(local/preds/L2rule_O.npz)한 뒤 규칙별 판정."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05ab_eval_L2rule as AB  # noqa (rule, 모델 설정 공유)
Z, P, V3, D, bb, lookfeats = AB.Z, AB.P, AB.V3, AB.D, AB.bb, AB.lookfeats
LOCAL = D.LOCAL; SETS = AB.SETS; SEEDS = AB.SEEDS; NB = 2000; ALPHA = AB.ALPHA; rule = AB.rule

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; EXk = EX[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"
    test = np.isin(sets, ["im_set_gpt3", "im_set_gpt3r", "im_set_realn", "im_set_realg"]); ev = ~gtr & ~test
    realE = ev & (kind == "real"); fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    cache = LOCAL / "preds" / "L2rule_O.npz"
    if cache.exists():
        zc = np.load(cache); O = {a: [(zc[f"{a}_o{i}"], zc[f"{a}_f{i}"], zc[f"{a}_p{i}"]) for i in range(3)] for a in ("A", "G3c1")}
    else:
        O = {}; sv = {}
        for arm in ("A", "G3c1"):
            m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m])
            if arm == "G3c1":
                for x in P.GT:
                    g, o_ = gtr & (t == x), ev & (t == x); tot = w[g].sum() + w[o_].sum()
                    if w[o_].sum() == 0 or g.sum() == 0: continue
                    w[g] = tot * .10 / g.sum(); w[o_] *= tot * .90 / w[o_].sum()
            O[arm] = []
            for i, s in enumerate(SEEDS):
                o = np.zeros(N); fold = np.zeros(N, int)
                for f, (tr, te) in enumerate(V3.folds(grp, s)):
                    tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2; fold[te] = f
                rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
                po = (D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2
                O[arm].append((o, fold, po)); sv.update({f"{arm}_o{i}": o, f"{arm}_f{i}": fold, f"{arm}_p{i}": po}); print(f"  학습 완료 {arm} s{s}", flush=True)
        np.savez(cache, **sv)
    def balh(h, realsel):
        return .5 * (1 - h[realsel].mean()) + .5 * np.mean([h[ev & (t == x)].mean() for x in V3.TYPES6 if (ev & (t == x)).any()])
    res = {}
    for k in ("R0", "R1", "R2", "R3"):
        for arm in ("A", "G3c1"):
            rr = []
            for i, s in enumerate(SEEDS):
                o, fold, po = O[arm][i]; p, hit = rule(o, Lk, k)
                if arm == "A": th = np.full(N, .5); tp = .5
                else:
                    pA0 = rule(O["A"][i][0], Lk, "R0")[0]
                    ths = [float(np.quantile(p[realE & (fold != f)], 1 - (pA0[realE & (fold != f)] >= .5).mean())) for f in range(5)]; th = np.array(ths)[fold]; tp = float(np.mean(ths))
                h = p >= th; pp, _ = rule(po, PL, k)
                r = dict(fp=h[realE].mean(), bal=balh(h, realE), balK=balh(h, realE & ~EXk), det=h[v3].mean(), pub=int(((pp >= tp) == py).sum()), h=h,
                         hrE=int((hit & realE & EXk).sum()), hrK=int((hit & realE & ~EXk).sum()), hf=int((hit & ev & (y == 1)).sum()))
                rr.append(r)
                print(f"  [{k}/{arm}] s{s} 진짜오탐 {r['fp']:.4f} 균형 표준 {r['bal']:.4f} 원본있음 {r['balK']:.4f} v3 {r['det']:.3f} | 규칙발동 진짜(원본뺌) {r['hrE']} 진짜(원본있음) {r['hrK']} 가짜 {r['hf']}"
                      f" | 공개 {r['pub']}/21 | 공개규칙대상 " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, pp) if f[:4] in ("uexm", "ttmn", "xeeo", "bpem", "yuag")), flush=True)
            res[(k, arm)] = rr
    print("\n== 판정(IM_L2규칙개선_v2.md) ==")
    use = np.where(ev)[0]; farms = np.array(sorted(set(grp[use]))); fi = {f: use[grp[use] == f] for f in farms}; ok = {}
    def balw(h, idx):
        wt = np.bincount(idx, minlength=N).astype(float); rate = lambda sel: (h[sel] * wt[sel]).sum() / max(wt[sel].sum(), 1e-9)
        return .5 * (1 - rate(realE)) + .5 * np.mean([rate(ev & (t == x)) for x in V3.TYPES6])
    for k in ("R1", "R2", "R3"):
        A0, Ak, G0, Gk = res[("R0", "A")], res[(k, "A")], res[("R0", "G3c1")], res[(k, "G3c1")]
        c = [all(b["fp"] < a["fp"] and b["bal"] > a["bal"] for a, b in zip(A0, Ak)), all(b["balK"] >= a["balK"] - .001 for a, b in zip(A0, Ak)),
             all(b["bal"] > a["bal"] and b["det"] >= a["det"] - .02 for a, b in zip(G0, Gk)), all(b["pub"] >= a["pub"] for a, b in zip(A0, Ak))]
        Hk = (np.mean([r["h"] for r in Ak], 0) >= .5).astype(float); H0 = (np.mean([r["h"] for r in A0], 0) >= .5).astype(float); rng = np.random.default_rng(7)
        bs = [balw(Hk, idx) - balw(H0, idx) for idx in (np.concatenate([fi[f] for f in rng.choice(farms, len(farms))]) for _ in range(NB))]
        pw = float((np.array(bs) <= 0).mean()); c.append(pw < ALPHA); ok[k] = all(c)
        lost = int(np.mean([a["hf"] - b["hf"] for a, b in zip(A0, Ak)]))
        print(f"  {k}: A표준 {c[0]} · A원본있음 {c[1]} · G3c1 {c[2]} · 공개 {c[3]} · p_worse {pw:.4f} ({c[4]}) → {'후보' if ok[k] else '불채택'} | 잃은 가짜 발동(평균) {lost}")
    cands = [k for k in ok if ok[k]]
    if len(cands) > 1: print("  후보 여럿 → A 표준 균형 최고:", max(cands, key=lambda k: np.mean([r["bal"] for r in res[(k, "A")]])))

if __name__ == "__main__":
    main()
