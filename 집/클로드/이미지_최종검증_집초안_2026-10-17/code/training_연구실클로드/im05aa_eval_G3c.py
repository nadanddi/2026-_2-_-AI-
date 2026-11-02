"""IM05aa: 후보 G3c — IM_GPT학습_G3c_v1.md. A(.5) / G3c1(GPT 몫 10% + τ) / G3c2(GPT 기존 가중 + τ). 시드 3/4/5, α=.004.
τ: 폴드 f는 다른 폴드 OOF 진짜(L2 후)에서 A의 .5 오탐률과 같아지는 값(교차 적합)."""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05z_eval_shortcut as Z  # noqa (TYPE·SETS: G3 + v3R·realN·realG)
P, V3, D, bb, lookfeats = Z.P, Z.V3, Z.D, Z.bb, Z.lookfeats
LOCAL = D.LOCAL; SEEDS = (3, 4, 5); NB = 2000; ALPHA = .004; SETS = Z.SETS

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; rn = sets == "im_set_realn"
    test = np.isin(sets, ["im_set_gpt3", "im_set_gpt3r", "im_set_realn", "im_set_realg"]); ev = ~gtr & ~test
    meth = np.array([r["info"].split(";")[1].split("_")[2] if r["_set"] == "im_set_gpt3" else "" for r in R])
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    yy = np.r_[np.zeros(realm.sum()), np.ones(v3.sum())]; auc = lambda q: roc_auc_score(yy, np.r_[q[realm], q[v3]])
    def bal(p, thr): r = (p[ev & (y == 0)] < thr).mean(); return .5 * r + .5 * np.mean([(p[ev & (t == x)] >= thr).mean() for x in V3.TYPES6 if (ev & (t == x)).any()])
    out = {}
    for arm in ("A", "G3c1", "G3c2"):
        m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m])
        if arm == "G3c1":
            for x in P.GT:
                g, o_ = gtr & (t == x), ev & (t == x); tot = w[g].sum() + w[o_].sum()
                if w[o_].sum() == 0 or g.sum() == 0: continue
                w[g] = tot * .10 / g.sum(); w[o_] *= tot * .90 / w[o_].sum()
        out[arm] = []
        for s in SEEDS:
            c_ = np.zeros(N); l_ = np.zeros(N); fold = np.zeros(N, int)
            for f, (tr, te) in enumerate(V3.folds(grp, s)):
                tr = tr[m[tr]]; c_[te] = D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1]; l_[te] = D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]; fold[te] = f
            o = (c_ + l_) / 2; p = P.rule(o, Lk)
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            pp = P.rule((D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            out[arm].append(dict(o=o, c=c_, p=p, fold=fold, pp=pp))
    rows_out = {a: [] for a in out}
    for i, s in enumerate(SEEDS):
        A = out["A"][i]; aP, fold = A["p"], A["fold"]
        for arm in out:
            d = out[arm][i]; p = d["p"]
            if arm == "A": th = np.full(N, .5); tp = .5; ths = [.5] * 5
            else:
                ths = []
                for f in range(5):
                    oth = realm & (fold != f); target = (aP[oth] >= .5).mean(); ths.append(float(np.quantile(p[oth], 1 - target)))
                th = np.array(ths)[d["fold"]]; tp = float(np.mean(ths))
            hit = p >= th; r = dict(auc=auc(d["o"]), aucc=auc(d["c"]), det=hit[v3].mean(), fp=hit[realm].mean(), fn=hit[rn].mean(),
                                    bal=.5 * (1 - hit[ev & (y == 0)].mean()) + .5 * np.mean([hit[ev & (t == x)].mean() for x in V3.TYPES6 if (ev & (t == x)).any()]),
                                    pub=int(((d["pp"] >= tp) == py).sum()), pll=log_loss(py, np.clip(d["pp"], 1e-4, 1 - 1e-4)))
            rows_out[arm].append(r)
            print(f"  [{arm}] s{s} τ {[round(x, 3) for x in ths]} | AUC {r['auc']:.4f} CLIP {r['aucc']:.4f} | v3탐지 {r['det']:.3f} (1단계 {hit[v3 & (kind == 'gpt3_s1')].mean():.2f} A최종 {hit[v3 & (meth == 'A') & (kind == 'gpt3_pgen')].mean():.2f}"
                  f" B최종 {hit[v3 & (meth == 'B')].mean():.2f} 부분스타일 {hit[v3 & (kind == 'gpt3_pstyle')].mean():.2f}) | 진짜오탐 OOF {r['fp']:.4f} realN {r['fn']:.3f} | 균형 {r['bal']:.4f}"
                  f" | 유형 " + " ".join(f"{x[:6]}={hit[ev & (t == x)].mean():.2f}" for x in V3.TYPES6) + f" | 공개 {r['pub']}/21 logloss {r['pll']:.3f}", flush=True)
    Am = np.mean([d["o"] for d in out["A"]], 0); vi = np.where(v3)[0]; ri = np.where(realm)[0]
    farms = np.array(sorted(set(grp[vi]) | set(grp[ri]))); fi = {f: (vi[grp[vi] == f], ri[grp[ri] == f]) for f in farms}
    print("\n== 판정(IM_GPT학습_G3c_v1.md) =="); ok = {}; Ar = rows_out["A"]
    for arm in ("G3c1", "G3c2"):
        Gm = np.mean([d["o"] for d in out[arm]], 0); Gr = rows_out[arm]; rng = np.random.default_rng(7); bs = []
        for _ in range(NB):
            pick = rng.choice(farms, len(farms)); a_ = np.concatenate([fi[f][0] for f in pick]); b_ = np.concatenate([fi[f][1] for f in pick])
            if len(a_) == 0 or len(b_) == 0: continue
            q = np.r_[np.zeros(len(b_)), np.ones(len(a_))]; bs.append(roc_auc_score(q, np.r_[Gm[b_], Gm[a_]]) - roc_auc_score(q, np.r_[Am[b_], Am[a_]]))
        pw = float((np.array(bs) <= 0).mean())
        c = [all(g["auc"] > a["auc"] and g["aucc"] > a["aucc"] for g, a in zip(Gr, Ar)), all(g["det"] >= a["det"] + .05 for g, a in zip(Gr, Ar)),
             all(g["fp"] <= a["fp"] + .005 for g, a in zip(Gr, Ar)), all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar)), pw < ALPHA]; ok[arm] = all(c)
        print(f"  {arm}: AUC·CLIP↑ {c[0]} · 탐지+.05 {c[1]} · 오탐 {c[2]} · 균형 {c[3]} · p_worse {pw:.4f} ({c[4]}) → {'후보(GPT 계열 한정)' if all(c) else '불채택'}")
    if all(ok.values()): print("  둘 다 후보 → 균형 높은 쪽:", max(ok, key=lambda a: np.mean([r["bal"] for r in rows_out[a]])))

if __name__ == "__main__":
    main()
