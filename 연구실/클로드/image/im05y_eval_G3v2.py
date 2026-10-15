"""IM05y: 후보 G3 v2 — IM_GPT학습_G3_v2.md. 안·자료는 im05x(v1)와 같고 판정·진단만 바뀜. α=.0125."""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05x_eval_G3 as X  # noqa (TYPE 갱신·SETS)
P, V3, D, bb, lookfeats = X.P, X.V3, X.D, X.bb, X.lookfeats
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; ALPHA = .0125; SETS = X.SETS

def main():
    rows, Xd, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = Xd["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; ev = ~gtr & ~v3
    meth = np.array([r["info"].split(";")[1].split("_")[2] if r["_set"] == "im_set_gpt3" else "" for r in R])   # A/B/C (1단계는 A_stage1)
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    hit = (Lk[:, 0] > 3.5) & (Lk[:, 0] < 20) & (Lk[:, 2] > 8)
    print("행", N, "GPT 학습분", gtr.sum(), "v3", v3.sum(), "v3 농장 수", len(set(grp[v3])), "v3 L2 적중률", round(float(hit[v3].mean()), 3), "진짜 L2 적중률", round(float(hit[realm].mean()), 4), flush=True)
    yy = np.r_[np.zeros(realm.sum()), np.ones(v3.sum())]; auc = lambda q: roc_auc_score(yy, np.r_[q[realm], q[v3]])
    res = {}
    for arm in ("A", "G3a", "G3b"):
        m = ev.copy() if arm == "A" else (ev | gtr); w = np.zeros(N); w[m] = V3.weights(kind[m])
        if arm == "G3b":
            for x in P.GT:
                g, o_ = gtr & (t == x), ev & (t == x); tot = w[g].sum() + w[o_].sum()
                if w[o_].sum() == 0 or g.sum() == 0: continue
                w[g] = tot * P.SHARE / g.sum(); w[o_] *= tot * (1 - P.SHARE) / w[o_].sum()
        pre, rr = [], []
        for s in SEEDS:
            c_ = np.zeros(N); l_ = np.zeros(N)
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; c_[te] = D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1]; l_[te] = D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]
            o = (c_ + l_) / 2; p = P.rule(o, Lk); pre.append(o)
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            pp = P.rule((D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            r = dict(auc=auc(o), aucc=auc(c_), aucl=auc(l_), det=(p[v3] >= .5).mean(), det_pre=(o[v3] >= .5).mean(), fp=(p[realm] >= .5).mean(), bal=V3.bal_acc(p[ev], y[ev], t[ev]),
                     ll=log_loss(yy, np.clip(np.r_[p[realm], p[v3]], 1e-4, 1 - 1e-4)), pll=log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)), pub=int(((pp >= .5) == py).sum()))
            rr.append(r)
            print(f"  [{arm}] s{s} AUC 평균 {r['auc']:.4f} CLIP {r['aucc']:.4f} 저수준 {r['aucl']:.4f} | v3탐지 L2후 {r['det']:.3f} L2전 {r['det_pre']:.3f} (1단계 {(p[v3 & (meth == 'A') & (kind == 'gpt3_s1')] >= .5).mean():.2f}"
                  f" A최종 {(p[v3 & (meth == 'A') & (kind == 'gpt3_pgen')] >= .5).mean():.2f} B최종 {(p[v3 & (meth == 'B')] >= .5).mean():.2f} 부분스타일 {(p[v3 & (kind == 'gpt3_pstyle')] >= .5).mean():.2f})"
                  f" | 진짜오탐 {r['fp']:.4f} 균형 {r['bal']:.4f} | 진짜+v3 logloss {r['ll']:.3f} | 공개 {r['pub']}/21 logloss {r['pll']:.3f}", flush=True)
        res[arm] = (np.array(pre), rr)
    A, Ar = res["A"]; Am = A.mean(0); fpA = np.mean([r["fp"] for r in Ar])
    vi = np.where(v3)[0]; ri = np.where(realm)[0]; farms = np.array(sorted(set(grp[vi]) | set(grp[ri]))); fi = {f: (vi[grp[vi] == f], ri[grp[ri] == f]) for f in farms}
    print("\n== 판정(IM_GPT학습_G3_v2.md) =="); ok = {}
    for arm in ("G3a", "G3b"):
        G, Gr = res[arm]; Gm = G.mean(0); rng = np.random.default_rng(7); bs = []
        for _ in range(NB):
            pick = rng.choice(farms, len(farms)); a_ = np.concatenate([fi[f][0] for f in pick]); b_ = np.concatenate([fi[f][1] for f in pick])
            if len(a_) == 0 or len(b_) == 0: continue
            q = np.r_[np.zeros(len(b_)), np.ones(len(a_))]
            bs.append(roc_auc_score(q, np.r_[Gm[b_], Gm[a_]]) - roc_auc_score(q, np.r_[Am[b_], Am[a_]]))
        pw = float((np.array(bs) <= 0).mean())
        c = [all(g["auc"] > a["auc"] for g, a in zip(Gr, Ar)), all(g["aucc"] > a["aucc"] for g, a in zip(Gr, Ar)), all(g["det"] >= a["det"] + .05 for g, a in zip(Gr, Ar)),
             all(g["fp"] <= a["fp"] + .01 for g, a in zip(Gr, Ar)), all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar)), pw < ALPHA]; ok[arm] = all(c)
        thr = np.quantile(Gm[realm], 1 - fpA)
        print(f"  {arm}: AUC↑ {c[0]} · CLIP AUC↑ {c[1]} · 탐지+.05 {c[2]} · 오탐 {c[3]} · 균형 {c[4]} · p_worse {pw:.4f} ({c[5]}) → {'후보(GPT 계열 한정)' if all(c) else '불채택'}"
              f" | 진단: 오탐 맞춤({fpA:.3f}) v3 탐지 A {(Am[v3] >= np.quantile(Am[realm], 1 - fpA)).mean():.3f} {arm} {(Gm[v3] >= thr).mean():.3f}")
    if all(ok.values()): print("  둘 다 후보 → AUC 높은 쪽:", max(ok, key=lambda a: np.mean([r["auc"] for r in res[a][1]])))

if __name__ == "__main__":
    main()
