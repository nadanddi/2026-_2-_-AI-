"""IM05x: 후보 G3 — IM_GPT학습_G3_v1.md. A(GPT 학습 안 함) / G3a(GPT v1·v2 250 추가) / G3b(GPT 몫 25%). 시험대 GPT v3(im_set_gpt3)는 학습에 안 씀. α=.0125."""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05p_eval_gpt as P  # noqa (TYPE·rule·GT·SHARE)
V3, D, bb, lookfeats = P.V3, P.D, P.bb, P.lookfeats
V3.TYPE.update({"gpt3_pgen": "partial_gen", "gpt3_pstyle": "partial_style", "gpt3_s1": "full_style"})
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; ALPHA = .0125
SETS = ["im_set_v4", "im_set_sd2", "im_set_v5", "im_set_gpt1", "im_set_gpt2", "im_set_gpt3"]

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"; N = len(R)
    gtr = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v3 = sets == "im_set_gpt3"; ev = ~gtr & ~v3
    edit = np.array([r["info"].split(";")[2] if r["_set"] == "im_set_gpt3" else "" for r in R])
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    print("행", N, "GPT 학습분", gtr.sum(), "v3 시험", v3.sum(), {k: int((v3 & (kind == k)).sum()) for k in ("gpt3_pgen", "gpt3_pstyle", "gpt3_s1")}, flush=True)
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
            o = np.zeros(N)
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            p = P.rule(o, Lk); pre.append(p)
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            pp = P.rule((D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            r = dict(det=(p[v3] >= .5).mean(), fp=(p[realm] >= .5).mean(), bal=V3.bal_acc(p[ev], y[ev], t[ev])); rr.append(r)
            auc = roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(v3.sum())], np.r_[o[realm], o[v3]])
            print(f"  [{arm}] s{s} v3 탐지 {r['det']:.3f} (부분생성 {(p[v3 & (kind == 'gpt3_pgen')] >= .5).mean():.2f} 부분스타일 {(p[v3 & (kind == 'gpt3_pstyle')] >= .5).mean():.2f}"
                  f" 1단계 {(p[v3 & (kind == 'gpt3_s1')] >= .5).mean():.2f}) AUC {auc:.3f} | 진짜오탐 {r['fp']:.4f} | GPT외 균형 {r['bal']:.4f} | 유형 "
                  + " ".join(f"{x[:6]}={(p[ev & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6)
                  + f" | 공개 {int(((pp >= .5) == py).sum())}/21 logloss {log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f}", flush=True)
        res[arm] = (np.array(pre), rr)
        es = sorted({e for e in edit[v3] if e})
        if es: print("     편집종류별 v3 탐지(시드평균): " + " ".join(f"{e[:20]}={(np.mean(pre, 0)[v3 & (edit == e)] >= .5).mean():.2f}(n={(v3 & (edit == e)).sum()})" for e in es), flush=True)
    A, Ar = res["A"]; Am = A.mean(0); use = np.where(v3 | realm)[0]; farms = np.array(sorted(set(grp[use]))); fi = {f: use[grp[use] == f] for f in farms}
    def gap(p, idx):
        wt = np.bincount(idx, minlength=N).astype(float); h = (p >= .5).astype(float)
        return (h[v3] * wt[v3]).sum() / max(wt[v3].sum(), 1e-9) - (h[realm] * wt[realm]).sum() / max(wt[realm].sum(), 1e-9)
    print("\n== 판정(IM_GPT학습_G3_v1.md) =="); ok = {}
    for arm in ("G3a", "G3b"):
        G, Gr = res[arm]; Gm = G.mean(0); rng = np.random.default_rng(7); bs = []
        for _ in range(NB):
            idx = np.concatenate([fi[f] for f in rng.choice(farms, len(farms))])
            if not (v3[idx].any() and realm[idx].any()): continue
            bs.append(gap(Gm, idx) - gap(Am, idx))
        pw = float((np.array(bs) <= 0).mean())
        c = [all(g["det"] >= a["det"] + .05 for g, a in zip(Gr, Ar)), all(g["fp"] <= a["fp"] + .01 for g, a in zip(Gr, Ar)),
             all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar)), pw < ALPHA]; ok[arm] = all(c)
        print(f"  {arm}: v3탐지≥+.05 {c[0]} · 오탐 {c[1]} · 균형 {c[2]} · p_worse {pw:.4f} ({c[3]}) → {'후보' if all(c) else '불채택'}")
    if all(ok.values()): print("  둘 다 후보 → v3 탐지 높은 쪽:", max(ok, key=lambda a: np.mean([r["det"] for r in res[a][1]])))

if __name__ == "__main__":
    main()
