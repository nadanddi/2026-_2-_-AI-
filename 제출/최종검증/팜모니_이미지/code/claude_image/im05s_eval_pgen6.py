"""IM05s: 후보 D1(부분생성 6판) — IM_부분생성_D1_v1.md. A / D1a(6판 추가) / D1b(v5 부분생성 → 6판 대체). GPT는 학습 제외.
주 지표 = GPT 부분생성 대 실제 AUC(OOF, L2 전).
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05p_eval_gpt as P  # noqa
V3, D, bb, lookfeats = P.V3, P.D, P.bb, P.lookfeats
V3.TYPE["partial_gen_v6"] = "partial_gen"
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; SETS = P.SETS + ["im_set_v6"]

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); gpt = sets == "im_set_gpt1"; v6 = sets == "im_set_v6"; realm = kind == "real"
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    pg_g = gpt & (t == "partial_gen"); ps_g = gpt & (t == "partial_style"); xlpg = kind == "partial_gen_xl"
    print("행", len(R), "6판", v6.sum(), "v5부분생성", xlpg.sum(), "GPT부분생성", pg_g.sum(), flush=True)
    masks = {"A": ~gpt & ~v6, "D1a": ~gpt, "D1b": ~gpt & ~xlpg}; ev = ~gpt & ~v6   # 균형·오탐은 공통 평가 집합(A의 자료)에서
    res = {}
    for arm, m in masks.items():
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); pre, rr = [], []
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            post = P.rule(o, Lk); pre.append(o)
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            pp = P.rule((D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            auc = lambda sel: roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(sel.sum())], np.r_[o[realm], o[sel]])
            r = dict(auc=auc(pg_g), aucps=auc(ps_g), bal=V3.bal_acc(post[ev], y[ev], t[ev]), fp=(post[realm] >= .5).mean())
            rr.append(r)
            print(f"  [{arm}] s{s} AUC GPT부분생성 {r['auc']:.4f} 부분스타일 {r['aucps']:.4f} | 탐지(L2후) GPT부분생성 {(post[pg_g] >= .5).mean():.2f} v5부분생성 {(post[xlpg] >= .5).mean():.3f} 6판 {(post[v6] >= .5).mean():.3f}"
                  f" | 실제오탐 {r['fp']:.4f} 균형 {r['bal']:.4f} | 유형 " + " ".join(f"{x[:6]}={(post[ev & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6)
                  + f" | 공개 {int(((pp >= .5) == py).sum())}/21 logloss {log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f} 공개부분생성 " + " ".join(f"{f[:4]}={q:.2f}" for f, q, tt in zip(pf, pp, pt) if tt == "partial_gen"), flush=True)
        res[arm] = (np.array(pre), rr)
    print("\n== 판정(IM_부분생성_D1_v1.md) ==")
    A, Ar = res["A"]; ok = {}; fh = grp[pg_g]; farms = np.array(sorted(set(fh))); gi = np.where(pg_g)[0]; ri = np.where(realm)[0]
    def aucd(G, sel):
        yy = np.r_[np.zeros(len(ri)), np.ones(len(sel))]
        return roc_auc_score(yy, np.r_[G.mean(0)[ri], G.mean(0)[sel]]) - roc_auc_score(yy, np.r_[A.mean(0)[ri], A.mean(0)[sel]])
    for arm in ("D1a", "D1b"):
        G, Gr = res[arm]
        c = [all(g["auc"] > a["auc"] for g, a in zip(Gr, Ar)), all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar)), all(g["fp"] <= a["fp"] + .01 for g, a in zip(Gr, Ar))]
        rng = np.random.default_rng(7); idx = {f: gi[fh == f] for f in farms}
        bs = [aucd(G, np.concatenate([idx[f] for f in rng.choice(farms, len(farms))])) for _ in range(NB)]; pw = float((np.array(bs) <= 0).mean())
        c.append(pw < .025); ok[arm] = all(c)
        print(f"  {arm}: AUC↑ {c[0]} · 균형 {c[1]} · 오탐 {c[2]} · p_worse {pw:.4f} ({c[3]}) | AUC 차(시드평균 예측) {aucd(G, gi):+.4f} → {'후보' if ok[arm] else '불채택'}")
    if all(ok.values()): print("  둘 다 후보 → AUC 높은 쪽:", max(ok, key=lambda a: np.mean([r["auc"] for r in res[a][1]])))

if __name__ == "__main__":
    main()
