"""IM05t: 후보 D1c(6판 중 넓이 ≥10%만 추가) — IM_부분생성_D1c_v1.md. 시험대 = GPT v1+v2 부분생성 125장(학습 제외). α=.0167."""
import sys, re, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05s_eval_pgen6 as S  # noqa (TYPE에 partial_gen_v6 포함)
P, V3, D, bb, lookfeats = S.P, S.V3, S.D, S.bb, S.lookfeats
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000; ALPHA = .025 * 2 / 3; SETS = S.SETS + ["im_set_gpt2"]

def main():
    rows, X, isred = D.load(SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); sets = np.array([r["_set"] for r in R]); realm = kind == "real"
    test = np.isin(sets, ["im_set_gpt1", "im_set_gpt2"]); v6 = sets == "im_set_v6"
    area = np.array([float(re.search(r"area=([\d.]+)", r["info"]).group(1)) if s == "im_set_v6" else 0 for r, s in zip(R, sets)])
    fz = {}
    for sn in SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    pz = bb("public21", "B2"); Pb = np.stack([pz[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    pg = test & (t == "partial_gen"); ps = test & (t == "partial_style"); ev = ~test & ~v6
    masks = {"A": ~test & ~v6, "D1c": ~test & (~v6 | (area >= .10))}
    print("행", len(R), "시험 부분생성", pg.sum(), "부분스타일", ps.sum(), "6판 사용", (v6 & (area >= .10)).sum(), flush=True)
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
            r = dict(auc=auc(pg), bal=V3.bal_acc(post[ev], y[ev], t[ev]), fp=(post[realm] >= .5).mean()); rr.append(r)
            print(f"  [{arm}] s{s} AUC 부분생성 {r['auc']:.4f} (v1 {auc(pg & (sets == 'im_set_gpt1')):.3f} v2 {auc(pg & (sets == 'im_set_gpt2')):.3f}) 부분스타일 {auc(ps):.4f}"
                  f" | 탐지(L2후) 부분생성 {(post[pg] >= .5).mean():.3f} 부분스타일 {(post[ps] >= .5).mean():.3f} | 실제오탐 {r['fp']:.4f} 균형 {r['bal']:.4f}"
                  f" | 유형 " + " ".join(f"{x[:6]}={(post[ev & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6)
                  + f" | 공개 {int(((pp >= .5) == py).sum())}/21 logloss {log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f} 공개부분생성 " + " ".join(f"{f[:4]}={q:.2f}" for f, q, tt in zip(pf, pp, pt) if tt == "partial_gen"), flush=True)
        res[arm] = (np.array(pre), rr)
    A, Ar = res["A"]; G, Gr = res["D1c"]; gi = np.where(pg)[0]; ri = np.where(realm)[0]; fh = grp[gi]; farms = np.array(sorted(set(fh)))
    def aucd(sel):
        yy = np.r_[np.zeros(len(ri)), np.ones(len(sel))]
        return roc_auc_score(yy, np.r_[G.mean(0)[ri], G.mean(0)[sel]]) - roc_auc_score(yy, np.r_[A.mean(0)[ri], A.mean(0)[sel]])
    rng = np.random.default_rng(7); idx = {f: gi[fh == f] for f in farms}
    bs = [aucd(np.concatenate([idx[f] for f in rng.choice(farms, len(farms))])) for _ in range(NB)]; pw = float((np.array(bs) <= 0).mean())
    c = [all(g["auc"] > a["auc"] for g, a in zip(Gr, Ar)), all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar)), all(g["fp"] <= a["fp"] + .01 for g, a in zip(Gr, Ar)), pw < ALPHA]
    print(f"\n== 판정(IM_부분생성_D1c_v1.md) ==\n  D1c: AUC↑ {c[0]} · 균형 {c[1]} · 오탐 {c[2]} · p_worse {pw:.4f} (<{ALPHA:.4f} {c[3]}) | AUC 차 {aucd(gi):+.4f} → {'후보' if all(c) else '불채택'}")

if __name__ == "__main__":
    main()
