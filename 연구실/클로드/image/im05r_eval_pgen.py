"""IM05r: 후보 P(부분생성 개선, 조각 단위 판단) — IM_부분생성_P_v1.md.
A = 조각 평균, P1 = 평균‖최대, P2 = 조각 MIL(상위 2 평균). GPT 100장은 학습 제외, OOF만.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05p_eval_gpt as P  # noqa (TYPE 갱신, rule, SETS)
V3, D, lookfeats = P.V3, P.D, P.lookfeats
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); NB = 2000

def crops(name):
    z = np.load(LOCAL / "feats" / f"{name}_B2.npz"); F = z["feats"].astype(np.float32)[:, 1:]
    F /= np.linalg.norm(F, axis=-1, keepdims=True) + 1e-8; return dict(zip(z["files"], F))

def main():
    rows, X, isred = D.load(P.SETS); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk = L[keep]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); gpt = np.array([r["_set"] == "im_set_gpt1" for r in R]); realm = kind == "real"
    fz = {}
    for sn in P.SETS: fz.update({(sn, k): v for k, v in crops(sn).items()})
    C = np.stack([fz[(r["_set"], r["file"])] for r in R]); N, V, Dm = C.shape
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pc = crops("public21"); PC = np.stack([pc[f] for f in pf])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    feat = {"A": lambda c: c.mean(1), "P1": lambda c: np.concatenate([c.mean(1), c.max(1)], 1)}
    m = ~gpt; w = np.zeros(N); w[m] = V3.weights(kind[m])
    pg_o = m & (t == "partial_gen"); pg_g = gpt & (t == "partial_gen"); ps_o = m & (t == "partial_style"); ps_g = gpt & (t == "partial_style")
    print("행", N, "조각", V, "부분생성 우리", pg_o.sum(), "GPT", pg_g.sum(), flush=True)

    def clip_pred(arm, tr, Xte_c, s):
        if arm in feat:
            Xa = feat[arm](C); return D.fit("clip", Xa, y, w, tr, s).predict_proba(feat[arm](Xte_c))[:, 1]
        Xc = C[tr].reshape(-1, Dm); yc = np.repeat(y[tr], V); wc = np.repeat(w[tr] / V, V)
        mm = D.mdl("clip", s).fit(Xc, yc, logisticregression__sample_weight=wc)
        pp = mm.predict_proba(Xte_c.reshape(-1, Dm))[:, 1].reshape(len(Xte_c), V)
        return np.sort(pp, 1)[:, -2:].mean(1)

    res = {}
    for arm in ("A", "P1", "P2"):
        pre, rr = [], []
        for s in SEEDS:
            c_ = np.zeros(N); l_ = np.zeros(N)
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; c_[te] = clip_pred(arm, tr, C[te], s); l_[te] = D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]
            o = (c_ + l_) / 2; post = P.rule(o, Lk); pre.append(o)
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            pp = P.rule((clip_pred(arm, bag, PC, s) + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2, PL)
            r = dict(pgo=o[pg_o].mean(), pgg=o[pg_g].mean(), fp=(post[realm] >= .5).mean(), bal=V3.bal_acc(post[m], y[m], t[m]))
            rr.append(r)
            print(f"  [{arm}] s{s} 부분생성 평균p 우리 {r['pgo']:.4f} GPT {r['pgg']:.4f} | 탐지(L2후) 우리 {(post[pg_o] >= .5).mean():.3f} GPT {(post[pg_g] >= .5).mean():.2f}"
                  f" | 부분스타일 우리 {(post[ps_o] >= .5).mean():.3f} GPT {(post[ps_g] >= .5).mean():.2f} | 실제오탐 {r['fp']:.4f} 균형 {r['bal']:.4f}"
                  f" | CLIP단독 GPT부분생성 {(c_[pg_g] >= .5).mean():.2f} | 유형 " + " ".join(f"{x[:6]}={(post[m & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6)
                  + f" | 공개 {int(((pp >= .5) == py).sum())}/21 logloss {log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f}", flush=True)
        res[arm] = (np.array(pre), rr)
    print("\n== 판정(IM_부분생성_P_v1.md) ==")
    A, Ar = res["A"]; pg = pg_o | pg_g; ok = {}
    for arm in ("P1", "P2"):
        G, Gr = res[arm]
        c = [all(g["pgo"] > a["pgo"] for g, a in zip(Gr, Ar)), all(g["pgg"] > a["pgg"] for g, a in zip(Gr, Ar)),
             all(g["fp"] <= a["fp"] + .01 for g, a in zip(Gr, Ar)), all(g["bal"] >= a["bal"] - .003 for g, a in zip(Gr, Ar))]
        d = (G.mean(0) - A.mean(0))[pg]; fh = grp[pg]; farms = np.array(sorted(set(fh))); idx = {f: np.where(fh == f)[0] for f in farms}; rng = np.random.default_rng(7)
        bs = [np.concatenate([d[idx[f]] for f in rng.choice(farms, len(farms))]).mean() for _ in range(NB)]; pw = float((np.array(bs) <= 0).mean())
        c.append(pw < .025); ok[arm] = all(c)
        print(f"  {arm}: 우리↑ {c[0]} · GPT↑ {c[1]} · 오탐 {c[2]} · 균형 {c[3]} · p_worse {pw:.4f} ({c[4]}) | 평균p 차 {d.mean():+.4f} → {'후보' if ok[arm] else '불채택'}")
    if all(ok.values()): print("  둘 다 후보 → 균형 높은 쪽:", max(ok, key=lambda a: np.mean([r["bal"] for r in res[a][1]])))

if __name__ == "__main__":
    main()
