"""IM05l: 후보 L2(좁은 원본 대조 규칙) 판정 — IM_후보L1_v1.md 'L2 등록'. stack 확률에 규칙만 덧씌움."""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05d_eval_v4 as D, im05c_eval_v3 as V3  # noqa
from im05i_eval_L1 import lookfeats  # noqa
LOCAL = D.LOCAL
def rule(p, L): hit = (L[:, 0] > 3.5) & (L[:, 0] < 20) & (L[:, 2] > 8); return np.where(hit, np.maximum(p, .9), p), hit
def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2"]); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xk = {u: v[keep] for u, v in X.items()}; Lk, EXk = L[keep], EX[keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = V3.weights(kind); realm = kind == "real"
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    for s in (0, 1, 2):
        so = np.zeros(len(y))
        for tr, te in V3.folds(grp, s):
            so[te] = (D.fit("clip", Xk["clip"], y, w, tr, s).predict_proba(Xk["clip"][te])[:, 1] + D.fit("low", Xk["low"], y, w, tr, s).predict_proba(Xk["low"][te])[:, 1]) / 2
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
        sp = (D.fit("clip", Xk["clip"], y, w, bag, s).predict_proba(PX["clip"])[:, 1] + D.fit("low", Xk["low"], y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2
        lo, hit = rule(so, Lk); lp, ph = rule(sp, PL)
        for nm, o, p in [("stack", so, sp), ("L2", lo, lp)]:
            print(f"  s{s} {nm:5s} OOF 균형acc={V3.bal_acc(o, y, t):.4f} | 실제오탐 원본있음={(o[realm & ~EXk] >= .5).mean():.3f} 원본제외={(o[realm & EXk] >= .5).mean():.3f} | "
                  + " ".join(f"{x[:6]}={(o[t == x] >= .5).mean():.2f}" for x in V3.TYPES6) + f" || 공개(참고) {((p >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f}")
        print(f"      규칙 발동: 실제(원본있음) {hit[realm & ~EXk].mean():.3f} 실제(원본제외) {hit[realm & EXk].mean():.3f} 가짜 {hit[y == 1].mean():.3f} | 공개 발동 {[f for f, h in zip(pf, ph) if h]}")
if __name__ == "__main__":
    main()
