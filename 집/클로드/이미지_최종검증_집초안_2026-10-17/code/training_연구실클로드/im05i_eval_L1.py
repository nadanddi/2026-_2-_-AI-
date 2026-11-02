"""IM05i: 후보 L1 판정 (IM_후보L1_v1.md 고정 정의).
기준 stack = (CLIP 패치평균 LR + 저수준 HistGB)/2, 자료 v4 + SD2, 빨간 조명 행은 진단만.
L1 = 로지스틱 결합기[logit(stack), d1, d2, gap, log d1, log(gap+.1), logit·log(gap+.1)] — 2단 OOF(같은 폴드).
지표: OOF 균형 정확도, 실제 오탐(원본 있음/원본 제외), 유형별 탐지. 공개 21장은 참고 서술만.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05d_eval_v4 as D, im05c_eval_v3 as V3  # noqa

LOCAL = D.LOCAL; SEEDS = (0, 1, 2)
def lg(p): p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))

def lookfeats(rows):
    cache = {}; out, ex = [], []
    for r in rows:
        sn = "im_set_v2red" if r["_set"] == "im_set_v2" else r["_set"]
        if sn not in cache:
            z = np.load(LOCAL / "feats" / f"{sn}_look.npz"); cache[sn] = (dict(zip(z["files"], range(len(z["files"])))), z["feats"], z["excl"])
        ix, f, e = cache[sn]; i = ix[r["file"]]; out.append(f[i]); ex.append(e[i])
    return np.array(out, np.float32), np.array(ex, bool)

def meta_X(p, L):
    d1, d2, gap = L[:, 0], L[:, 1], L[:, 2]; z = lg(p); lgap = np.log(gap + .1)
    return np.stack([z, d1, d2, gap, np.log(d1 + .01), lgap, z * lgap], 1)

def meta(): return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2"])
    L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xk = {u: v[keep] for u, v in X.items()}; Lk, EXk = L[keep], EX[keep]
    Xr = {u: v[isred] for u, v in X.items()}; Lr = L[isred]; yr = np.array([int(r["label"]) for r, k in zip(rows, isred) if k])
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = V3.weights(kind)
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    realm = kind == "real"
    print(f"학습 행 {len(R)} | 실제 {realm.sum()} (원본 제외 {(realm & EXk).sum()}) | 공개 21장은 참고 서술만", flush=True)
    for s in SEEDS:
        so = np.zeros(len(y))
        for tr, te in V3.folds(grp, s):
            so[te] = (D.fit("clip", Xk["clip"], y, w, tr, s).predict_proba(Xk["clip"][te])[:, 1] + D.fit("low", Xk["low"], y, w, tr, s).predict_proba(Xk["low"][te])[:, 1]) / 2
        M = meta_X(so, Lk); mo = np.zeros(len(y))
        for tr, te in V3.folds(grp, s):
            mo[te] = meta().fit(M[tr], y[tr], logisticregression__sample_weight=w[tr]).predict_proba(M[te])[:, 1]
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp)))
        bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
        mc, ml = D.fit("clip", Xk["clip"], y, w, bag, s), D.fit("low", Xk["low"], y, w, bag, s)
        sp = (mc.predict_proba(PX["clip"])[:, 1] + ml.predict_proba(PX["low"])[:, 1]) / 2
        sr = (mc.predict_proba(Xr["clip"])[:, 1] + ml.predict_proba(Xr["low"])[:, 1]) / 2
        mm = meta().fit(M, y, logisticregression__sample_weight=w)
        mp = mm.predict_proba(meta_X(sp, PL))[:, 1]; mr = mm.predict_proba(meta_X(sr, Lr))[:, 1]
        for nm, o, p, rp in [("stack", so, sp, sr), ("L1", mo, mp, mr)]:
            print(f"  s{s} {nm:5s} OOF 균형acc={V3.bal_acc(o, y, t):.4f} | 실제오탐 원본있음={(o[realm & ~EXk] >= .5).mean():.3f} 원본제외={(o[realm & EXk] >= .5).mean():.3f} | "
                  + " ".join(f"{x[:6]}={(o[t == x] >= .5).mean():.2f}" for x in V3.TYPES6)
                  + f" || 공개(참고) {((p >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f} 실제오탐={(p[py == 0] >= .5).sum()} | 빨강실제정답 {(rp[yr == 0] < .5).mean():.2f}", flush=True)
            if s == 0: print("      공개 개별: " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, p)))
        if s == 0:
            coef = mm[-1].coef_[0]; print("      L1 계수(표준화): " + " ".join(f"{n}={c:+.2f}" for n, c in zip(["logit", "d1", "d2", "gap", "logd1", "loggap", "z*loggap"], coef)))

if __name__ == "__main__":
    main()
