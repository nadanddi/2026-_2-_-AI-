"""IM05k: 백본 비교 평가 (IM_백본비교_v1.md). 각 백본 특징(feats/<set>_<BID>.npz, 조각 평균) → LR(C=.05) → 저수준 HistGB와 평균.
보조 세트는 B1(현재 CLIP) 특징으로 한 번만 골라 모든 백본에 공통 적용. 시드 3개(폴드 순열 + 90% 농장 부분표본).
사용: python im05k_backbones.py B1,B2,...  [sets=im_set_v4,im_set_sd2]
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05d_eval_v4 as D, im05c_eval_v3 as V3  # noqa

LOCAL = D.LOCAL; SEEDS = (0, 1, 2)
SIZE_GB = {"B1": .61, "B2": .61, "B3": .61, "B4": .86, "B5": .61, "B6": .40, "B7": .61}   # fp16 시각부 근사(8비트 저장 시 절반)

def bb(name, bid):
    p = LOCAL / "feats" / f"{name}_{bid}.npz"
    if not p.exists(): return None
    z = np.load(p); F = z["feats"].astype(np.float32); F /= np.linalg.norm(F, axis=-1, keepdims=True) + 1e-8
    return dict(zip(z["files"], F[:, 1:].mean(1)))

def main(bids, sets=("im_set_v4", "im_set_sd2")):
    rows, X, isred = D.load(list(sets))
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); w = V3.weights(kind); Xlow = X["low"][keep]; Xc = X["clip"][keep]
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    S = Xc / np.linalg.norm(Xc, axis=1, keepdims=True); PS = PX["clip"] / np.linalg.norm(PX["clip"], axis=1, keepdims=True); aux = []
    for x, n in [("real", 240)] + [(x, 40) for x in V3.TYPES6]:
        c = np.where(t == x)[0]
        if len(c): aux += list(c[np.argsort(-(S[c] @ PS[pt == x].T).max(1))[:n]])
    aux = np.array(aux); print(f"학습 행 {len(R)}, 보조 {len(aux)} | 공개 21장은 참고 서술(열람 상한 초과)", flush=True)
    low_oof, low_pub = {}, {}
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in V3.folds(grp, s): o[te] = D.fit("low", Xlow, y, w, tr, s).predict_proba(Xlow[te])[:, 1]
        rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
        low_oof[s] = o; low_pub[s] = D.fit("low", Xlow, y, w, bag, s).predict_proba(PX["low"])[:, 1]
    summary = []; keep_prob = {}
    for bid in bids:
        fz = {}
        for sn in sets: fz.update({(sn, k): v for k, v in (bb(sn, bid) or {}).items()})
        pz = bb("public21", bid)
        if pz is None or not all((r["_set"], r["file"]) in fz for r in R): print(f"[{bid}] 특징 없음 → 건너뜀"); continue
        Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); Pb = np.stack([pz[f] for f in pf]); res = []
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s): o[te] = D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1]
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)))[0]
            p = D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1]
            keep_prob[(bid, s)] = (o, p)
            so, sp = (o + low_oof[s]) / 2, (p + low_pub[s]) / 2
            res.append((V3.bal_acc(so, y, t), V3.bal_acc(so[aux], y[aux], t[aux]), (so[kind == "real"] >= .5).mean(), ((sp >= .5) == py).sum(),
                        log_loss(py, np.clip(sp, 1e-4, 1 - 1e-4)), roc_auc_score(py, sp), V3.bal_acc(o, y, t),
                        " ".join(f"{x[:6]}={(so[t == x] >= .5).mean():.2f}" for x in V3.TYPES6)))
        a = np.array([r[:7] for r in res], float); summary.append((bid, a.mean(0)))
        print(f"[{bid}] 결합 OOF 균형acc {a[:, 0].round(4)} | 보조 {a[:, 1].round(4)} | 실제오탐 {a[:, 2].round(3)} | 백본단독 OOF {a[:, 6].round(4)} | "
              f"공개(참고) {a[:, 3].astype(int)}/21 logloss {a[:, 4].round(3)} auc {a[:, 5].round(3)}\n      유형별(s0): {res[0][7]}", flush=True)
    # 2개 앙상블(사전 등록 10-12): 백본 확률 평균 → 저수준과 평균. 모든 쌍.
    done = sorted({b for b, _ in keep_prob})
    for i1, a in enumerate(done):
        for b2 in done[i1 + 1:]:
            res = []
            for s in SEEDS:
                (oa, pa), (ob, pb) = keep_prob[(a, s)], keep_prob[(b2, s)]
                so, sp = ((oa + ob) / 2 + low_oof[s]) / 2, ((pa + pb) / 2 + low_pub[s]) / 2
                res.append((V3.bal_acc(so, y, t), V3.bal_acc(so[aux], y[aux], t[aux]), (so[kind == "real"] >= .5).mean(), ((sp >= .5) == py).sum(),
                            log_loss(py, np.clip(sp, 1e-4, 1 - 1e-4)), roc_auc_score(py, sp), V3.bal_acc((oa + ob) / 2, y, t)))
            a_ = np.array(res, float); sz = SIZE_GB.get(a, 0) + SIZE_GB.get(b2, 0)
            summary.append((f"{a}+{b2} ({sz:.2f}GB, 8비트 {sz / 2:.2f})", a_.mean(0)))
            print(f"[{a}+{b2}] 결합 {a_[:, 0].round(4)} | 실제오탐 {a_[:, 2].round(3)} | 공개(참고) {a_[:, 3].astype(int)}", flush=True)
    print("\n== 순위 (결합 OOF 균형 정확도 시드 평균) ==")
    for bid, m in sorted(summary, key=lambda z: -z[1][0]):
        print(f"  {bid}: 결합 {m[0]:.4f} | 보조 {m[1]:.4f} | 실제오탐 {m[2]:.3f} | 백본단독 {m[6]:.4f} | 공개(참고) {m[3]:.1f}/21 logloss {m[4]:.3f}")

if __name__ == "__main__":
    main(sys.argv[1].split(","), tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("im_set_v4", "im_set_sd2"))
