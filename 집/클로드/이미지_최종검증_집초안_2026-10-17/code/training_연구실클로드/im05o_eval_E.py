"""IM05o: 후보 묶음 E 비교(IM_해상도후보E_v1.md) — im05n 일반화: 백본 ID 목록(첫째=기준 B2), 학습 자료 B(v4+SD2+5판), 모두 L2 덧씌움.
사용: python im05o_eval_E.py B2,B2c20,B2c20f,B2f,SG384
원 설명: IM05n: B2 기준 A(기존)·B(+5판) 각각에 L2 원본 대조 규칙 결합 비교 (사용자 지시 10-12).
L2(IM_후보L1_v1.md 등록 그대로): 3.5 < d1 < 20 AND gap > 8 → p ← max(p, .9).
지표: 공개 21장(정확도·로그손실·유형·개별), OOF 실제 오탐(원본 있음/원본 제외), 가짜 유형 탐지.
"""
import sys, numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
V3.TYPE.update({"composite_v5b": "composite", "ai_gen_xl": "AI_gen", "full_style_xl": "full_style", "partial_style_xl": "partial_style", "partial_gen_xl": "partial_gen"})
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa
from im05i_eval_L1 import lookfeats  # noqa
LOCAL = D.LOCAL; SEEDS = (0, 1, 2); BIDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["B2"]
def rule(p, L): hit = (L[:, 0] > 3.5) & (L[:, 0] < 20) & (L[:, 2] > 8); return np.where(hit, np.maximum(p, .9), p)
def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", "im_set_v5"]); L, EX = lookfeats(rows)
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Lk, EXk = L[keep], EX[keep]
    Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); v5 = np.array([r["_set"] == "im_set_v5" for r in R]); realm = kind == "real"
    lab, pf, PX = D.pub_feats(); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    z = np.load(LOCAL / "feats" / "public21_look.npz"); ix = dict(zip(z["files"], range(21))); PL = np.stack([z["feats"][ix[f]] for f in pf])
    m = np.ones(len(R), bool); summ = []
    for BID in BIDS:
        fz = {}
        for sn in ("im_set_v4", "im_set_sd2", "im_set_v5"): fz.update({(sn, k): v for k, v in bb(sn, BID).items()})
        Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); pz = bb("public21", BID); Pb = np.stack([pz[f] for f in pf]); nm = BID
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); pubs = {}
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            p = (D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2
            summ.append((BID, s, V3.bal_acc(rule(o, Lk), y, t), (rule(o, Lk)[realm] >= .5).mean(), log_loss(py, np.clip(rule(p, PL), 1e-4, 1 - 1e-4)), ((rule(p, PL) >= .5) == py).sum()))
            for tag, oo, pp in [(nm + "+L2", rule(o, Lk), rule(p, PL))]:
                pubs.setdefault(tag, []).append(pp)
                print(f"  [{tag:5s}] s{s} 공개 {((pp >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(pp, 1e-4, 1 - 1e-4)):.3f} 실제오탐={(pp[py == 0] >= .5).sum()} | "
                      + " ".join(f"{x[:6]}={((pp[pt == x] >= .5) == (x != 'real')).sum()}" for x in ["real"] + V3.TYPES6)
                      + f" || OOF 실제오탐 원본있음 {(oo[realm & ~EXk] >= .5).mean():.3f} 원본제외 {(oo[realm & EXk] >= .5).mean():.3f} | 기존가짜 "
                      + " ".join(f"{x[:6]}={(oo[~v5 & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6), flush=True)
        for tag, ps in pubs.items():
            print(f"    [{tag}] 공개 개별(시드 평균): " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, np.mean(ps, 0))), flush=True)
    print()
    print("== 요약(+L2, 기준 = 첫 번째) ==")
    base = {s_: r for b, s_, *r in summ if b == BIDS[0]}
    for b in BIDS:
        rs = [(s_, r) for bb_, s_, *r in summ if bb_ == b]
        up = all(r[0] > base[s_][0] for s_, r in rs); fp = all(r[1] <= base[s_][1] + .01 for s_, r in rs); ll = all(r[2] <= base[s_][2] for s_, r in rs)
        print(f"  {b:7s} 균형acc {[round(r[0], 4) for _, r in rs]} 실제오탐 {[round(r[1], 3) for _, r in rs]} 공개 logloss {[round(r[2], 3) for _, r in rs]} 공개 {[int(r[3]) for _, r in rs]}"
              + ("" if b == BIDS[0] else f" → 조건 상승 {up}·오탐 {fp}·logloss {ll} → {'후보' if up and fp and ll else '불채택'}"))


if __name__ == "__main__":
    main()
