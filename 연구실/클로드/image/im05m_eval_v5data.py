"""IM05m: 가짜 제작 5판 자료 효과 비교 (사용자 지시 10-12 'B2로 5판 자료 비교'). 결과 보기 전 고정.
백본 B2(CLIP L/14 @336) LR + 저수준 HistGB 평균(stack). 학습 자료만 바꿈:
 A = v4 + SD2 (기준)            B = A + 5판 가짜            C = 실제(전부) + 5판 가짜만
5판 구성: filter_blur, composite_v5b, ai_gen_xl, full_style_xl, partial_style_xl, partial_gen_xl (composite_v5 첫 버전은 직선 경계 문제로 제외).
평가: 공개 21장(운영진 가짜, 세 안 공통 — 중심), OOF(같은 농장 폴드)는 '기존 가짜' 탐지·실제 오탐(비악화 확인)과 '5판 가짜' 탐지(참고).
판정(로컬): 공개 로그손실이 시드 3개 모두 A보다 낮고, 공개 정확도가 낮지 않고, 실제 오탐(OOF)이 A+1%p 이내 → 리더보드 후보. 최종은 리더보드.
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05c_eval_v3 as V3  # noqa
V3.TYPE.update({"composite_v5b": "composite", "ai_gen_xl": "AI_gen", "full_style_xl": "full_style", "partial_style_xl": "partial_style", "partial_gen_xl": "partial_gen"})
import im05d_eval_v4 as D  # noqa
from im05k_backbones import bb  # noqa

LOCAL = D.LOCAL; SEEDS = (0, 1, 2); BID = sys.argv[1] if len(sys.argv) > 1 else "B2"
V5KINDS = {"filter_blur", "composite_v5b", "ai_gen_xl", "full_style_xl", "partial_style_xl", "partial_gen_xl"}

def main():
    rows, X, isred = D.load(["im_set_v4", "im_set_sd2", "im_set_v5"])
    keep = ~isred; R = [r for r, k in zip(rows, keep) if k]
    fz = {}
    for sn in ("im_set_v4", "im_set_sd2", "im_set_v5"): fz.update({(sn, k): v for k, v in bb(sn, BID).items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R]); Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); t = np.array([V3.TYPE[k] for k in kind])
    grp = np.array([r["farm_code"] for r in R]); v5 = np.array([r["_set"] == "im_set_v5" for r in R]); real = y == 0
    lab, pf, PX = D.pub_feats(); pz = bb("public21", BID); Pb = np.stack([pz[f] for f in pf]); py = np.array([int(r["label"]) for r in lab]); pt = np.array([r["type"] for r in lab])
    masks = {"A": ~v5, "B": np.ones(len(R), bool), "C": real | v5}
    print(f"백본 {BID} | 행 {len(R)} (5판 {v5.sum()}) | " + " ".join(f"{x}={((t == x) & v5).sum()}" for x in V3.TYPES6), flush=True)
    for nm, m in masks.items():
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m])
        out = []
        for s in SEEDS:
            o = np.zeros(len(y))
            for tr, te in V3.folds(grp, s):
                tr = tr[m[tr]]
                o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
            rng = np.random.default_rng(1000 + s); farms = np.array(sorted(set(grp))); bag = np.where(np.isin(grp, rng.choice(farms, int(.9 * len(farms)), replace=False)) & m)[0]
            p = (D.fit("clip", Xb, y, w, bag, s).predict_proba(Pb)[:, 1] + D.fit("low", Xl, y, w, bag, s).predict_proba(PX["low"])[:, 1]) / 2
            old = ~v5 & ~real
            out.append(p)
            print(f"  [{nm}] s{s} 공개 {((p >= .5) == py).sum()}/21 logloss={log_loss(py, np.clip(p, 1e-4, 1 - 1e-4)):.3f} auc={roc_auc_score(py, p):.3f} 실제오탐={(p[py == 0] >= .5).sum()} | "
                  + " ".join(f"{x[:6]}={((p[pt == x] >= .5) == (x != 'real')).sum()}" for x in ["real"] + V3.TYPES6)
                  + f" || OOF 실제오탐 {(o[real] >= .5).mean():.3f} | 기존가짜 " + " ".join(f"{x[:6]}={(o[old & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6 if (old & (t == x)).any())
                  + " | 5판가짜 " + " ".join(f"{x[:6]}={(o[v5 & (t == x)] >= .5).mean():.2f}" for x in V3.TYPES6 if (v5 & (t == x)).any()), flush=True)
        print(f"    [{nm}] 공개 개별(시드 평균): " + " ".join(f"{f[:4]}={q:.2f}" for f, q in zip(pf, np.mean(out, 0))), flush=True)

if __name__ == "__main__":
    main()
