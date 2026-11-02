"""IM05s 진단(판정 없음): 6판 OOF 확률을 마스크 넓이 구간 × 인페인터로 나눠, 실제 사진 분포와 겹치는지(라벨 잡음 여부) 본다.
A(6판 학습 안 함)와 D1a(6판 추가) 두 안, 시드 0. 각 칸: 장수, 평균p, 탐지율(.5), 실제 대비 AUC.
"""
import sys, re, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import im05s_eval_pgen6 as S  # noqa
P, V3, D, bb, lookfeats = S.P, S.V3, S.D, S.bb, S.lookfeats

def main():
    rows, X, isred = D.load(S.SETS); keep = ~isred; R = [r for r, k in zip(rows, keep) if k]; Xl = X["low"][keep]
    kind = np.array([r["kind"] for r in R]); y = np.array([int(r["label"]) for r in R]); grp = np.array([r["farm_code"] for r in R])
    sets = np.array([r["_set"] for r in R]); gpt = sets == "im_set_gpt1"; v6 = sets == "im_set_v6"; realm = kind == "real"
    area = np.array([float(re.search(r"area=([\d.]+)", r["info"]).group(1)) if r["_set"] == "im_set_v6" else np.nan for r in R])
    gen = np.array([r["info"].split("|")[0] if r["_set"] == "im_set_v6" else "" for r in R])
    fz = {}
    for sn in S.SETS: fz.update({(sn, k): v for k, v in bb(sn, "B2").items()})
    Xb = np.stack([fz[(r["_set"], r["file"])] for r in R])
    for arm, m in {"A": ~gpt & ~v6, "D1a": ~gpt}.items():
        w = np.zeros(len(R)); w[m] = V3.weights(kind[m]); o = np.zeros(len(y)); s = 0
        for tr, te in V3.folds(grp, s):
            tr = tr[m[tr]]; o[te] = (D.fit("clip", Xb, y, w, tr, s).predict_proba(Xb[te])[:, 1] + D.fit("low", Xl, y, w, tr, s).predict_proba(Xl[te])[:, 1]) / 2
        print(f"[{arm}] 실제 평균p {o[realm].mean():.3f} 오탐 {(o[realm] >= .5).mean():.3f}")
        for g in ("xl", "sd15"):
            for lo, hi in [(0, .05), (.05, .10), (.10, .15), (.15, 1)]:
                sel = v6 & (gen == g) & (area >= lo) & (area < hi)
                if sel.sum() < 5: continue
                auc = roc_auc_score(np.r_[np.zeros(realm.sum()), np.ones(sel.sum())], np.r_[o[realm], o[sel]])
                print(f"  {g:4s} 넓이 {lo:.2f}~{hi:.2f}: n={sel.sum():3d} 평균p {o[sel].mean():.3f} 탐지 {(o[sel] >= .5).mean():.2f} 실제대비AUC {auc:.3f}", flush=True)

if __name__ == "__main__":
    main()
