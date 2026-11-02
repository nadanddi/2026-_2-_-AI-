"""IM05: CLIP 특징 위 선형 판별기(B0) — 원본 계보 그룹 단위 교차검증 + 공개 21장 외부 점검.
- 교차검증: farm_code(파일명 앞 C??) 단위 GroupKFold(5). 같은 원본의 실제/가짜는 항상 같은 폴드.
- 특징 집계 변형(사전 고정): G=전체 보기 v0, P=패치 9개 평균, GP=둘 이어붙임, MIL=패치별 분류 후 평균/최대 확률.
- 공개 21장은 학습에 쓰지 않고, 전체 학습 모델로 예측만 함.
- red_cast(붉은 조명 tar) 행은 v1에서 제외(공개 실제 사진에 없음, 코덱스 6.500 부근 사용자 정정과 같은 판단).
"""
import sys, csv, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "local"
PUB = HERE.parents[2] / "공용/대회자료/이미지데이터/참가자_배포/공개샘플"
RED_TARS = {"VS_2.설향1"}

def load(name):
    z = np.load(LOCAL / "feats" / f"{name}_clipL14.npz")
    F = z["feats"].astype(np.float32); F /= np.linalg.norm(F, axis=-1, keepdims=True)
    return F, list(z["files"])

def agg(F, mode):
    if mode == "G": return F[:, 0]
    if mode == "P": return F[:, 1:].mean(1)
    if mode == "GP": return np.concatenate([F[:, 0], F[:, 1:].mean(1)], 1)
    raise ValueError(mode)

def clf(C): return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))

def fit_predict(Ftr, ytr, Fte, mode, C):
    if mode.startswith("MIL"):
        Xtr = Ftr[:, 1:].reshape(-1, Ftr.shape[-1]); yr = np.repeat(ytr, Ftr.shape[1] - 1)
        m = clf(C).fit(Xtr, yr)
        p = m.predict_proba(Fte[:, 1:].reshape(-1, Fte.shape[-1]))[:, 1].reshape(len(Fte), -1)
        return p.max(1) if mode == "MILmax" else p.mean(1)
    m = clf(C).fit(agg(Ftr, mode), ytr)
    return m.predict_proba(agg(Fte, mode))[:, 1]

def main(set_name="set_v1", C=0.1):
    F, files = load(set_name)
    man = {r["file"]: r for r in csv.DictReader(open(LOCAL / "im_set_v1" / "manifest.csv", encoding="utf-8"))}
    rows = [man[f] for f in files]
    keep = np.array([r["tar"] not in RED_TARS for r in rows])
    F = F[keep]; rows = [r for r, k in zip(rows, keep) if k]
    y = np.array([int(r["label"]) for r in rows]); kind = np.array([r["kind"] for r in rows])
    grp = np.array([r["farm_code"] for r in rows])
    Fp, pf = load("public21")
    plab = {r["file"]: r for r in csv.DictReader(open(PUB / "sample_labels.csv"))}
    py = np.array([int(plab[f]["label"]) for f in pf]); pk = np.array([plab[f]["type"] for f in pf])
    print(f"n={len(y)} (fake {y.sum()}), farms={len(set(grp))}: {sorted(set(grp))}")
    for mode in ["G", "P", "GP", "MILmean", "MILmax"]:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, grp):
            oof[te] = fit_predict(F[tr], y[tr], F[te], mode, C)
        acc = ((oof >= .5) == y).mean()
        per = {k: ((oof[kind == k] >= .5) == y[kind == k]).mean() for k in sorted(set(kind))}
        pp = fit_predict(F, y, Fp, mode, C)
        pacc = ((pp >= .5) == py).mean()
        print(f"\n[{mode}] CV acc={acc:.3f} auc={roc_auc_score(y, oof):.3f} | " + " ".join(f"{k}={v:.2f}" for k, v in per.items()))
        print(f"   public21 acc={pacc:.3f} auc={roc_auc_score(py, pp):.3f}  real-correct={((pp[py==0]<.5)).sum()}/3")
        print("   " + " ".join(f"{f[:4]}({t[:6]})={p:.2f}" for f, t, p in zip(pf, pk, pp)))

if __name__ == "__main__":
    main(*(sys.argv[1:2]), *(float(x) for x in sys.argv[2:3]))
