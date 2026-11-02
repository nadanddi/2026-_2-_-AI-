"""IM25b: TruFor 무학습 점수 분석(사전 시험). 진짜 대비 AUC, 진짜 오탐 1%·2% 기준에서의 탐지율. 비교 대상: 후보 L 조각 판별기(v2 시드0 OOF top1 AUC .863)."""
import sys, zlib, csv, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
LOCAL = Path(__file__).resolve().parents[1] / "local"
NPZ = sys.argv[1] if len(sys.argv) > 1 else "trufor_scores.npz"; z = np.load(LOCAL / "feats" / NPZ); S, F = z["set"], z["fname"]
farm = {}
for sn in ("im_set_gpt1", "im_set_gpt2"):
    for r in csv.DictReader(open(LOCAL / sn / "manifest.csv", encoding="utf-8")): farm[(sn, r["file"])] = (r["farm_code"], r["kind"])
real = np.isin(S, ["im_set_v4", "im_set_sd2", "im_set_v5"]) & ~np.char.startswith(F, "partial_style")
g = np.array([(s, f) in farm for s, f in zip(S, F)])
val = np.array([g_ and zlib.crc32(farm[(s, f)][0].encode()) % 2 == 1 for g_, s, f in zip(g, S, F)])
kd = np.array([farm[(s, f)][1] if g_ else "" for g_, s, f in zip(g, S, F)])
groups = {"GPT검증 부분편집": val, "GPT검증 부분생성": val & (kd == "gpt_pgen"), "GPT검증 부분스타일": val & (kd == "gpt_pstyle"),
          "GPT전체 부분생성": g & (kd == "gpt_pgen"), "GPT전체 부분스타일": g & (kd == "gpt_pstyle"),
          "6판": S == "im_set_v6", "5판 부분스타일": (S == "im_set_v5") & np.char.startswith(F, "partial_style")}
print("진짜", real.sum())
for key in [k for k in z.files if k not in ("set", "fname")]:
    v = z[key]; t1, t2 = np.quantile(v[real], .99), np.quantile(v[real], .98)
    print(f"[{key}] 진짜 중앙값 {np.median(v[real]):.3f}")
    for nm, sel in groups.items():
        auc = roc_auc_score(np.r_[np.zeros(real.sum()), np.ones(sel.sum())], np.r_[v[real], v[sel]])
        print(f"   {nm:12s} n={sel.sum():3d} AUC {auc:.3f} | 오탐1%에서 탐지 {(v[sel] > t1).mean():.2f} 오탐2% {(v[sel] > t2).mean():.2f}")
pub = S == "public21"; lab = {r["file"]: (r["label"], r["type"]) for r in csv.DictReader(open(Path.home() / "storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플/sample_labels.csv"))}
K0 = sys.argv[2] if len(sys.argv) > 2 else "det"; print(f"공개({K0}, 진짜 분위):", " ".join(f"{f[:4]}({lab[f][1][:7]})={d:.2f}/{np.mean(z[K0][real] < d):.2f}" for f, d in zip(F[pub], z[K0][pub])))
