"""IM49: 공식 6유형별 탐지율 표(나단6 vs 나단8=H2+게이트). 저장 특징(feats_h) 사용. 서버(nadan)."""
import sys, csv, json, numpy as np
from pathlib import Path
sys.argv = [sys.argv[0], "no", ".0025", ".001", "h2", "jig"]
sys.path.insert(0, str(Path(__file__).resolve().parent)); import im44_h_train_eval as H  # noqa
from im43_h_feats import paths_of  # noqa
from im48_std_gate import std  # noqa
L = H.L; z = np.load(L / "preds/h_model_h2.npz"); M = {}
for k, v in z.items(): a, c = k.split("__"); M.setdefault(a, {})[c] = v
th = json.load(open(L / "preds/h_theta_h2.json"))
OFF = {"real": "0 진짜", "ai_gen": "1 전체 생성", "gpt_gen": "1 전체 생성", "gpt3_s1": "1 전체 생성",
       "composite": "2 합성", "filter_blur": "3 필터", "filter_noise": "3 필터", "filter_jpeg": "3 필터", "filter_sp": "3 필터",
       "full_style_sd": "4 전체 스타일", "gpt_fstyle": "4 전체 스타일",
       "partial_gen_v7": "5 부분 생성", "partial_gen_sd": "5 부분 생성", "gpt_pgen": "5 부분 생성", "gpt3_pgen": "5 부분 생성", "qwen_pgen": "5 부분 생성", "qwen_regen": "5 부분 생성", "qwen_brown": "5 부분 생성", "qwen_pcolor": "6 부분 스타일", "qwen_pstyle": "6 부분 스타일", "qwen_fstyle": "4 전체 스타일",
       "partial_style": "6 부분 스타일", "pstyle_color": "6 부분 스타일", "partial_style_sd": "6 부분 스타일", "gpt_pstyle": "6 부분 스타일", "gpt3_pstyle": "6 부분 스타일"}
def rows_of(sn):
    if sn.startswith("sealed_"): return list(csv.DictReader(open(L / "sealed_c3" / sn[7:] / "manifest.csv", encoding="utf-8")))
    return list(csv.DictReader(open(L / sn / "manifest.csv", encoding="utf-8")))
for sn in sys.argv_sets if hasattr(sys, "argv_sets") else ["im_set_r3", "im_set_r2", "sealed_real", "sealed_brown", "sealed_bright", "sealed_v7", "sealed_qwen", "im_set_sd2", "im_set_gpt1", "im_set_gpt3"]:
    try: F = H.load(sn)
    except FileNotFoundError: print(sn, "특징 없음"); continue
    rr = rows_of(sn); s = np.array([std(p) for p in paths_of(sn)]); p6 = F["p6"]; ph = np.where(s, H.apply(p6, H.score(M, F), th["theta"], th["theta_rc"], F["rc"]), p6)
    for r, a, b in zip(rr, p6, ph): print("ROW", sn, OFF.get(r["kind"], "? " + r["kind"]), r["kind"], int(r["label"]), int(a >= .5), int(b >= .5))
print("TABLE_DONE")
