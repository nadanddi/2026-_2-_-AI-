# -*- coding: utf-8 -*-
"""GEN1-TEMP (2026-10-14 집 클로드; 사용자: '완전 일반 구성 측정 진행해봐').  서술(판정 아님).  TV3 하네스 재사용.
비교: G = W40G-S 기본만(일반), GV = G + V3(일반), C = v23(사슬 새벽 보정 + v7 + TNA8 앵커, 구조 의존), CV = C + V3.  [기본의 계절 지표는 외기 일치 추정 — 포함된 상태]
원문 TV3: (2026-10-14 집 클로드; 사용자: 'V3 다시 판정해봐').  사전 고정(커밋 후 실행).  온도.  자기 기록 입력만(일반 방법).
V3 = 2층 GRU(48시간 원입력) 10% 섞기, 날 문턱 10℃ (6.531 TS7 승격, 6.532 TT26 묶음 K PASS·TS7b 추운 안전 PASS) — 9회차 기본(W40G-S) 위에서만 판정, 한 번도 제출 안 함.
14회차 온도(v23) = W40G-S → TC1c 사슬 새벽 보정 → v7 + TNA8 거울 앵커.  여기서 V3 를 기본 단계에 넣고 같은 사슬·앵커를 거친 것을 v23 과 비교.
  섞기(원래 규칙 그대로): a = W40G-S 셀 출력, g = clip((in_temp − 8)/2, 0, 1)(결측 1), h = clip((그날 0시~현재 in_temp 최저 − 10)/2, 0, 1)(결측 1), b = a + .1 g h (V3 − a).
  V3 저장 예측: DIAG10·EL1·EXT10·EXT12 = ts7b_ckpt(시드 1·2), F·G·H·I = ts7_ckpt, K = tt26_ckpt(V3 열).  사슬 보정기 학습 틀(D = DIAG10 프레임)은 그대로.
판정(단일 안 α .025): DIAG10·EL1·F·G·H·I·K 모든 셀(4) 전체 행 개선 & 합친(DIAG10+F+G+H+I+K, 농장 x 5일 블록, 셀 1 기준) P(악화) < .025
  & EXT10·EXT12 안전(셀 평균 악화이면서 '나음' 비율 < .025 이면 FAIL).  서술: 2차 행, 추운 행.
[한계: V3 가 이미 판정받은 같은 배치들 — 새 정보는 v23(사슬·앵커) 위에서도 이득이 남는지뿐]
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tb2_temp_agree_any_p_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb2_temp_agree_any_p_v1.py")
exec(src[:src.index("DD = pd.read_pickle")])
TNA8M = dict(ARMS["TNA8"][1])
src = open(os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ext1_combo_b4_tn8_window_v1.py")
exec(src[:src.index("res = {k: []")])
__file__ = os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"); rng = np.random.default_rng(20261224)
LV = pd.concat([L7f, pd.DataFrame(dict(rec=list(TNA8M), prev=list(TNA8M.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TNA8M.values()], dT_air=0.0, steps_to_anchor=1))], ignore_index=True)
def v3_for(L, M):
    if "V3" in M.columns: return M.V3.values
    if L in ("DIAG10", "EL1", "EXT10", "EXT12"):
        d = os.path.join(env.LOCAL, "ts7b_ckpt"); X = pd.concat([pd.read_csv(os.path.join(d, f)) for f in os.listdir(d) if f.startswith(L + "_")]); X = X[X.validator == L]
    else:
        d = os.path.join(env.LOCAL, "ts7_ckpt"); X = pd.concat([pd.read_csv(os.path.join(d, f)) for f in os.listdir(d) if f.startswith(L + "_")])
    return M[["row_id"]].merge(X[["row_id", "V3"]].drop_duplicates("row_id"), on="row_id", how="left").V3.values

rows = []
for L in ["DIAG10", "EL1", "F", "G", "H", "I", "K", "EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); y = M.sub_temp.values; v3 = v3_for(L, M); miss = np.isnan(v3)
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
    order = M.sort_values(["farm", "day", "hour"]).index; mm = M.loc[order].groupby(["farm", "day"]).in_temp.cummin().reindex(M.index)
    h = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1)); wgt = .1 * g * h * (~miss); p2 = M.day.values >= 179
    acc = {k: [] for k in ("G", "GV", "C", "CV")}
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam); a2 = a + wgt * (np.nan_to_num(v3) - a)
            out = {"G": a, "GV": a2, "C": anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV), "CV": anch(M, C2.corrected(M, a2, D1, bi, fam, 7), LV)}
            for k, p in out.items(): acc[k].append((rm(p - y), rm((p - y)[p2])))
    rows.append(dict(layout=L, **{f"{k}_all": np.mean([x[0] for x in v]) for k, v in acc.items()}, **{f"{k}_p2": np.mean([x[1] for x in v]) for k, v in acc.items()}))
    print("done", L, flush=True)
R = pd.DataFrame(rows).set_index("layout"); pd.set_option("display.width", 240); print(R.round(4).to_string())
for sc in ("all", "p2"):
    print(f"
[{sc}] general G -> GV {100 * (R[f'GV_{sc}'] / R[f'G_{sc}'] - 1).round(2).to_dict()}")
    print(f"[{sc}] structure C vs general G {100 * (R[f'C_{sc}'] / R[f'G_{sc}'] - 1).round(2).to_dict()}")
    print(f"[{sc}] best general GV vs current C {100 * (R[f'GV_{sc}'] / R[f'C_{sc}'] - 1).round(2).to_dict()}")
