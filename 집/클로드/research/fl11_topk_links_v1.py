# -*- coding: utf-8 -*-
"""FL11 (2026-10-16 집 클로드; 사용자 '더 찾아라' -> AG6(6.709, v7 기반 층별 보조 앵커: 전날 정답 후보들의 23시 배지를 확률 가중 평균, 4칸 −.7%) 를 현장형 FL 로 다시: 후보 상위 5개와 확률 저장).
연결표 생성(판정 없음), 실행 전 커밋.  FL5/FL8 과 같은 폴드 거울·후보·가중(fl3_w).  -> local/fl11_topk_links.csv (layout, fold, rec, rank, cand, p).
원 FL8 설명: (2026-10-16 집 클로드; TAG1 비평(7.029): 일치 링커가 번호·달력 특징 + 좁은 숨김 -> 5점·누설 문제.  현장형 두 갈래 증거로 다시 정의).
연결표 생성(판정 없음), 실행 전 커밋.  FL5 와 같은 폴드 거울(배치 EXT10·EXT12·DIAG10 폴드 k 의 가림 기록 전체 + 같은 농장 번호 ±1 정답 숨김, 평가 기록 입력 안 씀),
같은 후보(features7)와 같은 가중치(fl3_w.npy)에서 세 가지 선택:
  FL  = 전체 점수(oj + v0..v7 + excl + xf)  — FL5 와 같음(재현 확인)
  OUT = 바깥 날씨 이어짐만(oj: 어제 23시 외기 4종 외삽 vs 오늘 0시) + excl + xf
  IN  = 실내 센서·운영 이어짐만(v0..v7: 실내 온·습·CO2 + 구동기 5종의 어제 23시 vs 오늘 0시) + excl + xf
번호 차·달력 위치 특징(d1·d3·ldn·d2w·d2c·isprev·dsdev) 은 FL 처럼 쓰지 않음.
-> local/fl8_two_view_links.csv (layout, fold, rec, prev, p, prev_out, p_out, prev_in, p_in).  서술: FL5 와 일치율, 갈래 일치 분포, 정답 표(cert) 진짜 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl8_two_view_mirror_links_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; w = np.load(os.path.join(env.LOCAL, "fl3_w.npy"))
MO = np.array([1] + [0] * 8 + [1, 1], float); MI = np.array([0] + [1] * 8 + [1, 1], float)
def soft(s): p = np.exp(s - s.max()); return p / p.sum()
def picks(b, lab, removed):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None
    p = soft(df[FL].values.astype(float) @ w); o = np.argsort(-p)[:5]
    return [dict(rank=i, cand=rid(int(df.a.values[k])), p=round(float(p[k]), 5)) for i, k in enumerate(o)]
RID = {rid(b): b for b in range(N)}
CK = os.path.join(env.LOCAL, "tt1_ckpt"); G = pd.concat([pd.read_csv(os.path.join(CK, f), usecols=["validator", "fold", "row_id"]) for f in sorted(os.listdir(CK))])
G["rec"] = G.row_id.str[:7]; G = G[G.validator.isin(["EXT10", "EXT12", "DIAG10"])].drop_duplicates(["validator", "fold", "rec"])
out = []
for (L, k), S in G.groupby(["validator", "fold"]):
    held = np.array([RID[r] for r in S.rec if r in RID])
    for b in held:
        lab = TRAIN.copy(); lab[held] = False; lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        o = picks(b, lab, set(TEST_IDX))
        if o is not None: out += [dict(layout=L, fold=int(k), rec=rid(b), **d) for d in o]
    print(L, k, "done", len(held), flush=True)
D = pd.DataFrame(out); D.to_csv(os.path.join(env.LOCAL, "fl11_topk_links.csv"), index=False); print("rows", len(D), "| records", D.groupby(["layout", "fold", "rec"]).ngroups)
