# -*- coding: utf-8 -*-
"""FL5 (2026-10-16 집 클로드; 사용자 '3번 연결표도 확인해' — PCH2 비평: FL3 학습 표는 ±1 번호 정답만 숨겨 'excl'(정답 이어짐) 특징이 같은 폴드 가림 기록의 정답을 씀 -> 검증 사슬 연결이 평가보다 낙관적).
연결표 생성(판정 없음), 실행 전 커밋.  검증기 거울 연결표: 배치(EXT10·EXT12·DIAG10) 폴드 k 의 가림 기록 b 마다
  labeled = 학습 기록 - (그 폴드 가림 기록 전체) - (b 와 같은 농장 번호 ±1)   <- 평가 때처럼 '앞 기록에 정답이 없음' 을 재현
  removed = 평가 기록(입력 안 씀, FL3 와 같음),  가중치 = FL3 의 fl3_w.npy(다시 맞추지 않음),  0시 결정.
  -> local/fl5_mirror_links.csv (layout, fold, rec, prev, p).  서술: FL3 학습 표와 고른 전날 일치율, p >= .8 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl5_mirror_links_validators_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; w = np.load(os.path.join(env.LOCAL, "fl3_w.npy"))
def pick(b, lab, removed):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FL].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
RID = {rid(b): b for b in range(N)}
CK = os.path.join(env.LOCAL, "tt1_ckpt"); G = pd.concat([pd.read_csv(os.path.join(CK, f), usecols=["validator", "fold", "row_id"]) for f in sorted(os.listdir(CK))])
G["rec"] = G.row_id.str[:7]; G = G[G.validator.isin(["EXT10", "EXT12", "DIAG10"])].drop_duplicates(["validator", "fold", "rec"])
out = []
for (L, k), S in G.groupby(["validator", "fold"]):
    held = np.array([RID[r] for r in S.rec if r in RID])
    for b in held:
        lab = TRAIN.copy(); lab[held] = False; lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        a0, p0 = pick(b, lab, set(TEST_IDX))
        if a0 is not None: out.append(dict(layout=L, fold=int(k), rec=rid(b), prev=rid(a0), p=round(p0, 4)))
    print(L, k, "done", len(held), flush=True)
T = pd.DataFrame(out); T.to_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv"), index=False)
F3 = pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv")).set_index("rec"); T["fl3_prev"] = T.rec.map(F3.prev); T["fl3_p"] = T.rec.map(F3.p)
for L, S in T.groupby("layout"):
    print(f"{L}: recs {len(S)} | same prev as FL3 {np.mean(S.prev == S.fl3_prev):.3f} | p>=.8 mirror {np.mean(S.p >= .8):.3f} vs FL3 {np.mean(S.fl3_p >= .8):.3f} | mean p {S.p.mean():.3f} vs {S.fl3_p.mean():.3f}")
