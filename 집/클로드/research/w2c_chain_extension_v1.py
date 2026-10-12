# -*- coding: utf-8 -*-
"""W2C (2026-10-13 집 클로드; 사용자: 'W2 사슬 연장 해봐').  사전 고정(이 파일 커밋 후 실행).  EC 만.  서술·평가 적용 계산(제출물 아님).
근거: 6.549 순수 2차 창(F13 216~238, F47 211~231)에서 '같은 ID 번호 −2 = 같은 동 전날'(순서 유지 F13 91%·F47 94%); W2(6.550) 가 F13_235 를 r−2 정답으로
  1.08 → .65 로 내렸고 리더보드 항등식상 실제 ≈ .66(6.783).  F13_237 은 r−2 = F13_235 가 평가 기록이라 W2 미발동.
규칙(평가 기록 r, h 시 행; 규정: r 의 0..h 입력, 같은 ID 앞 평가 기록 입력·계산값, 학습 정답):
  (a) r ≥ 179 이고 r 의 외기 0..h 가 1차(번호 < 179) 어느 기록과도 쌍둥이 아님(표준화 외기 RMS ≤ .05, W2 와 같음);
  (b) r−2 는 평가 기록이고, r−2 에 W2 가 발동(= r−4 정답 있음, r−4 하루 전체가 1차 쌍둥이 없음, r−2 의 짝 조건);
  (c) r 의 짝 조건(r−1 이 r−2 의 외기 쌍둥이, 또는 r−1 지워짐 & r−3 이 r−2 쌍둥이 아님);  정확히 2단계만(재귀 없음 — 다일 드리프트 위험 제한).
  값: a = 기준 열(12회차 또는 13회차 EC)의 r−2 23시 예측(= W2 반영된 계산값), pred_h = clip(p_h + (a − p 0..h 누적평균)).
검증: 학습 순수 창에서 2단계 짝 조건이 모두 맞는 정답 사례는 F47_215 하나(이 파일 작성 전 확인: a4 .681, 실제 .705, DIAG10 모델 .673) → 판정 불가, 가설 규칙.
보고: 평가 발동 기록·행·변화량(12·13회차 기준).
"""
import os, sys, glob
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN  # noqa: E402
idx = {(FARM[i], int(NUM[i])): i for i in range(N)}
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
sd = np.nanstd(X[TRAIN].reshape(-1, 4), 0)
twin = lambda a, b, h=23: float(np.sqrt(np.nanmean(((X[a, :h + 1] - X[b, :h + 1]) / sd) ** 2))) <= .05
P1 = [i for i in range(N) if NUM[i] < 179]
p1twin = lambda i, h=23: any(twin(i, j, h) for j in P1)
def pair(f, r):
    i2, i1, i3 = idx.get((f, r - 2)), idx.get((f, r - 1)), idx.get((f, r - 3))
    return (i1 is not None and i2 is not None and twin(i1, i2)) or (i1 is None and i2 is not None and (i3 is None or not twin(i3, i2)))
def w2_fires(f, r):  # W2 on eval record r (full-day condition on r-2; r's own hour condition checked by caller)
    i2 = idx.get((f, r - 2))
    return i2 is not None and TRAIN[i2] and not p1twin(i2) and pair(f, r)
for lab, d in (("12th", "12회차_2026-10-11(나)"), ("13th", "13회차_2026-10-12(나)")):
    f_ = [p for p in glob.glob(os.path.join(env.ROOT, "제출", d, "*.csv")) if os.path.basename(p).startswith("submission")][0]
    P = pd.read_csv(f_); P["f"] = P.row_id.str[:3]; P["r"] = P.row_id.str[4:7].astype(int); P["h"] = P.row_id.str[8:10].astype(int)
    P = P.sort_values(["f", "r", "h"]).reset_index(drop=True); new = P.sub_ec.values.copy(); log = []
    for (f, r), g in P.groupby(["f", "r"]):
        i, i2 = idx[(f, r)], idx.get((f, r - 2))
        if TRAIN[i] or r < 179 or i2 is None or TRAIN[i2]: continue
        if not (w2_fires(f, r - 2) and pair(f, r)): continue
        a = float(P[(P.f == f) & (P.r == r - 2) & (P.h == 23)].sub_ec.iloc[0]); cm = g.sub_ec.expanding().mean().values; n_ = 0
        for k, (ii, row) in enumerate(g.iterrows()):
            if p1twin(i, int(row.h)): continue
            new[ii] = np.clip(row.sub_ec + (a - cm[k]), lo_, hi_); n_ += 1
        log.append(dict(rec=f"{f}_{r:03d}", via=f"{f}_{r-2:03d}<-{f}_{r-4:03d}", a=round(a, 3), rows=n_, old=round(g.sub_ec.mean(), 3), new=round(new[g.index].mean(), 3)))
    d_ = new - P.sub_ec.values
    print(f"== {lab} ({os.path.basename(f_)}): fired", pd.DataFrame(log).to_string() if log else "none")
    print(f"   change RMS over 1440 rows {np.sqrt(np.mean(d_ ** 2)):.4f}, rows changed {(np.abs(d_) > 1e-9).sum()}")
