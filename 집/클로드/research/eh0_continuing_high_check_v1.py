# -*- coding: utf-8 -*-
"""EH0 (2026-10-14 집 클로드; EL1E·EP1R 후속, 서술).  '고EC 가 이어지는 중' 조건: 같은 동 앞 기록(cert 1일 또는 CP2 빈 날 연결, 2일 이내)이 정답 고EC 이고 하루 운영 지문 거리 <= .27(학습 고EC 유지 쌍 최대).
그 조건을 만족하는 학습 기록의 실제 고EC 비율·기본(OOF) 잔차 — '이어지는 고EC 면 +.3 정도 올려도 되나' 의 근거 확인."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
P = pd.read_csv(os.path.join(HERE, "ep1r_pairs_v1.csv")); T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")).set_index("rec")
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock]; R = A.groupby("rec").agg(y=("y", "mean"), base=("base_oof", "mean")); R["res"] = R.y - R.base
P["gap"] = [T.at[b, "day"] - T.at[a, "day"] for a, b in zip(P.a, P.b)]
P["a_high"] = P.change.isin(["keep_high", "END"]); P = P[P.b.isin(R.index)]
for cut in (.27, .35):
    m = P.a_high & (P.fp <= cut) & (P.gap <= 2)
    G = P[m].drop_duplicates("b"); r = R.loc[G.b]
    print(f"condition (ancestor high, gap<=2, fp<={cut}): records {len(G)} | truly high {int((r.y >= 1).sum())} ({(r.y >= 1).mean():.2f}) | residual mean {r.res.mean():+.3f} median {r.res.median():+.3f} | min {r.res.min():+.3f} | base mean {r.base.mean():.3f}")
m2 = P.a_high & (P.fp > .27); print("ancestor high but fp > .27:", P[m2][["a", "b", "fp", "change", "dEC"]].round(3).to_string(index=False))
