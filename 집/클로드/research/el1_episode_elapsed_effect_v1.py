# -*- coding: utf-8 -*-
"""EL1E (2026-10-14 집 클로드; 사용자: '2 → 1 순서대로' — 2 = 고EC 경과일).  서술(판정 전 점검).  EC.
HS4(6.697, 진단 연결): 고EC 잔차 경과 0일 +.01 → 1~2일 +.24 → 3~5일 +.33 → 6일+ +.35.  AG1(6.698): 입력만으론 경과일 추정 실패(지워진 날에서 사슬 끊김).
여기: CP2 동 트랙(1일 cert + 빈 날 잇기, gap_days)으로 경과일을 다시 셈 — 에피소드 = 트랙 안에서 고EC(하루 평균 >= 1.0) 날이 이어지는 구간(빈 날 <= 2일은 이어진 것으로).
  경과일 e = 에피소드 첫 고EC 날부터의 학습 달력 날짜 차.  잔차 = y − 기본(DIAG10 CUR OOF), 하루 평균.  잠금 날 제외.
(1) 경과일 구간별 고EC 기록 잔차(평균·n), 에피소드 수.  (2) 에피소드 하나씩 빼고 '경과일 구간 평균 잔차' 로 맞혔을 때 오차 vs 0 보정(일반화 확인).
(3) F13 평가 덩어리: F13_217 의 트랙을 거슬러 고EC 가 언제 시작됐나 → F13_221·223·225 의 추정 경과일.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")); A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"))
A = A[A.is_train].copy(); R = A.groupby("rec").agg(y=("y", "mean"), base=("base_oof", "mean")); R["res"] = R.y - R.base
T = T.set_index("rec"); T["res"] = R.res; T["y"] = R.y
rows = []
for tr, g in T[T.train].sort_values(["track", "track_pos"]).groupby("track"):
    start = None; prev_day = None
    for r, row in g.iterrows():
        hi = bool(row.high); d = row.day
        if hi:
            if start is None or (prev_day is not None and np.isfinite(d) and np.isfinite(prev_day) and d - prev_day > 3): start = d
            e = (d - start) if (np.isfinite(d) and start is not None and np.isfinite(start)) else np.nan
            rows.append(dict(rec=r, track=tr, ep_start=start, elapsed=e, res=row.res, y=row.y, lock=bool(row.lock), p2=bool(row.pass2)))
        else: start = None
        prev_day = d
H = pd.DataFrame(rows); H["ep"] = H.track.astype(str) + "_" + H.ep_start.astype(str)
U = H[~H.lock & H.res.notna() & H.elapsed.notna()].copy()
U["bin"] = pd.cut(U.elapsed, [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"])
pd.set_option("display.width", 200)
print(f"high-EC records with elapsed days: {len(U)} in {U.ep.nunique()} episodes (lock excluded)")
print(U.groupby("bin").agg(n=("rec", "size"), episodes=("ep", "nunique"), mean_res=("res", "mean"), median_res=("res", "median"), mean_y=("y", "mean")).round(3).to_string())
print("Spearman elapsed vs residual:", round(float(U[["elapsed", "res"]].corr(method="spearman").iloc[0, 1]), 3))
err0, err1 = [], []
for ep, g in U.groupby("ep"):
    rest = U[U.ep != ep]; m = rest.groupby("bin").res.mean()
    for b, r_ in zip(g.bin, g.res): err0.append(r_); err1.append(r_ - m.get(b, 0.0))
print(f"leave-one-episode-out: residual RMSE no correction {np.sqrt(np.mean(np.square(err0))):.3f} -> elapsed-bin correction {np.sqrt(np.mean(np.square(err1))):.3f}")
for r in ("F13_217", "F13_216", "F47_145", "F47_215"):
    if r in T.index:
        tr = T.at[r, "track"]; g = T[T.track == tr].sort_values("track_pos")[["track_pos", "day", "seg", "high", "y", "res", "gap_days", "link_type"]]
        print(f"\ntrack of {r} (track {tr}):"); print(g.round(3).to_string())
H.to_csv(os.path.join(HERE, "el1e_episodes_v1.csv"), index=False)
