# -*- coding: utf-8 -*-
"""LBD1 (2026-10-13 집 클로드; 사용자: '평가 오차가 실제로 어디 있나' 진단).  서술 전용 — 제출물·계수 선택에 쓰지 않음(규칙 5 해석: 리더보드 분석 허용, 점수 맞춤 금지).
점수 항등식(1440행 RMSE 채점 가정): ||e_b||² = ||e_a||² + ||d||² + 2 d·e_a,  d = p_b − p_a.  → d·e_a 를 앎(반올림 ±.00005 범위).
  (1) 바뀐 행 집합 S 에서 옛 제출 오차 제곱합 하한: ||e_a,S||² ≥ (d·e_a)² / ||d||² (코시-슈바르츠).
  (2) 기록별로 보면: 바뀐 기록 묶음의 '오차 하한' 이 전체 SSE(= 점수² × 1440)의 몇 % 인가.
"""
import os, sys, glob, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
SUB = {8: ("08회차_2026-10-03(팀)", .1455), 9: ("09회차_2026-10-06(팀)", .1384), 10: ("10회차_2026-10-11(나)", .3538),
       11: ("11회차_2026-10-11(나)", .1305), 12: ("12회차_2026-10-11(나)", .1271), 13: ("13회차_2026-10-12(나)", .1279)}
P = {}
for k, (d, s) in SUB.items():
    f = [p for p in glob.glob(os.path.join(env.ROOT, "제출", d, "*.csv")) if os.path.basename(p).startswith("submission")][0]
    x = pd.read_csv(f).sort_values("row_id").reset_index(drop=True); P[k] = x
rows = P[9].row_id.values; rec = pd.Series(rows).str[:7].values; n = len(rows); print("rows", n)
for k in P: assert (P[k].row_id.values == rows).all()
pd.set_option("display.width", 250)
def band(sa, sb):
    out = []
    for da, db in itertools.product((-5e-5, 5e-5), repeat=2): out.append(((sb + db) ** 2 - (sa + da) ** 2) * n)
    return min(out), max(out)
res = []
for a, b in [(8, 9), (9, 10), (9, 11), (11, 12), (9, 12), (12, 13), (9, 13)]:
    d = P[b].sub_ec.values - P[a].sub_ec.values; S = np.abs(d) > 1e-9; dd = float(d @ d)
    lo, hi = band(SUB[a][1], SUB[b][1]); de = ((lo - dd) / 2, (hi - dd) / 2)
    lb = min(x * x for x in de) / dd if not (de[0] < 0 < de[1]) else 0.0
    tot = SUB[a][1] ** 2 * n; recs = sorted(set(rec[S]))
    corr = [x / np.sqrt(dd * tot) for x in de]
    res.append(dict(pair=f"{a}->{b}", rows=int(S.sum()), recs=len(recs), rms_d=np.sqrt(dd / n), d_dot_e=f"{de[0]:+.3f}..{de[1]:+.3f}", corr=f"{corr[0]:+.3f}..{corr[1]:+.3f}",
                    SSE_total=round(tot, 2), SSE_S_lowerbound=round(lb, 2), share_lb=round(lb / tot, 3)))
    print(f"{a}->{b} changed records:", " ".join(recs) if len(recs) <= 40 else f"{len(recs)} records")
R = pd.DataFrame(res); print(R.to_string())
# 9->12 split by pass-2 pure-window high (W2 rows) vs rest: projections not separable without more pairs -> report composition only
for a, b in [(9, 12), (12, 13)]:
    d = P[b].sub_ec.values - P[a].sub_ec.values; df = pd.DataFrame(dict(rec=rec, d=d, pa=P[a].sub_ec.values))
    g = df[np.abs(df.d) > 1e-9].groupby("rec").agg(rows=("d", "size"), mean_d=("d", "mean"), rms_d=("d", lambda s: np.sqrt(np.mean(s ** 2))), pa=("pa", "mean"))
    print(f"\n{a}->{b} per-record change:"); print(g.round(3).to_string())
