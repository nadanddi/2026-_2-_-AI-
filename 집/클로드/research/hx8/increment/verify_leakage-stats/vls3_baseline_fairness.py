# -*- coding: utf-8 -*-
"""verify_leakage-stats vls3: 비교 기준 M(ec_rv1 CUR 'sg' = 모델 + SG2 후처리) 공정성.
SG2 는 같은 farm 의 ±3일 정답 기록(검증 폴드 밖)을 참조해 2차 행을 고침 -> 가짜 사례에서 숨겨야 할 이웃(앵커·사이 날)이
M 쪽에는 들어갔을 수 있음. 점검: (a) 대상과 앵커/사이 날의 DIAG10 폴드가 다른 비율, (b) nosg vs sg 차이, (c) M=nosg 로 바꿨을 때 결과.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
import numpy as np, pandas as pd
inc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
print(O.columns.tolist())
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
fold = O.groupby("rec").fold.first(); day = O.groupby("rec").day.first() if "day" in O else None
Msg = O.groupby("rec").sg.mean(); Mno = O.groupby("rec").nosg.mean() if "nosg" in O else None
C = pd.read_csv(os.path.join(inc, "inc2_cases_v1.csv")); S1 = C[C.m >= .9].copy()
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
S1["fa"] = S1.a.map(fold); S1["ft"] = S1.t.map(fold); S1["dt"] = S1.t.map(day) if day is not None else np.nan
S1["mno"] = S1.t.map(Mno) if Mno is not None else np.nan
print("anchor fold known:", S1.fa.notna().mean())
for (d, k), g in S1.groupby(["dir", "k"]):
    if k > 4: continue
    diff = (g.fa != g.ft) & g.fa.notna()
    print(f" {d} k{k}: n {len(g)} | anchor in other DIAG10 fold {diff.mean():.2f} | target day>=179 (2차) {np.mean(g.dt >= 179):.2f} | mean |sg-nosg| {np.mean(np.abs(g.m - g.mno)):.3f}"
          f" | RMSE sg {rm(g.m, g.yt):.3f} nosg {rm(g.mno, g.yt):.3f} | P {rm(g.P, g.yt):.3f}")
    if diff.any():
        print(f"      (anchor other fold) RMSE M {rm(g.m[diff], g.yt[diff]):.3f} vs P {rm(g.P[diff], g.yt[diff]):.3f}, n {diff.sum()} | (same fold) M {rm(g.m[~diff], g.yt[~diff]) if (~diff).any() else np.nan:.3f} vs P {rm(g.P[~diff], g.yt[~diff]) if (~diff).any() else np.nan:.3f}, n {(~diff).sum()}")
# eval 8 records: are they 2차 rows affected by SG2-type correction in the submission? (descriptive only: day numbers of eval unknown here)
print("\nDIAG10 fold sizes:", fold.value_counts().sort_index().to_dict())
# are DIAG10 folds contiguous in day?
if day is not None:
    dd = pd.DataFrame(dict(f=fold, d=day)).dropna().sort_values("d")
    runs = (dd.f != dd.f.shift()).sum(); print("number of fold runs along day order:", int(runs), "for", len(dd), "records")
