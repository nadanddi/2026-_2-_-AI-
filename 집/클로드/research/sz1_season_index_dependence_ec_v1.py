# -*- coding: utf-8 -*-
"""SZ1 (2026-10-14 집 클로드; 사용자: '컴퓨터 감당 가능하면 다른 것도' → 후보 ④ 계절 지표 의존 점검).  서술(판정 아님).  EC.  대리 모델.
계절 지표(dc4.season_index): 1차 = 기록 번호 자체(번호 = 시간 순서 가정), 2차 = 1차와 하루 외기가 똑같은 2차 학습 '쌍둥이'로 날짜를 잡고 사이를 번호로 보간
  → 정보(날짜·계절)는 현장에서도 있으나 얻는 방법이 자료 생성 구조(번호 순서·날씨 재사용)에 의존.
측정: P2T2 와 같은 대리 ET(DP1 + 계절, 시드 47) 로 ALL(계절 포함) vs NOS(계절 열 제거, 나머지 같음).  폴드 = 농장 x 번호 20 묶음(학습 정답 날 전체), 숨김 = 같은 농장 ±1, 잠금 ±1.
보고: 전체·2차 RMSE, 변화 %.  (기본 모델 전체 재학습은 CPU 여유 부족으로 대리로 대신)
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc4, p3, core = dp1.dc4, dp1.p3, dp1.core
raw, full, lab, lock, signatures, fds = p3.prepare()
lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
wv = dc4.weather_vectors(full)
FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW; FN = [c for c in FS if c != "season"]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
days = lab[["farm", "day"]].drop_duplicates(); folds = [set(map(tuple, g[["farm", "day"]].values)) for _, g in days.groupby([days.farm, days.day // 20])]
print("folds", len(folds), flush=True); out = []
for fi, vd in enumerate(folds):
    va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    tr = lab[np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])].copy(); va = lab[va_m].copy()
    vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tr[["farm", "day"]].drop_duplicates(), vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; va = va.merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
    pa = p3.final(core.predict_model(core.et(47), tr, va, FS), tr, va); pn = p3.final(core.predict_model(core.et(47), tr, va, FN), tr, va)
    out.append(pd.DataFrame(dict(fold=fi, farm=va.farm.values, day=va.day.values, y=va.sub_ec.values, ALL=pa, NOS=pn))); print("fold", fi, "done", flush=True)
D = pd.concat(out, ignore_index=True); p2 = D.day >= 179; hi = D.groupby(["farm", "day"]).y.transform("mean") >= 1.0
for nm, m in (("all", np.ones(len(D), bool)), ("pass-2", p2.values), ("normal", ~hi.values), ("high", hi.values)):
    a, n = rm((D.ALL - D.y)[m]), rm((D.NOS - D.y)[m]); print(f"{nm:7s}: with season {a:.4f} | without {n:.4f} ({100 * (n / a - 1):+.1f}%)")
D.to_csv(os.path.join(HERE, "sz1_preds_v1.csv.gz"), index=False)
