# -*- coding: utf-8 -*-
"""P2T1 (2026-10-14 집 클로드; 사용자: '2차 구간만 학습해서 평가 맞춰볼 수 있나?').  사전 고정(커밋 후 실행).  EC.  기본 모델 대리 실험.
근거: 같은 앵커가 1차에선 거의 무효·2차에선 오차 1/4(6.846) — 1차와 2차의 EC 성질이 다름.  평가 60기록은 전부 2차 번호.  EC 에서 2차만 학습은 처음(온도는 2차 가중 PW1 근소 FAIL, 6.525).
모델: R3 의 ET 구성원(core.et, DP1 운영 특징 + 계절 지표 DC4, p3.final 후처리 — AB1 과 같은 틀), 시드 47/1414/6464.  앵커·SG2·TabPFN 없음(대리).
검증: EL1 꼴 — 2차 정답 날 46개를 농장별로 번호순 5개씩 묶어 한 묶음씩 검증, 검증 날과 잠금 날의 같은 농장 ±1 날은 학습에서 뺌.
안(2개, Bonferroni .0125):  P2ONLY = 학습을 2차 날만으로;  W4 = 전체 학습 + 2차 날 행을 4번 복제(가중 4).  기준 ALL = 전체 학습.
판정: 3시드 모두 2차 검증 행 RMSE 개선 AND 묶음(폴드) 단위 부트스트랩(시드 평균 묶음별 dSSE) P(악화) < .0125.
서술: 보통/고EC 날 분해, 학습 날 수.
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
FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261206)
days = lab[["farm", "day"]].drop_duplicates(); p2days = days[days.day >= 179].sort_values(["farm", "day"])
folds = []
for f, g in p2days.groupby("farm"):
    dd = list(g.day)
    for k in range(0, len(dd), 5): folds.append({(f, int(d)) for d in dd[k:k + 5]})
print("folds", len(folds), [len(x) for x in folds], flush=True)
out = []
for fi, vd in enumerate(folds):
    va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    tr_all = lab[np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])].copy(); va = lab[va_m].copy()
    trs = {"ALL": tr_all, "P2ONLY": tr_all[tr_all.day >= 179].copy(), "W4": pd.concat([tr_all] + [tr_all[tr_all.day >= 179]] * 3)}
    for arm, tr in trs.items():
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr = tr.copy(); tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        v2 = va.copy().merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
        for sd in (47, 1414, 6464):
            p = p3.final(core.predict_model(core.et(sd), tr, v2, FS), tr, v2)
            out.append(pd.DataFrame(dict(fold=fi, arm=arm, seed=sd, farm=va.farm.values, day=va.day.values, y=va.sub_ec.values, p=p, ntrain_days=len(tdays))))
    print("fold", fi, "done", flush=True)
D = pd.concat(out, ignore_index=True); D["hi"] = D.groupby(["fold", "arm", "seed", "farm", "day"]).y.transform("mean") >= 1.0
D.to_csv(os.path.join(HERE, "p2t1_preds_v1.csv"), index=False)
pd.set_option("display.width", 200)
T = D.groupby(["arm", "seed"]).apply(lambda g: pd.Series(dict(rmse=rm(g.p - g.y), normal=rm((g.p - g.y)[~g.hi]), high=rm((g.p - g.y)[g.hi]), train_days=g.drop_duplicates("fold").ntrain_days.mean()))).reset_index()
print(T.round(4).to_string(index=False))
base = D[D.arm == "ALL"].set_index(["fold", "seed", "farm", "day"]).sort_index()
for arm in ("P2ONLY", "W4"):
    c = D[D.arm == arm].set_index(["fold", "seed", "farm", "day"]).sort_index()
    pct = [100 * (rm(c.xs(sd, level="seed").p - c.xs(sd, level="seed").y) / rm(base.xs(sd, level="seed").p - base.xs(sd, level="seed").y) - 1) for sd in (47, 1414, 6464)]
    d = ((c.p - c.y) ** 2 - (base.p.values - base.y.values) ** 2).groupby(level=["fold", "seed"]).sum().groupby(level="fold").mean().values
    ix = rng.integers(0, len(d), (20000, len(d))); P = float((d[ix].sum(1) >= 0).mean())
    ok = all(x < 0 for x in pct) and P < .0125
    print(f"\n== P2T1 {arm} vs ALL: seeds {[round(x, 2) for x in pct]}% | fold-bootstrap P(worse) {P:.4f} | folds better {int((d < 0).sum())}/{len(d)} -> {'ADOPT' if ok else 'FAIL'}")
