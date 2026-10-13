# -*- coding: utf-8 -*-
"""P2T2 (2026-10-14 집 클로드; 사용자: '2차 날하고 연결된 1차까지만 포함해보는 건?').  사전 고정(커밋 후 실행).  EC.  P2T1(6.847) 과 같은 대리 모델·폴드·판정.
P2T1: 2차만 학습 +48%(학습 33일), 가중 4배 무효.  여기선 2차 날 + '2차와 연결된 1차 날' 만 학습.
연결된 1차 날(학습 자료만으로 정의): (1) cert 연결(truth.pkl, 학습 정답 연속)로 2차 날의 전날 또는 다음날인 1차 날;  (2) 학습 달력 DMT3e 로 어떤 2차 날과 같은 구간·날짜 차 <= 2 인 1차 날.
안(2개, Bonferroni .0125): P2C1 = 2차 + (1);  P2C2 = 2차 + ((1) 또는 (2)).  기준 ALL = 전체 학습.  계절 지표는 세 안 모두 같은 폴드 전체 학습 날로(P2T1 수정과 같음).
판정: 3시드 모두 2차 검증 RMSE 개선 AND 폴드 부트스트랩 P(악화) < .0125.
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
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import FARM as _F, NUM as _N, N as _NN, S3 as _S3  # noqa: E402
_rid = lambda i: f"{_F[i]}_{_N[i]:03d}"
TL = pd.read_pickle(os.path.join(_S3, "truth.pkl")); CERTP = [(_rid(int(a)), _rid(int(b))) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c]
key = lambda f, d: f"{f}_{int(d):03d}"
p2set = {key(f, d) for f, d in zip(p2days.farm, p2days.day)}
C1 = {a for a, b in CERTP if b in p2set and int(a[4:]) < 179} | {b for a, b in CERTP if a in p2set and int(b[4:]) < 179}
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s_, d_) for r, s_, d_ in zip(CAL.rec, CAL.seg, CAL.day)}
p2dates = [DAY[r] for r in p2set if r in DAY]
C2 = {r for r, (s_, d_) in DAY.items() if int(r[4:]) < 179 and any(s_ == s2 and abs(d_ - d2) <= 2 for s2, d2 in p2dates)}
print("connected pass-1 days: cert", len(C1), "| calendar +-2", len(C2), "| union", len(C1 | C2), flush=True)
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
    kk = np.array([key(f, d) for f, d in zip(tr_all.farm, tr_all.day)]); p2m = (tr_all.day >= 179).values
    trs = {"ALL": tr_all, "P2C1": tr_all[p2m | np.isin(kk, list(C1))].copy(), "P2C2": tr_all[p2m | np.isin(kk, list(C1 | C2))].copy()}
    # [실행 오류 수정, 결과 전·기준 불변] 계절 지표는 1차 날을 기준점으로 써서 2차만으로는 못 만듦 → 세 안 모두 같은 폴드의 전체 학습 날로 계절 지표를 만들고 학습 행만 다르게
    vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tr_all[["farm", "day"]].drop_duplicates(), vdays, wv)
    for arm, tr in trs.items():
        tdays = tr[["farm", "day"]].drop_duplicates()
        tr = tr.copy(); tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        v2 = va.copy().merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
        for sd in (47, 1414, 6464):
            p = p3.final(core.predict_model(core.et(sd), tr, v2, FS), tr, v2)
            out.append(pd.DataFrame(dict(fold=fi, arm=arm, seed=sd, farm=va.farm.values, day=va.day.values, y=va.sub_ec.values, p=p, ntrain_days=len(tdays))))
    print("fold", fi, "done", flush=True)
D = pd.concat(out, ignore_index=True); D["hi"] = D.groupby(["fold", "arm", "seed", "farm", "day"]).y.transform("mean") >= 1.0
D.to_csv(os.path.join(HERE, "p2t2_preds_v1.csv"), index=False)
pd.set_option("display.width", 200)
T = D.groupby(["arm", "seed"]).apply(lambda g: pd.Series(dict(rmse=rm(g.p - g.y), normal=rm((g.p - g.y)[~g.hi]), high=rm((g.p - g.y)[g.hi]), train_days=g.drop_duplicates("fold").ntrain_days.mean()))).reset_index()
print(T.round(4).to_string(index=False))
base = D[D.arm == "ALL"].set_index(["fold", "seed", "farm", "day"]).sort_index()
for arm in ("P2C1", "P2C2"):
    c = D[D.arm == arm].set_index(["fold", "seed", "farm", "day"]).sort_index()
    pct = [100 * (rm(c.xs(sd, level="seed").p - c.xs(sd, level="seed").y) / rm(base.xs(sd, level="seed").p - base.xs(sd, level="seed").y) - 1) for sd in (47, 1414, 6464)]
    d = ((c.p - c.y) ** 2 - (base.p.values - base.y.values) ** 2).groupby(level=["fold", "seed"]).sum().groupby(level="fold").mean().values
    ix = rng.integers(0, len(d), (20000, len(d))); P = float((d[ix].sum(1) >= 0).mean())
    ok = all(x < 0 for x in pct) and P < .0125
    print(f"\n== P2T2 {arm} vs ALL: seeds {[round(x, 2) for x in pct]}% | fold-bootstrap P(worse) {P:.4f} | folds better {int((d < 0).sum())}/{len(d)} -> {'ADOPT' if ok else 'FAIL'}")
