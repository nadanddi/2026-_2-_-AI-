# -*- coding: utf-8 -*-
"""CC1 (2026-10-13 집 클로드; 사용자: '이 인과 구조를 모델에 반영해봐').  사전 고정(커밋 후 실행).  EC.
인과 구조(6.793): EC ← 양액 관리(숨은 변수: 동 운영 유형·고EC 에피소드) ; 환경 입력은 계절·증산 경로로 간접.  모델은 '춥고 밀폐 = 고EC' 상관을 배움.
반영: '동 운영 유형' 을 입력으로 — 대상 0..h 운영값의 '같은 날 다른 동 대비 편차'(같은 날 = 0시 외기 4칸 일치 학습 기록, 보이는 것만; FP1 6.425 원리)
  + 절대 운영값 + 계절·밀폐 지표.  (고EC 에피소드 = 전날 EC 는 앵커 계열이라 로컬 판정 무효(6.788) → 넣지 않음.)
특징(h 시 행, 인과): 대상 0..h 누적평균 act_co2·act_heating·act_thermal·act_vent·act_circfan·in_co2·in_temp, 같은 날 학습 기록들의 같은 시각 누적평균과의 차(6개),
  out_temp 누적평균, in_temp − out_temp, hour, sg 누적평균(cm).  목표 = y_h − sg_h.
보정기: LightGBM(100그루, lr .03, 잎 15, min_child 100, subsample .8, colsample .8, seed 1) — TC1c 와 같은 강한 정칙화.  pred = sg + .5·보정.
학습: 각 검증 칸에서 '대상과 같은 폴드가 아닌' 기록 행으로 폴드별 교차 적합(같은 폴드 = 숨김 기록 집합).  같은 날 학습 기록 평균은 그 폴드에서 보이는 기록만.
판정(기준선 NONE = sg, 12칸 rv1 DIAG10·P2LOO·EL1 × 47/1414/6464 + EV × 7/101/2024): 모든 칸 2차 행 개선 AND ((DIAG10·EV 2차 기록 부트스트랩 P < .025) 또는 (2차 손해 기록 0)) → ADOPT.
서술: 전체 행, 가짜 고EC 6기록(6.793)·진짜 고EC 변화, 평가 적용(12회차 EC 위) 바뀌는 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261120)
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
OPS = ["act_co2", "act_heating", "act_thermal", "act_vent", "act_circfan", "in_co2"]; ABS = OPS + ["in_temp"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CUM = {v: _X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).T.expanding().mean().T.values for v in ABS + ["out_temp"]}
TW = {}
for i in range(N):
    m = np.where(TRAIN & np.all(np.abs(OUTX[:, 0] - OUTX[i, 0]) < 1e-6, axis=1))[0]; TW[i] = m[m != i]
FEATS = [f"c_{v}" for v in ABS] + [f"d_{v}" for v in OPS] + ["c_out_temp", "in_out", "hour", "cm"]
def feats(rec_arr, hours, cm, hidden):
    rows = np.zeros((len(rec_arr), len(FEATS)))
    for k, (r, h) in enumerate(zip(rec_arr, hours)):
        i = idx[r]; tw = [j for j in TW[i] if rid(j) not in hidden]
        f = [CUM[v][i, h] for v in ABS] + [CUM[v][i, h] - (np.nanmean([CUM[v][j, h] for j in tw]) if tw else np.nan) for v in OPS]
        f += [CUM["out_temp"][i, h], CUM["in_temp"][i, h] - CUM["out_temp"][i, h], h, cm[k]]
        rows[k] = f
    return rows
PRM = dict(n_estimators=100, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True)
res, rows, fh = [], [], []
FH = ["F47_161", "F13_156", "F13_233", "F47_130", "F13_231", "F13_130"]
def run(Gv, y, sg, p2, recs, hours, folds, hidden_of_fold, tag, sd):
    cm = pd.Series(sg).groupby(recs).transform(lambda s: s.expanding().mean()).values
    new = sg.copy()
    for k in np.unique(folds):
        te = folds == k; hs = hidden_of_fold(k)
        trm = ~te
        Ftr = feats(recs[trm], hours[trm], cm[trm], hs); Fte = feats(recs[te], hours[te], cm[te], hs)
        mdl = lgb.LGBMRegressor(**PRM).fit(Ftr, (y - sg)[trm]); new[te] = np.clip(sg[te] + .5 * mdl.predict(Fte), lo_, hi_)
    res.append(dict(val=tag, seed=str(sd), p2_pct=100 * (rm((new - y)[p2]) / rm((sg - y)[p2]) - 1), all_pct=100 * (rm(new - y) / rm(sg - y) - 1)))
    ds = pd.Series(((new - y) ** 2 - (sg - y) ** 2)[p2]).groupby(recs[p2]).sum()
    for r_, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), rec=r_, dsse=d))
    D = pd.DataFrame(dict(rec=recs, sg=sg, new=new, y=y)).groupby("rec").mean()
    for r_ in D.index.intersection(FH + ["F47_136", "F47_141", "F13_217", "F13_243", "F47_231"]): fh.append(dict(val=tag, seed=str(sd), rec=r_, sg=D.at[r_, "sg"], new=D.at[r_, "new"], y=D.at[r_, "y"]))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); hidden = Gv.groupby("fold").rec.apply(set).to_dict()
    run(Gv, Gv.sub_ec.values, Gv.sg.values, Gv.day.values >= 179, Gv.rec.values, Gv.hour.values.astype(int), Gv.fold.values, lambda k: hidden.get(k, set()), v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True)
    run(Gv, Gv.y.values, Gv.SG2.values, Gv.day.values >= 179, Gv.rec.values, Gv.hour.values.astype(int), Gv.fold.values.astype(int), lambda k: hid[int(k)], "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); Dd = pd.DataFrame(rows); H = pd.DataFrame(fh); pd.set_option("display.width", 220)
print("pass-2 RMSE % change vs current EC (sg):"); print(S.pivot_table(index="val", columns="seed", values="p2_pct").round(2).to_string())
print("all rows:"); print(S.groupby("val").all_pct.mean().round(2).to_string())
boot = lambda d: float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean())
c1 = bool((S.p2_pct < 0).all()); Pd = boot(Dd[Dd.val == "DIAG10"].groupby("rec").dsse.mean().values); Pe = boot(Dd[Dd.val == "EV"].groupby("rec").dsse.mean().values)
harm = Dd[Dd.dsse > 1e-6].rec.nunique(); g = Dd.groupby("rec").dsse.mean()
ok = c1 and ((Pd < .025 and Pe < .025) or harm == 0)
print(f"\nCC1: all 12 cells better {c1} | P DIAG10 {Pd:.4f} EV {Pe:.4f} | harmed {harm} | changed {int((g.abs() > 1e-6).sum())} -> {'ADOPT' if ok else 'FAIL'}")
print("top gains:", (-g).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-g).sort_values().head(6).round(3).to_dict())
print("\nfalse-high / true-high records (seed-mean day level):"); print(H.groupby(["rec", "val"])[["sg", "new", "y"]].mean().round(3).to_string())
