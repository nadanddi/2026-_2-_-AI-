# -*- coding: utf-8 -*-
"""FH1 (2026-10-13 집 클로드; 사용자: '입력이 고EC 날과 비슷해 보이는데 실제로 보통인 사례를 정리하고 인과관계 파악'). 서술.
자료: rv1 DIAG10 OOF(시드 3개 평균) 하루 수준.  분류: FH(가짜 고EC) = 예측 하루 평균 ≥ .95 & 실제 < .85;  TH(진짜 고EC) = 예측 ≥ .95 & 실제 ≥ 1.2;
  NM(보통) = 예측 < .75 & 실제 < .75.  각 기록의 입력 하루 요약(평균·낮/밤), 맥락(같은 날 고EC 쌍둥이, 진짜 전날·다음날 EC, 고EC 사슬과의 거리)을 표로.
비교: FH vs TH(무엇이 다른가 = 진짜를 가르는 단서), FH vs NM(무엇이 모델을 속였나) — 표준화 평균차.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Dd = O.groupby(["rec", "hour"]).agg(sg=("sg", "mean"), y=("sub_ec", "first")).reset_index()
R = Dd.groupby("rec").agg(pred=("sg", "mean"), y=("y", "mean"), pred0=("sg", "first")).reset_index()
R["cls"] = np.select([(R.pred >= .95) & (R.y < .85), (R.pred >= .95) & (R.y >= 1.2), (R.pred < .75) & (R.y < .75)], ["FH", "TH", "NM"], "other")
_X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
day = _X[(_X.h >= 9) & (_X.h <= 16)].groupby("rec")[V].mean().add_suffix("_d"); night = _X[(_X.h <= 5) | (_X.h >= 20)].groupby("rec")[V].mean().add_suffix("_n")
F = day.join(night); F["dT_in_out_d"] = F.in_temp_d - F.out_temp_d; F["vpd_d"] = 0.6108 * np.exp(17.27 * F.in_temp_d / (F.in_temp_d + 237.3)) * (1 - F.in_hum_d / 100)
F["co2_drawdown"] = F.in_co2_n - F.in_co2_d
R = R.merge(F, left_on="rec", right_index=True, how="left")
# context
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
key = {}
for i in range(N): key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
PREV = {}; NEXT = {}
for a, b in zip(TL.a1, TL.b):
    a, b = int(a), int(b)
    if D11.get(rid(b), np.nan) - D11.get(rid(a), np.nan) == 1: PREV[b] = a; NEXT.setdefault(a, b)
def ctx(r):
    i = idx[r]; tw = [j for j in key[np.round(X[i], 1).tobytes()] if j != i and TRAIN[j]]
    p = PREV.get(i); n = NEXT.get(i)
    return pd.Series(dict(twin_hi=any(DM[j] >= 1.2 for j in tw), twin_max=max([DM[j] for j in tw], default=np.nan),
                          prev_ec=DM[p] if p is not None else np.nan, next_ec=DM[n] if n is not None else np.nan, p2=NUM[i] >= 179))
R = R.join(R.rec.apply(ctx))
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
print("class counts:", R.cls.value_counts().to_dict())
cols = ["rec", "p2", "pred", "pred0", "y", "prev_ec", "next_ec", "twin_hi", "twin_max", "in_temp_d", "in_hum_d", "in_co2_d", "act_vent_d", "act_shade_d", "act_thermal_n", "act_heating_n", "act_co2_d", "act_fog_d", "out_rad_d"]
print("\n== FALSE-HIGH cases (pred >= .95, actual < .85):"); print(R[R.cls == "FH"].sort_values("pred", ascending=False)[cols].round(2).to_string(index=False))
print("\n== TRUE-HIGH (for contrast, top 15 by pred):"); print(R[R.cls == "TH"].sort_values("pred", ascending=False)[cols].head(15).round(2).to_string(index=False))
FE = [c for c in F.columns]
sd = R[FE].std()
smd = pd.DataFrame({"FH_vs_NM": (R[R.cls == "FH"][FE].mean() - R[R.cls == "NM"][FE].mean()) / sd, "TH_vs_NM": (R[R.cls == "TH"][FE].mean() - R[R.cls == "NM"][FE].mean()) / sd,
                    "FH_vs_TH": (R[R.cls == "FH"][FE].mean() - R[R.cls == "TH"][FE].mean()) / sd})
smd["looks_high"] = np.sign(smd.FH_vs_NM) == np.sign(smd.TH_vs_NM)
print("\n== standardized mean differences (top by |TH_vs_NM|): which inputs make a day 'look high' and do FH share them?")
print(smd.reindex(smd.TH_vs_NM.abs().sort_values(ascending=False).index).head(18).round(2).to_string())
print("\n== inputs that separate FH from TH (top |FH_vs_TH|):"); print(smd.reindex(smd.FH_vs_TH.abs().sort_values(ascending=False).index).head(10).round(2).to_string())
print("\n== context: share with same-date high twin | prev-day EC known & high(>=1.0) | next-day EC high")
for c in ("FH", "TH", "NM"):
    g = R[R.cls == c]; print(f"  {c}: n {len(g)} | twin_hi {g.twin_hi.mean():.2f} | prev known {g.prev_ec.notna().mean():.2f}, prev high {(g.prev_ec >= 1.0).mean():.2f} | next high {(g.next_ec >= 1.0).mean():.2f} | pass-2 {g.p2.mean():.2f}")
R.to_csv(os.path.join(env.LOCAL, "fh1_cases.csv"), index=False)
