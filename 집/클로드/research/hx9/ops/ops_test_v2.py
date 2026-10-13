# -*- coding: utf-8 -*-
"""HX9-OPS 견고성 v2 (2026-10-13 집 클로드): v1 에서 Y2h 순열 p < .05 인 후보 5개의 견고성 점검 (v1 결과 본 뒤 사후 점검임을 명시).
후보: chg_inp_co2h (v1 규칙상 signal), chg_inp_heath, chg_inp_fanh, chg_inp_sealed, proto_sim_hi.
사전 고정 생존 조건 (모두 만족해야 '생존'):
  S1. 잔차 최대 기록 F47_132(잔차 +1.47) 제외 후 |r| >= .25, 같은 부호
  S2. clean 링크판(chg_clean_*; proto 는 해당 없음)이 같은 부호, |r| >= .20  — 같은 동 전날 대비 '변화'라는 해석의 필요조건
  S3. 0~11시만으로 만든 판(평가 행 11시 이후 사용 가능)이 같은 부호
  S4. 무작위 전날(같은 농장·다른 날짜 학습 기록) 500회 위약 분포에서 관측 |r| 이 상위 5% (= 링커가 고른 전날이 특별함)
또 v1 의 순열 p 계산 결함 수정: 고EC 안에서 상수인 변수(co2_offday, th_night, chg_clean_fanh)는 p=NaN 으로 두고 BH 재계산.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(3)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
L = pd.read_csv(os.path.join(HERE, "link_input_v1.csv")).set_index("rec")
T1 = pd.read_csv(os.path.join(HERE, "test_v1_univariate.csv")).set_index("variable")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))

# ---- v1 permutation-p fix ----
T1.loc[T1.r_Y2h.isna(), ["p_perm_Y2h", "p_fwer_maxT_Y2h"]] = np.nan
ok = T1.p_perm_Y2h.notna(); ps = T1.loc[ok, "p_perm_Y2h"].values; o = np.argsort(ps); m = len(ps)
q = np.empty_like(ps); q[o] = np.minimum.accumulate((ps[o] * m / np.arange(1, m + 1))[::-1])[::-1]
T1["q_bh_Y2h_fixed"] = np.nan; T1.loc[ok, "q_bh_Y2h_fixed"] = np.minimum(q, 1)
T1.to_csv(os.path.join(HERE, "test_v2_univariate_pfix.csv"), encoding="utf-8")
print("valid permutation tests:", m, "| nominal p<.05:", int((ps < .05).sum()), "(expected by chance %.1f)" % (0.05 * m))
print(T1.loc[ok].sort_values("p_perm_Y2h")[["r_Y2h", "p_perm_Y2h", "p_fwer_maxT_Y2h", "q_bh_Y2h_fixed"]].head(8).round(3).to_string())

# ---- hourly arrays for all records (train + eval) to compute partial / placebo versions ----
V = ["act_vent", "act_heating", "act_co2", "act_circfan"]
def load(fn):
    D = pd.read_csv(os.path.join(env.DATA, fn)); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["hour"] = D.row_id.str[8:].astype(int); return D[["rec", "hour"] + V + ["out_temp", "out_hum", "out_rad", "out_wspd"]]
A = pd.concat([load("train_X.csv"), load("test_X.csv")])
ARR = {v: A.pivot(index="rec", columns="hour", values=v) for v in V + ["out_temp", "out_hum", "out_rad", "out_wspd"]}
def summ(rec, kind, hrs=slice(0, 24)):
    if kind == "co2h": return float((ARR["act_co2"].loc[rec].values[hrs] > 0).sum())
    if kind == "heath": return float((ARR["act_heating"].loc[rec].values[hrs] > 0).sum())
    if kind == "fanh": return float((ARR["act_circfan"].loc[rec].values[hrs] > 0).sum())
    if kind == "sealed": return float((ARR["act_vent"].loc[rec].values[hrs] == 0).mean())

S = R[(R.hi10 == 1) & (R.has_sg == 1)].join(F).join(L[["inp_prev"]], rsuffix="_L")
y = S.resid_mean.values; g = S.ep_id.values.astype(int)
def r_(x, yy):
    mm = np.isfinite(x) & np.isfinite(yy)
    return stats.pearsonr(x[mm], yy[mm])[0] if mm.sum() > 4 and np.std(x[mm]) > 0 else np.nan

# same-date map (outdoor 24h identical >= 70%) for placebo exclusion
O = np.stack([ARR[v].values for v in ["out_temp", "out_hum", "out_rad", "out_wspd"]], 2).reshape(len(ARR["out_temp"]), -1)
recs_all = list(ARR["out_temp"].index); RIa = {r: k for k, r in enumerate(recs_all)}
train_recs = set(R.index)
rows = []
for kind in ["co2h", "heath", "fanh", "sealed"]:
    v = f"chg_inp_{kind}"; d = dict(variable=v)
    x = S[v].values.astype(float)
    d["r_v1"] = r_(x, y); d["rho_v1"] = stats.spearmanr(x, y)[0]
    k = S.index != "F47_132"; d["r_drop_F47_132"] = r_(x[k], y[k])
    ag = (S.inp_prev == S.prev_rec).values
    d["n_agree"] = int(ag.sum()); d["r_link_agree"] = r_(x[ag], y[ag]); d["r_link_disagree"] = r_(x[~ag], y[~ag])
    d["r_clean"] = r_(S[f"chg_clean_{kind}"].values.astype(float), y)
    own = np.array([summ(r, kind) for r in S.index]); prv = np.array([summ(p, kind) for p in S.inp_prev])
    d["r_own_level"] = r_(own, y); d["r_pred_level"] = r_(prv, y)
    am = np.array([summ(r, kind, slice(0, 12)) - summ(p, kind, slice(0, 12)) for r, p in zip(S.index, S.inp_prev)])
    d["r_am_0to11"] = r_(am, y)
    # placebo: random same-farm training record from a different date
    null = []
    cand_by_farm = {f: [r for r in train_recs if r[:3] == f] for f in ["F13", "F47"]}
    for _ in range(500):
        pv = []
        for r in S.index:
            i = RIa[r]
            while True:
                c = cand_by_farm[r[:3]][rng.integers(len(cand_by_farm[r[:3]]))]
                if c != r and (np.abs(O[RIa[c]] - O[i]) < 1e-6).mean() < 0.7: break
            pv.append(summ(r, kind) - summ(c, kind))
        null.append(r_(np.array(pv), y))
    null = np.array(null)
    d["placebo_r_mean"] = np.nanmean(null); d["placebo_r_q95_abs"] = np.nanpercentile(np.abs(null), 95)
    d["placebo_p"] = (np.sum(np.abs(null) >= abs(d["r_v1"])) + 1) / (len(null) + 1)
    s = np.sign(d["r_v1"])
    d["S1"] = bool(abs(d["r_drop_F47_132"]) >= .25 and np.sign(d["r_drop_F47_132"]) == s)
    d["S2"] = bool(np.isfinite(d["r_clean"]) and np.sign(d["r_clean"]) == s and abs(d["r_clean"]) >= .20)
    d["S3"] = bool(np.sign(d["r_am_0to11"]) == s)
    d["S4"] = bool(d["placebo_p"] < .05)
    d["survive"] = d["S1"] and d["S2"] and d["S3"] and d["S4"]
    rows.append(d)
# proto_sim_hi: only S1 and S3-not-applicable; report drop + farm
x = S.proto_sim_hi.values; k = S.index != "F47_132"
rows.append(dict(variable="proto_sim_hi", r_v1=r_(x, y), rho_v1=stats.spearmanr(x, y)[0], r_drop_F47_132=r_(x[k], y[k]),
                 S1=bool(abs(r_(x[k], y[k])) >= .25 and np.sign(r_(x[k], y[k])) == np.sign(r_(x, y))), S2=np.nan, S3=np.nan, S4=np.nan,
                 survive=False))
T = pd.DataFrame(rows).set_index("variable")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.round(3).to_string())
T.to_csv(os.path.join(HERE, "test_v2_robust.csv"), encoding="utf-8")
# leave-out influence of top residual record on CV gain for chg_inp_co2h / heath
def cv_gain(x, yy, base, gg):
    pb = np.zeros_like(yy); pa = np.zeros_like(yy)
    for u in np.unique(gg):
        te = gg == u; tr = ~te
        Xb = np.column_stack([np.ones(tr.sum()), base[tr]]); Xa = np.column_stack([Xb, x[tr]])
        bb = np.linalg.lstsq(Xb, yy[tr], rcond=None)[0]; ba = np.linalg.lstsq(Xa, yy[tr], rcond=None)[0]
        pb[te] = np.column_stack([np.ones(te.sum()), base[te]]) @ bb; pa[te] = np.column_stack([np.ones(te.sum()), base[te], x[te]]) @ ba
    sst = np.sum((yy - yy.mean()) ** 2); return (np.sum((yy - pb) ** 2) - np.sum((yy - pa) ** 2)) / sst
for v in ["chg_inp_co2h", "chg_inp_heath"]:
    x = S[v].values.astype(float); b = S.sg_mean.values; k = S.index != "F47_132"
    print(v, "LOCO gain all %.3f | without F47_132 %.3f" % (cv_gain(x, y, b, g), cv_gain(x[k], y[k], b[k], g[k])))
