# -*- coding: utf-8 -*-
"""IV1 (2026-10-14 집 클로드; 사용자: '날마다 변하는 값에서 변하지 않는 값 만들기' — IEEE-CIS 'day − D1' 착안).  서술·탐색(판정 아님).
동마다 고정된 특성(센서·관리 설정)을 입력에서 만들고, 링크 판별력을 잼.
정답 쌍 = truth.pkl cert 연결(a → b, 학습 정답 연속만).  경쟁 후보 = a 와 같은 학습 달력(DMT3e) 날짜·구간의 다른 기록(학습·평가 모두 — 입력만 씀).
지표: 쌍마다 '진짜 전날의 |f(b) − f(a)| 가 경쟁 후보보다 작은 비율'(순위 정확도, .5 = 무작위), 그리고 v15R 1순위가 틀린 쌍에서의 순위 정확도(추가 정보인가).
후보 특징(하루 전체 — 발견용 상한; 쓸 때는 0..h 판 필요):
  센서: in_hum 최대·99% 분위·고유 소수 자릿수, in_co2 0~5시 최소·중앙, in_co2 값 간격(최빈 차), in_temp 해상도
  설정: 난방 켜진 시각들의 in_temp 최소·중앙(난방 설정), 환기 처음 열린 시각의 in_temp(환기 설정), 보온커튼 닫힘/열림 시각, 차광 닫힘 시각, CO₂ 공급 첫·끝 시각, 유동팬 가동률
  단열: 0~5시 (in_temp − out_temp) 중앙(난방 꺼진 시각만), 0~5시 (in_hum − out_hum)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); KEY = {r: (s, d) for r, s, d in zip(C.rec, C.seg, C.day)}
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
P = {v: _X.pivot(index="rec", columns="h", values=v).reindex(columns=range(24)) for v in ["in_hum", "in_co2", "in_temp", "out_temp", "out_hum", "act_heating", "act_vent", "act_thermal", "act_shade", "act_co2", "act_circfan"]}
def first_hour(row, cond):
    w = np.where(cond(row))[0]; return float(w[0]) if len(w) else np.nan
def last_hour(row, cond):
    w = np.where(cond(row))[0]; return float(w[-1]) if len(w) else np.nan
def ndec(x):
    x = x[np.isfinite(x)]; return float(np.mean([len(f"{v:.6f}".rstrip("0").split(".")[1]) if "." in f"{v:.6f}".rstrip("0") else 0 for v in x])) if len(x) else np.nan
def step(x):
    x = np.unique(np.round(x[np.isfinite(x)], 4)); d = np.diff(x); d = d[d > 0]
    return float(pd.Series(np.round(d, 4)).mode().iloc[0]) if len(d) else np.nan
F = {}
for r in P["in_hum"].index:
    g = {v: P[v].loc[r].values.astype(float) for v in P}; heat = g["act_heating"] > 0; night = np.arange(24) < 6
    F[r] = dict(hum_max=np.nanmax(g["in_hum"]), hum_q99=np.nanquantile(g["in_hum"], .99), hum_ndec=ndec(g["in_hum"]), co2_night_min=np.nanmin(g["in_co2"][night]),
                co2_night_med=np.nanmedian(g["in_co2"][night]), co2_step=step(g["in_co2"]), temp_ndec=ndec(g["in_temp"]),
                heat_set_min=np.nanmin(g["in_temp"][heat]) if heat.any() else np.nan, heat_set_med=np.nanmedian(g["in_temp"][heat]) if heat.any() else np.nan,
                vent_open_temp=(g["in_temp"][int(first_hour(g["act_vent"], lambda x: x > 0))] if np.isfinite(first_hour(g["act_vent"], lambda x: x > 0)) else np.nan),
                thermal_open_h=first_hour(g["act_thermal"], lambda x: x < 50), thermal_close_h=last_hour(g["act_thermal"], lambda x: x < 50),
                shade_close_h=first_hour(g["act_shade"], lambda x: x > 50), co2_first_h=first_hour(g["act_co2"], lambda x: x > 0), co2_last_h=last_hour(g["act_co2"], lambda x: x > 0),
                fan_duty=np.nanmean(g["act_circfan"] > 0), insul_night=np.nanmedian((g["in_temp"] - g["out_temp"])[night & ~heat]) if (night & ~heat).any() else np.nan,
                hum_gap_night=np.nanmedian((g["in_hum"] - g["out_hum"])[night]))
F = pd.DataFrame(F).T; FEATS = list(F.columns)
byday = {}
for r, k in KEY.items(): byday.setdefault(k, []).append(r)
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv")); PICK = dict(zip(PE.rec, PE.prev0))
rows = []
for b, a in CERT.items():
    if a not in KEY or a not in F.index or b not in F.index: continue
    rivals = [c for c in byday[KEY[a]] if c not in (a, b) and c in F.index]
    if not rivals: continue
    out = dict(b=b, a=a, v15_wrong=PICK.get(b) != a, nriv=len(rivals))
    for f in FEATS:
        dt = abs(F.at[b, f] - F.at[a, f]); dr = np.array([abs(F.at[b, f] - F.at[c, f]) for c in rivals])
        ok = np.isfinite(dr)
        out[f] = np.nan if (not np.isfinite(dt) or not ok.any()) else float(np.mean((dt < dr[ok]) + .5 * (dt == dr[ok])))
    rows.append(out)
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(f"cert pairs with same-date rivals: {len(R)} | of which v15R top pick wrong: {int(R.v15_wrong.sum())}")
S = pd.DataFrame({"rank_acc_all": R[FEATS].mean(), "rank_acc_v15wrong": R[R.v15_wrong][FEATS].mean(), "n": R[FEATS].notna().sum(), "share_unique_vals": [F[f].nunique() / F[f].notna().sum() for f in FEATS]}).sort_values("rank_acc_all", ascending=False)
print(S.round(3).to_string())
F.to_csv(os.path.join(HERE, "iv1_record_features_v1.csv")); R.to_csv(os.path.join(HERE, "iv1_pair_rank_v1.csv"), index=False)
