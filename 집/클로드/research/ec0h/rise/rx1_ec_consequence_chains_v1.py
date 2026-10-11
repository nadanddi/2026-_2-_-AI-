# -*- coding: utf-8 -*-
"""RX1 (2026-10-12 집 클로드; 사용자: "EC를 공급한 게 무슨 작용을 해서 천창을 열어야 했다거나 그런 것도 찾아봐. 이런 느낌의 현상들도").
서술 포렌식(학습 입력+라벨, F13/F47).  '원인 사건' 뒤 h+0..h+4 의 반응을 같은 농장·같은 시각·비슷한 외기(외기온·일사)·
같은 직전 장치 상태(천창·차광·보온·난방·팬 h−1 값) kNN 15개 대조와 비교(차이의 차이).  직전 추세(h−3→h−1)도 같이 봐서
'사건 전부터 이미 움직였는지'(= 사건이 결과가 아니라 원인의 결과인지) 구분.  기록 부트스트랩 95% 구간.
원인 사건(시각 2..20, 이웃 3시간 안 같은 종류 중복 제거):
  E1 EC25 급등 dEC25 >= +.05/h      E2 EC25 급락 <= −.05/h
  E3 배지온 급락(관수 흔적 후보: 찬 물) dST <= −.5 그리고 실내온 dIn >= −.2 (공기는 안 식는데 배지만 식음)
  E4 배지온 급등 dST >= +.8 그리고 dIn <= +.3
  E5 고EC 상태(그날 0시 EC25 >= 1.2) 의 낮 시각 vs 같은 시각·비슷한 외기의 보통 상태(0시 < .8) — 상태 비교(사건 아님)
반응 변수: act_vent, act_side, act_shade, act_thermal, act_heating, act_circfan, act_co2, act_fog, in_temp, in_hum, in_co2, sub_temp, EC25.
사전 판독(정성): 'EC 공급 → 천창'류 연쇄라면 E1/E2/E3 뒤 천창·습도 반응이 대조와 다르고(구간 0 제외) 직전 추세엔 차이 없음.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rng = np.random.default_rng(20261215)
Xr = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Yr = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
D = Xr.merge(Yr, on="row_id"); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
D["rec"] = D.row_id.str[:7]; D["h"] = D.row_id.str[8:].astype(int); D["farm"] = D.rec.str[:3]
D = D.sort_values(["rec", "h"]).reset_index(drop=True)
D["EC25"] = D.sub_ec / (1 + .02 * (D.sub_temp - 25))
RESP = ["act_vent", "act_side", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog", "in_temp", "in_hum", "in_co2", "sub_temp", "EC25"]
RESP = [v for v in RESP if D[v].notna().mean() > .5]
print("response vars with data:", RESP)
recs = D.rec.unique(); R = {v: D.pivot(index="rec", columns="h", values=v).reindex(recs) for v in set(RESP + ["out_temp", "out_rad"])}
farm = pd.Series([r[:3] for r in recs], index=recs)
lvl0 = R["EC25"][0]
dE = R["EC25"].diff(axis=1); dS = R["sub_temp"].diff(axis=1); dI = R["in_temp"].diff(axis=1)
def dedup(mask):
    out = []
    for r in recs:
        last = -9
        for h in range(2, 21):
            if mask.at[r, h] and h - last > 3: out.append((r, h)); last = h
    return out
EV = {"E1 EC25 up step": dedup(dE >= .05), "E2 EC25 down step": dedup(dE <= -.05),
      "E3 substrate cools, air not (irrigation?)": dedup((dS <= -.5) & (dI >= -.2)),
      "E4 substrate warms, air not": dedup((dS >= .8) & (dI <= .3))}
ACTS = [v for v in ("act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan") if v in R]
def ctrl_pool(r, h, evset):
    # candidates: same farm, same hour, not an event within +-2h
    c = [q for q in recs if farm[q] == farm[r] and q != r and (q, h) not in evset]
    if not c: return []
    f = np.array([[R["out_temp"].at[q, h], R["out_rad"].at[q, h]] + [R[a].at[q, h - 1] for a in ACTS] for q in c], float)
    t = np.array([R["out_temp"].at[r, h], R["out_rad"].at[r, h]] + [R[a].at[r, h - 1] for a in ACTS], float)
    sd = np.nanstd(f, 0) + 1e-6; d = np.nansum(np.abs((f - t) / sd), 1)
    return [c[k] for k in np.argsort(d)[:15]]
def effect(evlist, name):
    evset = set(evlist); rows = []
    for r, h in evlist:
        C = ctrl_pool(r, h, evset)
        if len(C) < 5: continue
        row = dict(rec=r, h=h)
        for v in RESP:
            base = R[v].at[r, h - 1]; cb = R[v].loc[C, h - 1]
            row[f"{v}|pre"] = (R[v].at[r, h - 1] - R[v].at[r, max(h - 3, 0)]) - (cb - R[v].loc[C, max(h - 3, 0)]).mean()
            for k in (0, 1, 2, 4):
                hh = min(h + k, 23); row[f"{v}|+{k}"] = (R[v].at[r, hh] - base) - (R[v].loc[C, hh] - cb).mean()
        rows.append(row)
    T = pd.DataFrame(rows); print(f"\n===== {name}: events {len(T)} (pass-2 {int((T.rec.str[4:7].astype(int) >= 179).sum())}) | hours {T.h.value_counts().sort_index().to_dict()}")
    g = T.groupby("rec"); keys = list(g.groups); idx = [g.groups[k] for k in keys]
    for v in RESP:
        out = []
        for col in [f"{v}|pre", f"{v}|+0", f"{v}|+1", f"{v}|+2", f"{v}|+4"]:
            m = T[col].mean(); bs = []
            for _ in range(500):
                s = rng.integers(0, len(keys), len(keys)); bs.append(np.nanmean(np.concatenate([T.loc[idx[j], col].values for j in s])))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]); out.append(f"{col.split('|')[1]} {m:+.2f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"  {v:12s}", " | ".join(out))
    return T
res = {k: effect(v, k) for k, v in EV.items()}
# E5 state comparison: high-EC days vs normal days, same farm, hour-matched with similar outdoor temp/rad (nearest 15 normal days)
print("\n===== E5 high-EC state (0h EC25>=1.2) minus matched normal days (0h<.8), same farm/hour/outdoor =====")
hi = [r for r in recs if lvl0[r] >= 1.2]; no = [r for r in recs if lvl0[r] < .8]
print("high-EC records", len(hi))
for v in RESP:
    out = []
    for h in (2, 6, 9, 12, 15, 18, 21):
        dd = []
        for r in hi:
            C = [q for q in no if farm[q] == farm[r]]
            f = np.array([[R["out_temp"].at[q, h], R["out_rad"].at[q, h], np.nanmean(R["out_temp"].loc[q])] for q in C]); t = np.array([R["out_temp"].at[r, h], R["out_rad"].at[r, h], np.nanmean(R["out_temp"].loc[r])])
            d = np.nansum(np.abs((f - t) / (np.nanstd(f, 0) + 1e-6)), 1); C = [C[k] for k in np.argsort(d)[:15]]
            dd.append(R[v].at[r, h] - R[v].loc[C, h].mean())
        dd = np.array(dd); bs = [np.nanmean(rng.choice(dd, len(dd))) for _ in range(500)]; lo, hi_ = np.nanpercentile(bs, [2.5, 97.5])
        out.append(f"{h}h {np.nanmean(dd):+.1f}{'*' if lo > 0 or hi_ < 0 else ' '}")
    print(f"  {v:12s}", " ".join(out))
for k, T in res.items(): T.to_csv(f"rx1_{k.split()[0]}_v1.csv", index=False)
