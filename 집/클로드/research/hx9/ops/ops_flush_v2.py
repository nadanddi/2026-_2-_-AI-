# -*- coding: utf-8 -*-
"""HX9-OPS P5 v2 (2026-10-13 집 클로드): v1 에서 상승 계단에 붙은 상위 3개(fog_on Bonf .042, d_in_temp .126, d_act_co2 .364)가
'그 기록(날) 특성'인지 '계단 시각 특성'인지 가르는 기록 안 비교.
사전 예측: 관리 사건 흔적이라면 같은 기록 안에서도 계단 시각에 포그 켜짐·CO2 변화가 다른 낮 시각보다 많아야 한다.
  기록 특성(따뜻한 늦은 사슬에서 포그를 쓰는 날일 뿐)이라면 기록 안 비교에서는 차이가 사라진다.
방법: 상승 계단이 있는 고EC 기록만, 9~17시 칸, 기록 안에서 사건 표지를 섞는 순열 2000회. 시험 3개(Bonferroni x3).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(11)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")).sort_values(["rec", "hour"])
H = HO[HO.rec.isin(R.index[R.hi10 == 1])].copy()
for c in ["sub_ec", "in_temp", "act_co2"]:
    H["d_" + c] = H.groupby("rec")[c].diff()
H["fog_on"] = (H.act_fog > 0).astype(float)
H = H[(H.hour >= 9) & (H.hour <= 17)].copy()
H["ev"] = (H.d_sub_ec >= 0.20).astype(int)
H = H[H.rec.isin(H[H.ev == 1].rec.unique())]
print("records", H.rec.nunique(), "daytime cells", len(H), "rise events", H.ev.sum())
rows = []
for c in ["fog_on", "d_in_temp", "d_act_co2"]:
    def stat(e):
        d = []
        for r, g in H.assign(e=e).groupby("rec"):
            if g.e.sum() and (g.e == 0).sum(): d.append(g[c][g.e == 1].mean() - g[c][g.e == 0].mean())
        return np.mean(d)
    obs = stat(H.ev.values); null = []
    recs = H.rec.values
    for _ in range(2000):
        e = H.ev.values.copy()
        for r in np.unique(recs):
            k = np.where(recs == r)[0]; e[k] = rng.permutation(e[k])
        null.append(stat(e))
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / 2001
    rows.append(dict(var=c, within_record_diff=obs, p_perm=p, bonf3=min(p * 3, 1)))
    print(c, "fog records:" if c == "fog_on" else "", H.groupby("rec").fog_on.max().sum() if c == "fog_on" else "")
T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(HERE, "flush_v2.csv"), index=False, encoding="utf-8")
# which records carry fog at all (record-level property?)
fr = HO[HO.rec.isin(R.index[R.hi10 == 1])].groupby("rec").act_fog.apply(lambda s: (s > 0).sum())
print("high records with any fog:", fr[fr > 0].to_dict())
print("ep_id of those:", R.loc[fr[fr > 0].index, "ep_id"].to_dict())
