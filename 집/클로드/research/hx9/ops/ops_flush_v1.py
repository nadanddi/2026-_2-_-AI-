# -*- coding: utf-8 -*-
"""HX9-OPS P5 v1 (2026-10-13 집 클로드): 고EC 기록 안 한 시간 EC 계단(하락 = flush 추정, 상승 = 농축/급액 감소 추정) 시각의 구동기 흔적.
사전 예측 (ops_vars_v1.py 머리말 P5): 관리 사건이 EC 계단을 만들고 구동기에 흔적을 남긴다면, 계단 시각(h−1→h)의
  구동기 변화(포그 펄스, 환기·커튼·난방·CO2 변화)가 같은 시각대 위약(|ΔEC| < .05 인 고EC 기록 칸)보다 뚜렷해야 한다.
  보이면 안 되는 것: 차이가 외기 일사 변화(음성 대조, 관리 아님)에서도 같은 크기로 나타남 → 날씨 공통 원인.
이전 시험과 다른 점: 6.653(ST1)은 |Δ|≥.3+3시간 유지 2건만 서술, 6.659(RX1)는 사건 '뒤' h+0..4 반응을 kNN 대조. 여기서는
  고EC 35기록의 |Δ|≥.20 계단 전부, 사건 '그 시각' 변화, 시각대 층화 순열(2000) + 외기 일사 음성 대조.
사용 가능성: 계단 자체가 정답(EC) 사건이라 이 분석 전체가 진단용(diagnostic). 구동기 흔적이 있다면 그 흔적만 usable 후보.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(7)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
HO = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")).sort_values(["rec", "hour"])
hiR = R.index[R.hi10 == 1]
H = HO[HO.rec.isin(hiR)].copy()
FE = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog", "in_hum", "in_temp", "in_co2", "out_rad", "sub_temp"]
for c in FE + ["sub_ec"]:
    H["d_" + c] = H.groupby("rec")[c].diff()
H = H[H.hour >= 1].copy()
H["fog_on"] = (H.act_fog > 0).astype(float); H["co2_on"] = (H.act_co2 > 0).astype(float)
dn = H.d_sub_ec <= -0.20; up = H.d_sub_ec >= 0.20; pl = H.d_sub_ec.abs() < 0.05
print(f"high records {len(hiR)}; steps: drop<=-.20 {dn.sum()} (records {H[dn].rec.nunique()}), rise>=+.20 {up.sum()} (records {H[up].rec.nunique()}), placebo {pl.sum()}")
print("drop events:\n", H[dn][["rec", "hour", "d_sub_ec", "sub_ec", "d_act_vent", "d_act_shade", "d_act_thermal", "d_act_heating", "act_fog", "d_in_hum", "d_out_rad", "d_sub_temp"]].round(2).to_string(index=False))
print("rise events:\n", H[up][["rec", "hour", "d_sub_ec", "sub_ec", "d_act_vent", "d_act_shade", "d_act_thermal", "d_act_heating", "act_fog", "d_in_hum", "d_out_rad", "d_sub_temp"]].round(2).to_string(index=False))
TV = ["d_" + c for c in FE] + ["fog_on", "co2_on"]
rows = []
for nm, ev in [("drop", dn), ("rise", up)]:
    S = H[ev | pl].copy(); S["ev"] = ev[ev | pl].astype(int).values
    # restrict placebo to hours where events occur (hour strata)
    S = S[S.hour.isin(S[S.ev == 1].hour.unique())]
    for c in TV:
        x = S[c].values; e = S.ev.values; hr = S.hour.values; m = np.isfinite(x)
        x, e, hr = x[m], e[m], hr[m]
        # hour-matched placebo mean: average over events of the placebo mean at the same hour
        def stat(ee):
            diffs = []
            for h in np.unique(hr[ee == 1]):
                a = x[(hr == h) & (ee == 1)]; b = x[(hr == h) & (ee == 0)]
                if len(b): diffs += [ai - b.mean() for ai in a]
            return np.mean(diffs) if diffs else np.nan
        obs = stat(e); null = []
        for _ in range(2000):
            ep = e.copy()
            for h in np.unique(hr):
                k = np.where(hr == h)[0]; ep[k] = rng.permutation(ep[k])
            null.append(stat(ep))
        null = np.array(null); sd = np.nanstd(x[e == 0])
        p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)
        rows.append(dict(event=nm, var=c, n_ev=int(e.sum()), n_plac=int((e == 0).sum()), diff=obs, diff_sd=obs / sd if sd > 0 else np.nan, p_perm=p))
T = pd.DataFrame(rows)
T["bonf_p"] = np.minimum(T.p_perm * len(T), 1)
T.to_csv(os.path.join(HERE, "flush_v1.csv"), index=False, encoding="utf-8")
print(f"\ntests: {len(T)} (Bonferroni x{len(T)})")
print(T.round(3).to_string(index=False))
