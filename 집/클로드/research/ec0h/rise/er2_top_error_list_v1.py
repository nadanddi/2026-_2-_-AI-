# -*- coding: utf-8 -*-
"""ER2 (2026-10-12 집 클로드; 사용자: '오차 가장 큰 곳 리스트로 쭉 뽑아봐, EC 랑 온도 둘 다').  서술.
기준: EC = EV EA2(3시드 평균), 온도 = EV v22(구성 평균).  400 기록(1차 344 + 2차 56) 모두 검증 예측.
기록별: 제곱오차 합(SSE), 그중 하루 수준 몫(24·평균오차²), 평균오차(부호), 0시 오차, 23시−0시 변화 오차, 정답 수준, 앵커 여부, 날짜 종류(DM11).
구간별: SSE 비중(차수·농장·시각대·정답 수준대·앵커·날짜 종류)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")).set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
QE = R.groupby(["rec", "hour"]).EA2.mean().unstack(); YE = R.groupby(["rec", "hour"]).y.first().unstack().loc[QE.index]; AE = R.groupby("rec").anchored.first()
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
T["q"] = T[[c for c in T.columns if c in ("v22_a", "v22_b", "v22_ad", "v22_bd")]].mean(axis=1)
QT = T.pivot(index="rec", columns="hour", values="q"); YT = T.pivot(index="rec", columns="hour", values="sub_temp").loc[QT.index]; AT = T.groupby("rec").anchored.first()
IT = T.pivot(index="rec", columns="hour", values="in_temp").loc[QT.index]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
def report(name, Q, Y, A, lvl_bins, unit):
    E = Y - Q; sse = (E ** 2).sum(1); tot = sse.sum(); me = E.mean(1)
    D = pd.DataFrame(dict(SSE=sse, share=sse / tot, day_part=24 * me ** 2 / sse, mean_err=me, err0=E[0], err23=E[23],
                          dchange_err=(Y[23] - Y[0]) - (Q[23] - Q[0]), y_mean=Y.mean(1), anchored=A.reindex(Q.index).fillna(False).astype(bool)))
    D["pass"] = [2 if int(r[4:7]) >= 179 else 1 for r in D.index]; D["farm"] = D.index.str[:3]
    D["kind"] = CAL.kind.reindex(D.index).values; D["day"] = CAL.day.reindex(D.index).values
    rm_all = float(np.sqrt((E ** 2).values.mean()))
    print(f"\n######## {name}: RMSE {rm_all:.4f} (pass-1 {np.sqrt((E[D['pass']==1]**2).values.mean()):.4f}, pass-2 {np.sqrt((E[D['pass']==2]**2).values.mean()):.4f}) | day-level share of SSE {(24*me**2).sum()/tot:.2f}")
    print(f"top 20 records ({unit}):")
    print(D.sort_values("SSE", ascending=False).head(20)[["share", "day_part", "mean_err", "err0", "err23", "dchange_err", "y_mean", "anchored", "pass", "kind", "day"]].round(3).to_string())
    k = D.sort_values("SSE", ascending=False); cs = k.share.cumsum()
    print("concentration: top 5 %.2f | top 10 %.2f | top 20 %.2f | top 40 %.2f of SSE" % tuple(cs.iloc[n - 1] for n in (5, 10, 20, 40)))
    print("SSE share by group (records share in parentheses):")
    D["level"] = pd.cut(D.y_mean, lvl_bins)
    for g in ("pass", "farm", "anchored", "kind", "level"):
        s = D.groupby(g, observed=True).SSE.sum() / tot; n = D.groupby(g, observed=True).size() / len(D)
        print(f"  {g:8s}: " + " | ".join(f"{i}: {s[i]:.2f} ({n[i]:.2f})" for i in s.index))
    hb = {f"{a}-{b}h": float((E.loc[:, a:b] ** 2).values.sum() / tot) for a, b in ((0, 5), (6, 11), (12, 17), (18, 23))}
    print("  hour blocks:", " | ".join(f"{k_}: {v:.2f}" for k_, v in hb.items()))
    for p in (1, 2):
        sub = D[D["pass"] == p].sort_values("SSE", ascending=False)
        print(f"  pass-{p} top 5 = {sub.SSE.head(5).sum()/sub.SSE.sum():.2f} of pass-{p} SSE: " + ", ".join(f"{r} ({v.mean_err:+.2f}, y {v.y_mean:.2f})" for r, v in sub.head(8).iterrows()))
    return D
DE = report("EC (EV EA2)", QE, YE, AE, [0, .4, .6, .8, 1.0, 1.5, 9], "EC")
DT = report("TEMP (EV v22)", QT, YT, AT, [0, 10, 12, 14, 16, 18, 40], "℃")
cold = IT.min(1) < 8
print("\nTEMP extra: records with indoor min < 8℃: %d (%.2f of records) hold %.2f of temp SSE" % (int(cold.sum()), cold.mean(), DT.SSE[cold.values].sum() / DT.SSE.sum()))
both = DE.SSE.rank(ascending=False) <= 40; both_t = DT.SSE.rank(ascending=False) <= 40
print("records in top-40 of BOTH EC and TEMP:", sorted(DE.index[both & both_t.reindex(DE.index).fillna(False)]))
DE.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "er2_ec_records_v1.csv")); DT.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "er2_temp_records_v1.csv"))
