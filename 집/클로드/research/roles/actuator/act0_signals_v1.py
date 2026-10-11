# 구동기 유래 0시 신호 3개 vs DIAG10 0시 EC 잔차 (탐색, 사후). 우연 대조 = 온실·구간 내 순열.
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
R = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
R = R[(R.tag=="CUR")&(R.validator=="DIAG10")&(R.hour==0)]
R = R.groupby(["row_id","farm","day"],as_index=False).agg(sg=("sg","mean"),y=("sub_ec","first"))
R["res"] = R.y - R.sg   # +: 실제가 예측보다 높음
x = X[X.row_id.isin(R.row_id)].copy()
def ah(t,rh): return 216.7*(rh/100*6.112*np.exp(17.62*t/(243.12+t)))/(273.15+t)
x["S1_moist_excess"] = ah(x.in_temp,x.in_hum)-ah(x.out_temp,x.out_hum)   # 밀폐 하 수분 공급(증산+배지 증발)
x["S2_co2_excess"] = np.where(x.act_co2.fillna(0)==0, x.in_co2-420, np.nan) # 공급 없을 때 호흡 축적
x["dT"] = x.in_temp-x.out_temp
# S3: 난방·커튼·환기·풍속으로 설명 안 되는 실내외 온도차(온실 내부 축열·근권 열원 대리), 온실별 선형잔차
out=[]
for f,g in x.groupby("farm" if "farm" in x else x.row_id.str[:3]):
    A = np.c_[np.ones(len(g)), g.act_heating.fillna(0), g.act_thermal.fillna(0), g.act_vent.fillna(0), g.out_wspd.fillna(0), g.out_temp]
    ok = ~np.isnan(g.dT.values)
    b = np.linalg.lstsq(A[ok], g.dT.values[ok], rcond=None)[0]
    out.append(pd.Series(g.dT.values - A@b, index=g.index))
x["S3_warm_unexpl"] = pd.concat(out)
D = R.merge(x[["row_id","S1_moist_excess","S2_co2_excess","S3_warm_unexpl","act_vent","act_circfan"]], on="row_id")
D["seg"] = np.where(D.day>=179,"2차","1차")
D["sealed"] = (D.act_vent.fillna(0)<=1)
rng = np.random.default_rng(0)
print("N 0시 행", len(D), "| res sd %.3f" % D.res.std())
for s in ["S1_moist_excess","S2_co2_excess","S3_warm_unexpl"]:
    for sub,lab in [(D,"전체"),(D[D.sealed],"밀폐")]:
        for (f,seg),g in sub.groupby(["farm","seg"]):
            g=g.dropna(subset=[s,"res"])
            if len(g)<10: print(f"{s:16s} {lab} {f} {seg} n={len(g)} (적음)"); continue
            r=spearmanr(g[s],g.res).correlation
            null=np.array([spearmanr(rng.permutation(g[s].values),g.res).correlation for _ in range(2000)])
            p=(np.abs(null)>=abs(r)).mean()
            print(f"{s:16s} {lab} {f} {seg} n={len(g):3d} rho={r:+.3f} 순열p={p:.3f}")
