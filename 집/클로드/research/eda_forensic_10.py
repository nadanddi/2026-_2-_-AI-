# -*- coding: utf-8 -*-
"""Forensic 10: does EC OOF error differ on input-anomaly rows/days?"""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
O = np.load(env.LOCAL+"/ec_oof_members.npy")
print("oof shape",O.shape)
tr = G[(G.set=="train")].sort_values(["farm","t"]).reset_index(drop=True)
assert len(tr)==9600
p = np.nanmean(O[0],axis=0)  # ET member, mean over placements A/B
ok=~np.isnan(p)
print("corr oof vs y", np.corrcoef(p[ok], tr.sub_ec.values[ok])[0,1], "rmse", np.sqrt(np.mean((p[ok]-tr.sub_ec.values[ok])**2)))
tr["err"]=p-tr.sub_ec
D=pd.read_csv(env.LOCAL+"/eda_forensic_7_days.csv")
tr=tr.merge(D[D.set=="train"][["farm","day","rough_AH","rough_T_night","rough_C_night","maxjump_res"]],on=["farm","day"],how="left")
flags={
 "na_day": tr.groupby(["farm","day"]).in_temp.transform(lambda s:s.isna().any()),
 "dist_na<=6": tr.dist_na<=6,
 "temp_run>=6": tr.temp_run>=6,
 "night_co2<300": (tr.in_co2<300)&~tr.hour.between(7,19),
 "co2_day_night_low": tr.groupby(["farm","day"]).in_co2.transform(lambda s:(s<300).any()),
 "rough_AH>test_p95": tr.rough_AH>0.70,
 "rough_T>test_p95": tr.rough_T_night>1.12,
 "rough_C>test_p95": tr.rough_C_night>17.9,
 "big_temp_res_day": tr.groupby(["farm","day"]).res.transform(lambda s:s.abs().mean())>1.5,
}
for k,m in flags.items():
    m=m.fillna(False).values.astype(bool)&ok
    r_in=np.sqrt(np.nanmean(tr.err[m]**2)); r_out=np.sqrt(np.nanmean(tr.err[ok&~m]**2))
    print(f"{k:22s} n={m.sum():5d}  EC oof rmse {r_in:.3f} vs {r_out:.3f}  |sub_temp res| {tr.res[m].abs().mean():.2f} vs {tr.res[ok&~m].abs().mean():.2f}")
tr[["row_id","err"]].to_csv(env.LOCAL+"/eda_forensic_10_err.csv",index=False)
