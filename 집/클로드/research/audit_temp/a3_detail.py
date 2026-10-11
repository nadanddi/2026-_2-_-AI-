import json, numpy as np, pandas as pd
O = r"C:\work\farmai\집\클로드\research\audit_temp"
L = pd.read_pickle(O + r"\lab_feats.pkl"); T = pd.read_pickle(O + r"\test_feats.pkl"); C = json.load(open(O + r"\cols.json"))
tr_days = {(f, d) for f, d in zip(L.farm, L.day)}; te_days = {(f, d) for f, d in zip(T.farm, T.day)}
ps = lambda f, d: "train" if (f, d-1) in tr_days else ("test" if (f, d-1) in te_days else "deleted")
for D in (L, T): D["prev"] = [ps(f, d) for f, d in zip(D.farm, D.day)]
pd.set_option("display.width", 250)
ev = [c for c in C["ct"] if "_hsince" in c]
print("== event clocks (hours): median / share>24h ; train-normal | train-aftergap | eval-start | eval-rest ==")
for c in ev:
    a = L[c]; b = T[c]
    f = lambda s: "%6.1f/%4.2f" % (np.nanmedian(s), (s > 24).mean())
    print("%-24s %s | %s | %s | %s" % (c, f(a[L.prev == "train"]), f(a[L.prev == "deleted"]), f(b[T.prev == "deleted"]), f(b[T.prev != "deleted"])))
print("\n== eval-start records, hour 0-6: act_*_hsince_chg values (one per record) ==")
S = T[(T.prev == "deleted") & (T.hour == 0)]
print(S[["row_id"] + [c for c in ev if c.endswith("chg")]].round(0).to_string())
print("\n== day / season ==")
print("train day range", L.day.min(), L.day.max(), " eval", T.day.min(), T.day.max())
print("train cx__day (season) range %.3f..%.3f ; eval %.3f..%.3f" % (L["cx__day"].min(), L["cx__day"].max(), T["cx__day"].min(), T["cx__day"].max()))
print("train second=1 records", L[L["cx__second"] == 1].day.nunique() if "cx__second" in L else None, "eval second", T["cx__second"].unique())
for f in ("F13", "F47"):
    a = L[L.farm == f].groupby("day")["cx__day"].first(); b = T[T.farm == f].groupby("day")["cx__day"].first()
    print(f, "eval season by day:", b.round(2).to_dict())
    near = a[(a.index >= 170)]
    print(f, "train season days>=170:", near.round(2).to_dict())
# noise scale: hourly diff std per input, train (clean-ish rows w==1 vs all) vs eval
print("\n== input roughness: std of hour-to-hour diff (same record), train all / train w=1 / eval ==")
import os
D = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
tX = pd.read_csv(os.path.join(D, "train_X.csv")); sX = pd.read_csv(os.path.join(D, "test_X.csv"))
for d in (tX, sX):
    d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int)
tX = tX[tX.farm.isin(["F13", "F47"])]
wmap = dict(zip(L.row_id, L.w))
tX["w"] = tX.row_id.map(wmap)
for c in ["in_temp", "in_hum", "in_co2", "out_temp", "act_heating", "act_vent"]:
    def rough(d):
        d = d.sort_values(["farm", "day", "hour"]); x = d.groupby(["farm", "day"])[c].diff(); return x.abs().median(), x.std()
    a = rough(tX); b = rough(tX[tX.w == 1]); e = rough(sX)
    print("%-12s train |d| med %.3f sd %.3f | train w=1 med %.3f sd %.3f | eval med %.3f sd %.3f" % (c, *a, *b, *e))
