import json, numpy as np, pandas as pd
O = r"C:\work\farmai\집\클로드\research\audit_temp"
L = pd.read_pickle(O + r"\lab_feats.pkl"); T = pd.read_pickle(O + r"\test_feats.pkl"); C = json.load(open(O + r"\cols.json"))
cols = list(dict.fromkeys(C["ct"] + C["phc"] + C["cx"]))
tr_days = {(f, d) for f, d in zip(L.farm, L.day)}; te_days = {(f, d) for f, d in zip(T.farm, T.day)}
def prev_status(f, d):
    p = (f, d - 1)
    return "train" if p in tr_days else ("test" if p in te_days else "deleted")
for D in (L, T):
    D["prev"] = [prev_status(f, d) for f, d in zip(D.farm, D.day)]
print("train records by prev status:", L.groupby("prev").day.size().div(24).to_dict())
print("test records by prev status:", T.groupby("prev").day.size().div(24).to_dict())
# in MASK world, a train record whose prev is deleted had test before that -> masked
rows = []
for c in cols:
    a = L[c].astype(float); b = T[c].astype(float)
    trn = a.isna().mean(); ten = b.isna().mean()
    tsn = b[T.prev == "deleted"].isna().mean(); tnn = b[T.prev != "deleted"].isna().mean()
    lgn = a[L.prev == "deleted"].isna().mean()
    lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
    out = ((b < lo) | (b > hi)).mean()
    # chunk-start vs other eval rows, standardized by train sd, matched hour
    sd = np.nanstd(a) or 1
    ds = (b[T.prev == "deleted"].groupby(T.hour).mean() - b[T.prev != "deleted"].groupby(T.hour).mean()).abs().mean() / sd
    rows.append(dict(feat=c, member=("BASE" if c in C["ct"] else "") + ("|PHYS" if c in C["phc"] else "") + ("|CODEX/PFN" if c in C["cx"] else ""),
                     nan_tr=trn, nan_te=ten, nan_te_start=tsn, nan_te_rest=tnn, nan_tr_aftergap=lgn,
                     out_of_train_995=out, start_vs_rest_sd=ds,
                     smd=(np.nanmean(b) - np.nanmean(a)) / sd))
R = pd.DataFrame(rows); R.to_csv(O + r"\feature_compare.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(R.sort_values("nan_te_start", ascending=False).head(40).round(3).to_string())
print("\n-- top out-of-range --"); print(R.sort_values("out_of_train_995", ascending=False).head(25).round(3).to_string())
print("\n-- top start vs rest shift (sd units) --"); print(R.sort_values("start_vs_rest_sd", ascending=False).head(25).round(3).to_string())
print("\n-- NaN only in train (nan_tr>0.02, nan_te==0) --"); print(R[(R.nan_tr > 0.02) & (R.nan_te == 0)].round(3).to_string())
