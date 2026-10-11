# Sensitivity: how much do BASE predictions on chunk-start eval rows move if the deleted-day artifacts are replaced by causal in-record fills?
import os, sys, json
os.environ["AGRI_DATA"] = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
sys.path.insert(0, r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도\code")
import env
import numpy as np, pandas as pd, time
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from make_submission_v3 import m_lgbh, m_ridge, m_nys
O = r"C:\work\farmai\집\클로드\research\audit_temp"
L = pd.read_pickle(O + r"\lab_feats.pkl"); T = pd.read_pickle(O + r"\test_feats.pkl"); C = json.load(open(O + r"\cols.json"))
ct, phc = C["ct"], C["phc"]
te_days = set(zip(T.farm, T.day))
start = np.array([(f, d - 1) not in te_days for f, d in zip(T.farm, T.day)])
D = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
sX = pd.read_csv(os.path.join(D, "test_X.csv")).set_index("row_id")
F = T.copy()
S = F[start].copy()
raw = sX.loc[S.row_id]
S["_rec"] = S.farm + S.day.astype(str)
# lags: persistence from the record's own earliest hours
for L_ in (1, 2, 3):
    c = "in_temp_lag%d" % L_
    if c in S: S[c] = S[c].fillna(S.groupby("_rec").in_temp.transform(lambda x: x.shift(L_)).fillna(S.groupby("_rec").in_temp.transform("first")))
for c, src in (("out_temp_lag1", raw.out_temp.values), ("rad_eff_lag1", None)):
    if c in S and src is not None:
        tmp = pd.Series(src, index=S.index); S[c] = S[c].fillna(tmp.groupby(S._rec).shift(1)).fillna(tmp)
if "rad_eff_lag1" in S: S["rad_eff_lag1"] = S["rad_eff_lag1"].fillna(0.0)
for c in ("in_temp_d1", "in_temp_d3"):
    if c in S: S[c] = S[c].fillna(0.0)
# r24 features: in-record expanding stats
base_src = {"in_temp": S.in_temp, "out_temp": pd.Series(raw.out_temp.values, index=S.index)}
for c in [x for x in ct if x.endswith("_r24m") or x.endswith("_r24s")]:
    v = c[:-5]
    if v in base_src:
        s = base_src[v]
        fill = s.groupby(S._rec).transform(lambda x: x.expanding().mean() if c.endswith("m") else x.expanding().std().fillna(0))
        S[c] = S[c].fillna(fill)
    else:
        # rad/transp/vpd/heating: use the eval non-start mean at the same hour as a neutral fill
        hm = T[~start].groupby("hour")[c].mean()
        S[c] = S[c].fillna(S.hour.map(hm))
# event clocks: count from record start instead of through the deleted day
for c in [x for x in ct if "_hsince" in x]:
    S[c] = np.minimum(S[c], S.hour + 1)
F.loc[start, ct] = S[ct].values
chg = (~np.isclose(F[ct].astype(float).values, T[ct].astype(float).values, equal_nan=True)).any(1)
print("eval rows with any changed feature: %d (all in chunk-start records: %s)" % (chg.sum(), bool(chg[~start].sum() == 0)))
w = L.w.values; y = L.sub_temp.values
t0 = time.time()
imp = SimpleImputer(strategy="median").fit(L[phc]); lin = LinearRegression().fit(imp.transform(L[phc]), y, sample_weight=w)
btr = lin.predict(imp.transform(L[phc])); bte = lin.predict(imp.transform(T[phc]))
s = 7
g = m_lgbh(s).fit(L[ct], y - btr, sample_weight=w)
r = m_ridge(s).fit(L[ct], y, ridge__sample_weight=w)
n = m_nys(s).fit(L[ct], y, ridge__sample_weight=w)
print("fit %.0fs" % (time.time() - t0))
out = {}
for name, X in (("orig", T), ("fixed", F)):
    pl, pr, pn = g.predict(X[ct]), r.predict(X[ct]), n.predict(X[ct])
    out[name] = dict(lgb=bte + pl, ridge=pr, nys=pn, base=0.65 * (bte + pl) + 0.25 * pr + 0.10 * pn)
for k in ("lgb", "ridge", "nys", "base"):
    d = out["fixed"][k] - out["orig"][k]
    print("%-6s change on changed rows: n=%d RMS %.3f mean %+.3f max|.| %.3f" % (k, chg.sum(), np.sqrt((d[chg] ** 2).mean()), d[chg].mean(), np.abs(d[chg]).max()))
d = out["fixed"]["base"] - out["orig"]["base"]
H = pd.DataFrame({"hour": T.hour[chg], "d": d[chg]}).groupby("hour").d.agg(["mean", "count"]).round(3)
print(H.T.to_string())
print("final-pred effect (BASE weight 0.4): RMS %.3f on %d rows; over all 1440 rows RMS %.4f" % (0.4 * np.sqrt((d[chg] ** 2).mean()), chg.sum(), 0.4 * np.sqrt((d ** 2).mean())))
# which group drives it: re-run with only one group fixed
groups = {"lags/d": [c for c in ct if "_lag" in c or c in ("in_temp_d1", "in_temp_d3")], "r24": [c for c in ct if "_r24" in c], "event": [c for c in ct if "_hsince" in c]}
for gname, cols in groups.items():
    X = T.copy(); X.loc[start, cols] = F.loc[start, cols].values
    dd = 0.65 * g.predict(X[ct]) + 0.25 * r.predict(X[ct]) + 0.10 * n.predict(X[ct]) - (0.65 * g.predict(T[ct]) + 0.25 * r.predict(T[ct]) + 0.10 * n.predict(T[ct]))
    cc = np.abs(dd) > 1e-9
    print("  only %-7s fixed: rows moved %d, RMS(BASE) %.3f, mean %+.3f" % (gname, cc.sum(), np.sqrt((dd[cc] ** 2).mean()) if cc.any() else 0, dd[cc].mean() if cc.any() else 0))
