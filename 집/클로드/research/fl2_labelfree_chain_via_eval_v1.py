import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); os.chdir(r"C:\work\farmai\집\클로드\research")
import env, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "linker_v4"); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN, S3
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join("ec0h","rise","dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
DAY = np.array([D11.get(rid(i), np.nan) for i in range(N)])
sdi = np.nanstd(IN_[TRAIN].reshape(-1, IN_.shape[2]), 0)
_V = ["act_heating","act_thermal","act_vent","act_shade","act_circfan","act_co2","act_fog","in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv","test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13","F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0,1))) / (np.nanstd(FPA, (0,1)) + 1e-9)
fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FPA[a]-FPA[b]), 4)))
ij = lambda a, b: float(np.nanmean(np.abs(IN_[a,23]-IN_[b,0])/sdi))
def prev(b):
    c = np.where(DAY == DAY[b] - 1)[0]
    if len(c) == 0: return None, None
    s = np.array([ij(k, b) + fpd(k, b) for k in c]); o = np.argsort(s)
    return c[o[0]], (s[o[1]] / s[o[0]] if len(o) > 1 else 9)
rows = []
for b in range(N):
    if not TRAIN[b] or NUM[b] < 179 or not np.isfinite(DAY[b]): continue
    cur = b; path = []; ok_rule = True
    for step in range(15):
        p, mg = prev(cur)
        if p is None: break
        path.append(rid(p)); ok_rule &= (FARM[p] == FARM[b]) or TRAIN[p]
        if TRAIN[p] and np.isfinite(EC[p, 23]): break
        cur = p
    anc = idx[path[-1]] if path and TRAIN[idx[path[-1]]] else None
    rows.append(dict(rec=rid(b), y0=EC[b,0], ydm=np.nanmean(EC[b]), k=len(path) if anc is not None else 0, anc=path[-1] if anc is not None else "", a23=EC[anc,23] if anc is not None else np.nan, via_eval=sum(not TRAIN[idx[q]] for q in path), path=" ".join(path)))
R = pd.DataFrame(rows); R["err"] = (R.a23 - R.y0).abs()
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(R.groupby("k").agg(n=("rec","size"), med_err=("err","median"), mean_err=("err","mean"), share_lt10=("err", lambda s: (s<.10).mean()), ydm=("ydm","mean")).round(3).to_string())
print(R[(R.k >= 2) | (R.ydm >= 1.0)].sort_values("k").round(3).to_string())
