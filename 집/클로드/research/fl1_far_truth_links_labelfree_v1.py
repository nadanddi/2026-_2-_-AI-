import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); os.chdir(r"C:\work\farmai\집\클로드\research")
import env, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "linker_v4"); from feat4 import EC, X, IN_, FARM, NUM, N, TRAIN, S3
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join("ec0h","rise","dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
sd = np.nanstd(X[TRAIN].reshape(-1, 4), 0); sdi = np.nanstd(IN_[TRAIN].reshape(-1, IN_.shape[2]), 0)
def oj(a, b):
    pa = X[a,23] + .5*(X[a,23]-X[a,22]); pb = X[b,0] - .5*(X[b,1]-X[b,0]); return float(np.sum(np.array([1,1,.3,.5])*np.abs(pa-pb)/sd))
def ij(a, b):
    return float(np.nanmean(np.abs(IN_[a,23]-IN_[b,0])/sdi))
# baseline: oj / ij distribution for gap-1 truth links (pass-2) and random pairs
g1 = [(int(a), int(b)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1]
print("gap-1 truth links: oj median %.2f q90 %.2f | ij median %.2f q90 %.2f" % (np.median([oj(a,b) for a,b in g1]), np.quantile([oj(a,b) for a,b in g1], .9), np.nanmedian([ij(a,b) for a,b in g1]), np.nanquantile([ij(a,b) for a,b in g1], .9)))
for b_ in ["F47_209","F13_231","F47_230","F13_212","F13_179","F47_229","F13_232","F13_211","F13_241"]:
    b = idx[b_]; a = int(TL.a1[TL.b == b].iloc[0])
    cand = [k for k in range(N) if k != b]
    ojs = np.array([oj(k, b) for k in cand]); order = np.argsort(ojs)
    best = [(rid(cand[k]), round(ojs[k], 2), D11.get(rid(cand[k]), np.nan) - D11.get(b_, np.nan), round(EC[cand[k], 23], 2) if TRAIN[cand[k]] else None) for k in order[:4]]
    rank_a = int(np.where(np.array(cand)[order] == a)[0][0]) + 1
    print(f"{b_} (b0 {EC[b,0]:.2f}, DM11 {D11.get(b_)}): truth a1 {rid(a)} oj {oj(a,b):.2f} rank {rank_a} ij {ij(a,b):.2f} a23 {EC[a,23]:.2f} | best outdoor: {best}")
