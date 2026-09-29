# -*- coding: utf-8 -*-
"""Q1: date-level successor by outdoor-weather continuity across midnight (analysis)."""
import env  # noqa
import numpy as np, pandas as pd
import common
from scipy.optimize import linear_sum_assignment
tX, ty, sX = common.load_raw()
k = pd.read_csv(env.LOCAL + "/deep_cal_2_days.csv")
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])]
rep = k.sort_values("day").groupby("date").first().reset_index()  # representative (farm, day)
W = ["out_temp", "out_hum", "out_wspd"]
M = {}
for _, r in rep.iterrows():
    g = a[(a.farm == r.farm) & (a.day == r.day)].sort_values("hour")
    M[r.date] = g[W].to_numpy(float)
D = sorted(M)
n = len(D)
X = np.stack([M[d] for d in D])  # n,24,3
# scale: typical hourly change within day for each variable
dif = np.diff(X, axis=1)
sc = np.nanstd(dif.reshape(-1, 3), axis=0)
print("hourly change sd", dict(zip(W, sc.round(3))))
def cost_mat(wts=(1, 1, 0.3)):
    C = np.zeros((n, n))
    for vi, w in enumerate(wts):
        a23 = X[:, 23, vi]; a22 = X[:, 22, vi]; b0 = X[:, 0, vi]; b1 = X[:, 1, vi]
        sa = a23 - a22; sb = b1 - b0
        pred_gap = 0.5 * (sa[:, None] + sb[None, :])
        gap = b0[None, :] - a23[:, None]
        C += w * ((gap - pred_gap) / sc[vi]) ** 2
    np.fill_diagonal(C, np.inf)
    return np.nan_to_num(C, nan=50.0)
C = cost_mat()
np.save(env.LOCAL + "/deep_cal_3_cost.npy", C)
# rank of date i+1 as successor of date i (ids from first-pass order)
rk = [(C[i] < C[i, i + 1]).sum() for i in range(133)]
rk = np.array(rk)
print("first-pass: rank of id+1 among successors: top1 %.2f top3 %.2f median %d" % ((rk == 0).mean(), (rk < 3).mean(), np.median(rk)))
# nearest successor margin
srt = np.sort(C, axis=1)
print("best cost median %.2f, 2nd best median %.2f, ratio2/1 median %.1f" % (np.median(srt[:, 0]), np.median(srt[:, 1]), np.median(srt[:, 1] / srt[:, 0])))
# null: cost for id+1 vs random pairs
print("cost(i,i+1) median %.2f | random pair median %.2f" % (np.median([C[i, i+1] for i in range(133)]), np.median(C[np.isfinite(C)])))
# assignment with a dummy 'no successor' option
big = np.full((n, n), 8.0)
CC = np.block([[C, np.where(np.eye(n) == 1, 8.0, 1e6)], [np.where(np.eye(n) == 1, 8.0, 1e6), np.zeros((n, n))]])
r, c = linear_sum_assignment(np.minimum(CC, 1e6))
succ = {}
for i, j in zip(r, c):
    if i < n and j < n:
        succ[i] = j
print("assigned successors", len(succ))
ok = sum(1 for i in range(133) if succ.get(i) == i + 1)
print("assignment agrees with id+1 for first pass: %d/133" % ok)
# chains
pred = {j: i for i, j in succ.items()}
chains = []
for s in range(n):
    if s in pred:
        continue
    ch = [s]
    while ch[-1] in succ and len(ch) < 500:
        ch.append(succ[ch[-1]])
    chains.append(ch)
print("n chains", len(chains), "lengths", sorted([len(c) for c in chains], reverse=True)[:15])
for ch in sorted(chains, key=len, reverse=True)[:12]:
    print("  ", ch)
pd.Series(succ).to_csv(env.LOCAL + "/deep_cal_3_succ.csv")
