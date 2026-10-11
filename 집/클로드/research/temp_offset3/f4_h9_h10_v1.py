# -*- coding: utf-8 -*-
"""temp_offset3 F4: H9(밤 실내<외기 입력 이상), H10(− 날 낮 반응 폭) — 계획 03."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from scipy.stats import spearmanr
RES = r"C:\work\farmai\집\클로드\research"; OUTD = os.path.join(RES, "temp_offset3")
F = pd.read_csv(os.path.join(OUTD, "f1_records_v1.csv"))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Yd = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); X = X.merge(Yd, on="row_id")
T = pd.read_pickle(os.path.join(RES, r"ev_validator\temp\local_ckpt\ev_all.pkl")); T["q"] = T[[c for c in T.columns if c.startswith("v22_")]].mean(axis=1)
X = X.merge(T[["row_id", "q"]], on="row_id"); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[-2:].astype(int)
g = X.groupby("rec")
night = X[X.h <= 5].groupby("rec").apply(lambda d: (d.in_temp - d.out_temp).mean(), include_groups=False)
amp_y = g.sub_temp.max() - g.sub_temp.min(); amp_q = g.q.max() - g.q.min()
F["inout_n"] = F.rec.map(night); F["ampr"] = F.rec.map(amp_y / amp_q)
print("=== H9 밤 실내−외기 ≤ −2 ===")
m = F.inout_n <= -2
print(F.loc[m, ["rec", "o", "inout_n", "tin_n", "tout_n", "big"]].round(2).to_string(index=False))
print(f" n {m.sum()} | + 부호 {np.mean(F.o[m] > 0):.2f} | 평균 o {F.o[m].mean():+.2f} | (−1 이하 n {(F.inout_n <= -1).sum()}, 평균 o {F.o[F.inout_n <= -1].mean():+.2f})")
print(" 전체 ρ(o, 실내−외기 밤)", round(spearmanr(F.o, F.inout_n, nan_policy="omit")[0], 3))
print("\n=== H10 낮 반응 폭 비율 (배지 폭 / q 폭) ===")
for nm, mm in (("큰 − 날 F47", F.big & (F.o < 0) & (F.farm == "F47")), ("큰 + 날 F47", F.big & (F.o > 0) & (F.farm == "F47")), ("나머지", ~F.big)):
    print(f"  {nm:10s} n {mm.sum():3d} | 폭 비율 중앙 {F.ampr[mm].median():.2f} 평균 {F.ampr[mm].mean():.2f} | <0.8 비율 {np.mean(F.ampr[mm] < .8):.2f}")
print("  ρ(o, 폭비율) 전체", round(spearmanr(F.o, F.ampr)[0], 3), "| F47", round(spearmanr(F.o[F.farm == 'F47'], F.ampr[F.farm == 'F47'])[0], 3))
# persistence of amplitude ratio along truth same-ID link
import pickle
R = pickle.load(open(os.path.join(RES, r"local\linker_v3\base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
ids = [f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]
TT = pd.read_pickle(os.path.join(RES, r"local\linker_v3\truth.pkl"))
pv = {ids[int(t.b)]: ids[int(t.a1)] for _, t in TT.iterrows() if t.cert and t.same_farm}
F["ampr_prev"] = F.rec.map(lambda x: F.set_index("rec").ampr.get(pv.get(x), np.nan))
mm = F.ampr_prev.notna()
print(f"  ρ(폭비율, 전날 같은 동 폭비율) n {mm.sum()}: {spearmanr(F.ampr[mm], F.ampr_prev[mm])[0]:+.3f} | F47 {spearmanr(F.ampr[mm & (F.farm=='F47')], F.ampr_prev[mm & (F.farm=='F47')])[0]:+.3f}")
F.to_csv(os.path.join(OUTD, "f4_records_v1.csv"), index=False, encoding="utf-8-sig")
