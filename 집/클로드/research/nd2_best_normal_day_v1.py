# -*- coding: utf-8 -*-
"""ND2 (2026-10-14 집 클로드; 사용자: '지금까지 실험 중 보통 날 EC 가장 잘 맞춘 게 뭐야').  서술.  저장된 예측이 있는 후보들을 같은 기준(실제 하루 평균 < 1.0, 2차 행)으로 비교.
rv1(DIAG10·P2LOO·EL1, 시드 47/1414/6464): 태그 CUR(.8 R3_DP1 + .2 PFN)·B4(.6/.4)·CURa·B4a(DIAG10 만, 다른 PFN 추출본), 열 nosg(SG2 전)·sg(SG2 후); + EG1 GRU 10%.
EV(시드 7/101/2024): 열 R3·SG2·EA2(앵커) + EG2 GRU 10%."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O["rec"] = O.row_id.str[:7]
O = O[(O.day >= 179).values & np.array([DM[idx[r]] < 1.0 for r in O.rec])]
G1 = pd.read_csv(os.path.join(HERE, "eg1_gru_preds_v1.csv")).set_index(["val", "rec"])
rows = []
for (v, tag, sd), G in O.groupby(["validator", "tag", "seed"]):
    rows.append(dict(val=v, cand=f"{tag} nosg", seed=sd, rmse=rm(G.nosg - G.sub_ec))); rows.append(dict(val=v, cand=f"{tag} sg", seed=sd, rmse=rm(G.sg - G.sub_ec)))
    if tag == "CUR":
        gr = np.array([G1.at[(v, r), f"h{h}"] for r, h in zip(G.rec, G.hour)]); rows.append(dict(val=v, cand="CUR sg + GRU10", seed=sd, rmse=rm(.9 * G.sg + .1 * gr - G.sub_ec)))
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.seed.astype(str) != "avg") & (R.day >= 179)]; R = R[[DM[idx[r]] < 1.0 for r in R.rec]]
GE = pd.read_csv(os.path.join(HERE, "eg2_gru_ev_preds_v1.csv")).set_index("rec")
for (arm, sd), G in R.groupby(["arm", R.seed.astype(str)]):
    for c in ("R3", "SG2", "EA2"): rows.append(dict(val="EV", cand=f"{arm} {c}", seed=sd, rmse=rm(G[c] - G.y)))
    if arm == "BASE":
        gr = np.array([GE.at[r, f"h{h}"] for r, h in zip(G.rec, G.hour)]); rows.append(dict(val="EV", cand="BASE SG2 + GRU10", seed=sd, rmse=rm(.9 * G.SG2 + .1 * gr - G.y)))
S = pd.DataFrame(rows).groupby(["val", "cand"]).rmse.mean().reset_index(); pd.set_option("display.width", 200)
for v, g in S.groupby("val"): print(f"\n{v} (pass-2 truly-normal rows, seed-mean RMSE):"); print(g.sort_values("rmse")[["cand", "rmse"]].round(4).to_string(index=False))
