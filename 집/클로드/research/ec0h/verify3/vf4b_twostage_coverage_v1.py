# VF4b (서술): 2단계 모델(vf3b) 0시 확률로 본 평가 60기록 — 1등 후보가 정답 있는 학습 기록이고 P0 >= .8 인데 지금 앵커가 없는 기록,
# 그리고 1등 후보 23시 EC 와 현재 0시 예측의 차이
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
B = pd.read_csv("vf3b_eval_v2.csv").set_index("rec")
C = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_confident_1step_anchor_v7.csv")).set_index("row")
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["rec"] = Y.row_id.str[:7]; Y["h"] = Y.row_id.str[8:].astype(int); EC23 = Y[Y.h == 23].set_index("rec").sub_ec
B["top_lab"] = B.top0.isin(EC23.index); B["top_ec23"] = B.top0.map(EC23); B["anch"] = B.index.isin(C.index)
print("records with top0 labelled & Ptop0 >= .8:", int((B.top_lab & (B.Ptop0 >= .8)).sum()), "| of which currently anchored:", int((B.top_lab & (B.Ptop0 >= .8) & B.anch).sum()))
print("bands of Ptop0 (labelled top):", pd.cut(B[B.top_lab].Ptop0, [0, .3, .5, .8, 1]).value_counts().sort_index().to_dict())
cols = ["top0", "Ptop0", "Phid0", "top_ec23", "pred0", "anch", "cand", "p_v7"]
print(B[B.top_lab & (B.Ptop0 >= .5)].assign(gap=lambda d: d.top_ec23 - d.pred0)[cols + ["gap"]].round(3).sort_values("gap", key=abs, ascending=False).to_string())
