# -*- coding: utf-8 -*-
"""AR1b (2026-10-13 집 클로드; AR1 후속 서술): 정답을 '자정 EC 차 > .10' 대신 '앵커가 모델보다 0시를 더 못 맞힘(|A − y0| > |sg0 − y0|)' 으로, 그리고 손해 크기((A−y0)² − (sg0−y0)²) 기준 거르기 효과."""
import os, env, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "linker_v4")); from feat4 import EC, FARM, NUM, N
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T = pd.read_csv(os.path.join(env.LOCAL, "ar1_pairs.csv")); idx = {f"{FARM[i]}_{NUM[i]:03d}": i for i in range(N)}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10") & (O.hour == 0)]
SG0 = O.assign(rec=O.row_id.str[:7]).groupby("rec").sg.mean()
A = np.array([EC[idx[p], 23] for p in T.prev]); y0 = np.array([EC[idx[r], 0] for r in T.rec]); s0 = SG0.reindex(T.rec).values
T["d0"] = (A - y0) ** 2 - (s0 - y0) ** 2; T["harm"] = (T.d0 > 0).astype(int)
print(f"pairs {len(T)} | harmful at 0h {T.harm.sum()} | sum gain of helpful {T.d0[T.d0 < 0].sum():.2f} | sum loss of harmful {T.d0[T.d0 > 0].sum():.2f}")
FEAT = ["p", "by_v7", "by_v12", "both", "hp2_agree", "date_ok", "cross_id", "jT", "jH", "jC", "fp", "gapA"]
print("single AUC:", {f: round(roc_auc_score(T.harm, T[f]), 3) for f in FEAT})
for nm, F in (("with cross_id", FEAT), ("without cross_id", [f for f in FEAT if f != "cross_id"])):
    oof = np.zeros(len(T))
    for g in T.blk.unique():
        tr, te = (T.blk != g).values, (T.blk == g).values
        Z = T[F].values.astype(float); mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        oof[te] = LogisticRegression(C=.5, max_iter=2000).fit((Z[tr] - mu) / sd, T.harm[tr]).predict_proba((Z[te] - mu) / sd)[:, 1]
    print(f"\n{nm}: block-CV AUC {roc_auc_score(T.harm, oof):.3f}")
    for q in (.5, .7, .8, .9):
        drop = oof > np.quantile(oof, q)
        print(f"   drop top {100 * (1 - q):.0f}% most suspicious ({drop.sum()} pairs): 0h ΔSSE kept {T.d0[~drop].sum():+.2f} (all kept {T.d0.sum():+.2f}) | removed harmful {int(T.harm[drop].sum())} helpful {int((1 - T.harm[drop]).sum())}")
pd.set_option("display.width", 200); print(T.sort_values("d0", ascending=False).head(8)[["rec", "prev", "d0", "gapA", "cross_id", "jC", "hp2_agree"]].round(3).to_string(index=False))
