# -*- coding: utf-8 -*-
"""EMAP1b (2026-10-16 집 클로드).  서술, 실행 전 커밋.  온도.  EMAP1 기록 표(emap1_records_v1.csv)에서 센서 이상 기록(SNF1 표시: 직선 보간·값 멈춤·실내 불일치 ANY, 학습 전용 현상 7.005)을 빼고
같은 평가 구성 가중 오차 지도를 다시 계산 — 평가엔 그런 기록이 없으므로(7.005) 평가 몫 추정이 이상 기록에 끌려가는지 확인.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u emap1b_clean_records_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(HERE, "emap1_records_v1.csv")); EV = pd.read_csv(os.path.join(HERE, "emap1_eval_map_v1.csv"))
F = pd.read_csv(os.path.join(HERE, "snf1_flags_v1.csv")); F["rec"] = F.row_id.str[:7]; bad = set(F[(F.set == "train") & F.ANY].rec)
pd.set_option("display.width", 220); print("validator records flagged as sensor anomaly:", int(D.rec.isin(bad).sum()), "of", len(D), "| unique", D[D.rec.isin(bad)].rec.nunique())
for tag, X in (("all", D), ("clean", D[~D.rec.isin(bad)])):
    g = X.groupby(["layout", "band", "status"]).agg(sse=("sse", "mean"), lev=("sse_level", "mean"), shp=("sse_shape", "mean")).reset_index()
    C = g.groupby(["band", "status"])[["sse", "lev", "shp"]].mean().reset_index(); Bd = X.groupby("band")[["sse", "sse_level", "sse_shape"]].mean()
    ec = EV[["band", "status", "n_eval"]].merge(C, on=["band", "status"], how="left")
    for i in ec.index[ec.sse.isna()]: ec.loc[i, ["sse", "lev", "shp"]] = Bd.loc[ec.at[i, "band"]].values
    tot = (ec.n_eval * ec.sse).sum(); ec["share"] = ec.n_eval * ec.sse / tot; ec["share_level"] = ec.n_eval * ec.lev / tot; ec["share_shape"] = ec.n_eval * ec.shp / tot; ec["rec_rmse"] = np.sqrt(ec.sse / 24)
    print(f"\n[{tag}] eval-reweighted RMSE {np.sqrt(tot / (24 * ec.n_eval.sum())):.3f} | level share {(ec.n_eval * ec.lev).sum() / tot:.2f}")
    print(ec[["band", "status", "n_eval", "rec_rmse", "share", "share_level", "share_shape"]].sort_values("share", ascending=False).round(3).to_string(index=False))
