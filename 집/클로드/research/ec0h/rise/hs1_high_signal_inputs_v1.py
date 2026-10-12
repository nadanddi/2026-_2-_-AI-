# -*- coding: utf-8 -*-
"""HS1 (2026-10-12 집 클로드; 사용자: '어떤 입력으로 고EC 알아보는지 확인').  서술.
같은 날짜(DM11) 짝 비교: 학습 정답상 고EC(하루평균 ≥ 1)가 정확히 하나인 날(형제 ≥ 1)마다, 변수별 하루 평균(또는 0~5시 평균)을
고EC 기록 vs 같은 날 다른 학습 기록 평균으로 비교.  같은 날이라 날씨·계절은 지워짐.
보고: 변수별 (1) 고EC 쪽이 더 큰 날 비율, (2) 차이 중앙값, (3) '그날 이 변수가 가장 큰(또는 작은) 기록 = 고EC' 적중률.
결측률(농장별)도 표시.  이어서 평가 83~86일 4기록의 같은 변수 값을 그날 다른 기록과 나란히(설명용, 특징으로 쓰지 않음).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)).assign(split=s) for f, s in (("train_X.csv", "train"), ("test_X.csv", "eval"))])
X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["rec"] = Y.row_id.str[:7]; y = Y.groupby("rec").sub_ec.mean()
C = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(C.rec, C.day))
V = [c for c in X.columns if c.startswith(("out_", "in_", "act_"))]
print("missing rate by farm:"); print((X.groupby(X.rec.str[:3])[V].apply(lambda g: g.isna().mean()) * 100).round(0).to_string())
M = X.groupby("rec")[V].mean(); M0 = X[X.h <= 5].groupby("rec")[V].mean().add_suffix("@0-5")
F = M.join(M0); F["day"] = F.index.map(DAY); F["y"] = F.index.map(y); F["split"] = F.index.map(dict(zip(X.rec, X.split)))
T = F[F.split == "train"].dropna(subset=["day"])
rows = []
for d, G in T.groupby("day"):
    if len(G) < 2 or (G.y >= 1).sum() != 1: continue
    hrec = G.y.idxmax(); sib = G.drop(hrec)
    for v in [c for c in F.columns if c not in ("day", "y", "split")]:
        g = G[v]
        if g.isna().any() or g.nunique() < 2: continue
        rows.append(dict(day=d, v=v, diff=G.at[hrec, v] - sib[v].mean(), top=g.idxmax() == hrec, bot=g.idxmin() == hrec))
R = pd.DataFrame(rows)
S = R.groupby("v").agg(days=("day", "nunique"), high_bigger=("diff", lambda s: (s > 0).mean()), med_diff=("diff", "median"), hit_max=("top", "mean"), hit_min=("bot", "mean"))
S["hit"] = S[["hit_max", "hit_min"]].max(1)
pd.set_option("display.width", 220)
print(f"\ndays with exactly one high-EC training record: {R.day.nunique()}")
print(S[S.days >= 10].sort_values("hit", ascending=False).round(3).head(25).to_string())
top = list(S[S.days >= 10].sort_values("hit", ascending=False).head(6).index)
print("\neval days 83-86 (and labelled neighbours) - values of the top variables; * = record predicted high in 12th round:")
P = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); P["rec"] = P.row_id.str[:7]; pm = P.groupby("rec").sub_ec.mean()
E = F[(F.day >= 80) & (F.day <= 90)].copy(); E["pred"] = E.index.map(pm); E["mark"] = np.where(E.pred >= 1, "*", "")
print(E.sort_values(["day"])[["day", "split", "y", "pred", "mark"] + top].round(2).to_string())
