# -*- coding: utf-8 -*-
"""DGX3 (2026-10-16 집 클로드; 사용자 '49개 나머지 온실로 실험했던 내용들을 DGX1·DGX2 처럼 분해해 봐').  서술 진단(판정 없음), 실행 전 커밋.  온도.
저장된 예측으로 다른 온실 실험들을 같은 잣대로 분해:  기준 = tt1_ckpt(DIAG10·EXT10·EXT12·EL1; base_REF 7/101, codex_REF, pfn A/B),
  셀 = .4 BASE + (.2 + .4(1−g)) CODEX + .4 g PFN, g = clip((in_temp − 8)/2) (결측 1) — TT 실험들의 판정 셀과 같음.
  변형(CODEX 를 바꿔 끼움, 전 행 = ALL 범위):  TT15_A1 F41 전 행 .3 + 수준 어댑터 | TT15_A2 F41 흐름 계수 비대칭 필터 특징 | TT16_HP202·HP303 49곳 사전학습 예측 특징
  | TT19_J 닮은 4곳(F30·F38·F45·F50) .3 + 수준 어댑터 | TT20_J.02·J.05 같은 4곳 낮은 가중.
이득 = 4셀(시드 7·101 x PFN A·B) 평균 SSE 변화(음수 = 좋음), 거친 날 제외.
 [1] 실험 x 검증기 합계·상대 %·상위 3기록 몫·빼고 남는 이득·좋아진/나빠진 기록
 [2] 실내온도 구간(<=4, 4-6, 6-8, 8-10, 10-12, 12-15, 15-20, >20) 행당 이득 [3] 난방 켬/끔, 계절, 농장, 시각대, 2차
 [4] 하루 수준(기록 평균) 몫 vs 하루 안 모양 몫 [5] CODEX 단독(변형 vs 기준) 구간별 RMSE·편향, 바뀐 양(변형 − 기준) 평균
 [6] 같은 기록의 검증기 간 부호 일치(기록 이득 상관) [7] 조각 표: 구간 x 난방 칸마다 '모든 실험 x 검증기에서 같은 방향' 인지 — 뽑아 쓸 만한 조각 찾기
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dgx3_other_farm_experiments_dissection_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); L_ = env.LOCAL
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
G = pd.concat([pd.read_csv(os.path.join(L_, "tt1_ckpt", f)) for f in sorted(os.listdir(os.path.join(L_, "tt1_ckpt")))], ignore_index=True)
EXP = {"TT15_A1": ("tt15", "codex_A1"), "TT15_A2": ("tt15", "codex_A2"), "TT16_HP202": ("tt16", "codex_pre_202"), "TT16_HP303": ("tt16", "codex_pre_303"),
       "TT19_J": ("tt19", "codex_J"), "TT20_J.02": ("tt20", "codex_J_0.02"), "TT20_J.05": ("tt20", "codex_J_0.05")}
for d in sorted(set(v[0] for v in EXP.values())):
    C = pd.concat([pd.read_csv(os.path.join(L_, d + "_ckpt", f)) for f in sorted(os.listdir(os.path.join(L_, d + "_ckpt")))], ignore_index=True).drop_duplicates(["validator", "row_id"])
    G = G.merge(C, on=["validator", "row_id"], how="left")
G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].reset_index(drop=True)
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_heating"]); G = G.merge(X, on="row_id", how="left")
G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1)); y = G.sub_temp.values
G["rec"] = G.row_id.str[:7]; G["heat"] = np.where(G.act_heating > 0, "on", "off")
G["band"] = pd.cut(G.in_temp, [-99, 4, 6, 8, 10, 12, 15, 20, 99], labels=["<=4", "4-6", "6-8", "8-10", "10-12", "12-15", "15-20", ">20"])
G["season"] = pd.cut(G.day, [0, 110, 160, 179, 400], labels=["<=110", "111-160", "161-178", "pass-2"]); G["hgrp"] = pd.cut(G.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
def cell(s, f, c): return (.4 * G["base_REF_%d" % s] + (.2 + .4 * (1 - G.g)) * G[c] + .4 * G.g * G["pfn_%s" % f]).values
REFC = [cell(s, f, "codex_REF") for s in (7, 101) for f in "AB"]; e0 = np.mean(REFC, 0) - y
for nm, (_, col) in EXP.items():
    V = [cell(s, f, col) for s in (7, 101) for f in "AB"]
    G["gain_" + nm] = np.mean([(a - y) ** 2 - (b - y) ** 2 for a, b in zip(V, REFC)], 0); G["d_" + nm] = np.mean(V, 0) - np.mean(REFC, 0)
G["e0"] = e0; NAMES = list(EXP); VALS = ["DIAG10", "EXT10", "EXT12", "EL1"]; pd.set_option("display.width", 260)
print("\n[1] totals (sum gain; rel % of cell RMSE; top3 share; without top3; records better/worse)")
rows = []
for nm in NAMES:
    for v in VALS:
        S = G[G.validator == v]; R = S.groupby("rec")["gain_" + nm].sum(); top = R.abs().nlargest(3).index
        rows.append(dict(exp=nm, val=v, rel=100 * (np.sqrt((S.e0 ** 2).sum() + S["gain_" + nm].sum()) / np.sqrt((S.e0 ** 2).sum()) - 1),
                         gain=S["gain_" + nm].sum(), top3_share=R.abs().nlargest(3).sum() / max(R.abs().sum(), 1e-9), without_top3=R.drop(top).sum(), better=(R < -1e-9).mean(), worse=(R > 1e-9).mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
def per_row(col):
    T = G.groupby(["validator", col], observed=True)[["gain_" + n for n in NAMES]].mean() * 1000; T.columns = NAMES; return T
for col, title in (("band", "[2] gain per row x1000 by in_temp band"), ("heat", "[3a] by heating"), ("season", "[3b] by season"), ("farm", "[3c] by farm"), ("hgrp", "[3d] by hour group")):
    print(f"\n{title}"); T = per_row(col); n = G.groupby(["validator", col], observed=True).size().rename("rows"); print(pd.concat([n, T.round(2)], axis=1).to_string())
print("\n[4] level (record-mean) vs shape part of gain, per experiment x validator")
rows = []
for nm in NAMES:
    for v in VALS:
        S = G[G.validator == v]; n = S.groupby("rec").size(); e = S.groupby("rec").e0.mean(); dm = S.groupby("rec")["d_" + nm].mean()
        lev = float((n * ((e + dm) ** 2 - e ** 2)).sum()); tot = float(((S.e0 + S["d_" + nm]) ** 2 - S.e0 ** 2).sum()); rows.append(dict(exp=nm, val=v, total=tot, level=lev, shape=tot - lev))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\n[5] CODEX alone: variant vs reference by band (all validators pooled): RMSE, bias, mean change (variant - ref)")
rows = []
for b, S in G.groupby("band", observed=True):
    r = dict(band=b, rows=len(S), REF_rmse=rm(S.codex_REF - S.sub_temp), REF_bias=float((S.codex_REF - S.sub_temp).mean()))
    for nm, (_, col) in EXP.items(): r[nm + "_rmse"] = rm(S[col] - S.sub_temp); r[nm + "_dchg"] = float((S[col] - S.codex_REF).mean())
    rows.append(r)
print(pd.DataFrame(rows).round(3).T.to_string())
print("\n[6] record-gain agreement across validators (corr of per-record gain; same-sign share for records with |gain| > .2)")
for nm in NAMES:
    P = G.groupby(["validator", "rec"])["gain_" + nm].sum().unstack(0); out = []
    for a in VALS:
        for b in VALS:
            if a < b:
                Q = P[[a, b]].dropna(); Q = Q[Q.abs().max(axis=1) > .2]
                if len(Q) >= 5: out.append(f"{a}-{b} n{len(Q)} same {np.mean(np.sign(Q[a]) == np.sign(Q[b])):.2f} r {Q[a].corr(Q[b]):.2f}")
    print(f"  {nm}: " + " | ".join(out))
print("\n[7] slice table: band x heat - sign of summed gain in each experiment x validator; 'consistent' = all cells with >= 15 rows same sign")
rows = []
for (b, h), S in G.groupby(["band", "heat"], observed=True):
    signs, tot = [], {}
    for nm in NAMES:
        for v in VALS:
            T = S[S.validator == v]
            if len(T) >= 15: signs.append(np.sign(T["gain_" + nm].sum()))
        tot[nm] = 1000 * S["gain_" + nm].mean()
    if not signs: continue
    rows.append(dict(band=b, heat=h, rows=len(S), cells=len(signs), better=int(sum(x < 0 for x in signs)), worse=int(sum(x > 0 for x in signs)), **{k: round(x, 1) for k, x in tot.items()}))
print(pd.DataFrame(rows).to_string(index=False))
G.to_csv(os.path.join(HERE, "dgx3_rows_v1.csv.gz"), index=False, compression="gzip")
