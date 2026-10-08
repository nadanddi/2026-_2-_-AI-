# -*- coding: utf-8 -*-
"""DV — 데이터와 검증 의심 (계획 v2, 실행 전 고정) — 2026-10-08 연구실 클로드
계획: 연구실/클로드/문서/DV_데이터검증_의심_계획_v2.md
DV0 검증기별 2차 RMSE + 날짜 묶음 부트스트랩 구간, 상위 k일 몫
DV1 같은 날 쌍대 비교 Δ = SSE_검증기 − SSE_P2LOO (묶음 부호 뒤집기 순열, 단측, p<.025)
    + Δ1 과 'DIAG10 에서 함께 빠지는 ±3 기록 정답일 수' 스피어만(묶음 순열)
DV2 라벨 이상 점수 S1~S4 (정답·학습 입력만, 1차 97.5% 분위 플래그, 합≥2 = 의심) vs P2LOO SSE
DV3 SG2·HG3 의 P2LOO 날별 기여 상위 5일 (서술)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
RNG = np.random.default_rng(20261008); NB = 20000; LB = .1384
SEEDS = (47, 1414, 6464)

# ---------- 원자료 ----------
TX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TY = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].merge(TY, on="row_id", how="left")
TX["farm"], TX["day"], TX["hour"] = TX.row_id.str[:3], TX.row_id.str[4:7].astype(int), TX.row_id.str[8:10].astype(int)
TX = TX.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
# 날짜 묶음 (RF2 와 같은 union-find, 거리 ≤ .05)
WV = TX.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV.mean()) / WV.std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
GRP = {k: fd(i) for i, k in enumerate(keys)}

# ---------- 예측 ----------
RV = pd.read_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv")); RV = RV[RV.tag == "CUR"]
P = RV.groupby(["validator", "row_id", "farm", "day", "hour"]).agg(y=("y", "first"), sg=("sg", "mean"), hg=("hg", "mean"), dm=("dm", "mean")).reset_index()
PS = RV.copy()
def day_sse(df, col):
    return ((df[col] - df.y) ** 2).groupby([df.farm, df.day]).sum()
def boot_rmse(sse_day, rows_day, grp):
    g = np.array([grp[k] for k in sse_day.index]); ug = np.unique(g)
    S = pd.Series(sse_day.values).groupby(g).sum().reindex(ug).values; N = pd.Series(rows_day.values).groupby(g).sum().reindex(ug).values
    idx = RNG.integers(0, len(ug), (NB, len(ug)))
    r = np.sqrt(S[idx].sum(1) / N[idx].sum(1)); return np.percentile(r, [2.5, 97.5])

print("=== DV0 검증기별 2차 RMSE (시드 평균 예측 sg) ===")
for v in ["P2LOO", "DIAG10", "EL1"]:
    d = P[P.validator == v]; s = day_sse(d, "sg"); n = d.groupby([d.farm, d.day]).size()
    rm = np.sqrt(s.sum() / n.sum()); lo, hi = boot_rmse(s, n, GRP)
    per = [np.sqrt(day_sse(PS[(PS.validator == v) & (PS.seed == sd)], "sg").sum() / n.sum()) for sd in SEEDS]
    ss = s.sort_values(ascending=False); sh = {k: ss.iloc[:k].sum() / ss.sum() for k in (1, 2, 4, 8)}
    print(f"{v:7s} 날 {len(s)} RMSE {rm:.4f} [95% {lo:.4f}, {hi:.4f}] 시드별 {[round(x,4) for x in per]} | LB .1384 구간 {'안' if lo <= LB <= hi else '밖'} | 상위 k일 몫 " + " ".join(f"{k}:{sh[k]:.2f}" for k in sh))
    print("        상위 4일:", [(f"{a}_{b}", round(c, 2)) for (a, b), c in ss.iloc[:4].items()])
    for k in (4, 8):
        rest = ss.iloc[k:].sum() / (n.sum() - 24 * k); print(f"        상위 {k}일 제외 RMSE {np.sqrt(rest):.4f}")

print("\n=== DV1 같은 날 쌍대 비교 (시드 평균 예측) ===")
S = {v: day_sse(P[P.validator == v], "sg") for v in ["P2LOO", "DIAG10", "EL1"]}
days = S["P2LOO"].index; G = np.array([GRP[k] for k in days]); ug = np.unique(G)
def signflip(delta):
    D = pd.Series(delta).groupby(G).sum().reindex(ug).values; obs = D.sum()
    sg = RNG.choice([-1, 1], (NB, len(ug))); null = (sg * D).sum(1); return obs, (null >= obs).mean()
# DIAG10 폴드 (360일)
WT = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv")); F10 = WT[WT.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first().to_dict()
lab = TX[TX.sub_ec.notna()].groupby(["farm", "day"]).size().index
labset = set(lab)
def comask(k):
    f, d = k; fo = F10.get(k); c = 0
    for dd in range(d - 3, d + 4):
        if dd != d and (f, dd) in labset and fo is not None and F10.get((f, dd)) == fo: c += 1
    return c
CM = np.array([comask(k) for k in days])
res = []
for v, nm in (("DIAG10", "Δ1"), ("EL1", "Δ2")):
    dl = (S[v].reindex(days) - S["P2LOO"]).values; obs, p = signflip(dl)
    print(f"{nm} = SSE_{v} − SSE_P2LOO 합 {obs:+.3f} (날 {len(dl)}, 묶음 {len(ug)}) 단측 p {p:.4f} → {'유의' if p < .025 else '유의 아님'} | 양수 날 {int((dl>0).sum())}·음수 {int((dl<0).sum())}")
    res.append(dl)
d1 = res[0]; rho = spearmanr(d1, CM).correlation
# 묶음 순열: CM 을 묶음 단위로 섞음 (묶음 크기 같은 것끼리 섞는 대신 날 단위 순열을 묶음 블록으로)
null = []
for _ in range(5000):
    perm = RNG.permutation(ug); mp = dict(zip(ug, perm))
    # 묶음 라벨을 바꾸고 각 날에 새 묶음의 CM 평균을 붙임
    cmg = pd.Series(CM).groupby(G).mean(); newcm = np.array([cmg[mp[g]] for g in G]); null.append(spearmanr(d1, newcm).correlation)
null = np.array(null); print(f"Δ1 ~ 함께 빠지는 ±3 기록 정답일 수: 스피어만 {rho:+.3f}, 묶음 순열 단측 p {(null >= rho).mean():.4f}; 함께 빠짐 분포 {np.bincount(CM).tolist()}")
for c in sorted(set(CM)):
    m = CM == c; print(f"   함께 빠짐 {c}: 날 {m.sum()} 평균 Δ1 {d1[m].mean():+.3f}")
o = np.argsort(-d1); yd = P[P.validator == "P2LOO"].groupby(["farm", "day"]).y.mean().reindex(days).values
print("Δ1 상위 5:", [(f"{days[i][0]}_{days[i][1]}", round(d1[i], 2), round(yd[i], 2), int(CM[i])) for i in o[:5]])
print("Δ1 하위 5:", [(f"{days[i][0]}_{days[i][1]}", round(d1[i], 2), round(yd[i], 2), int(CM[i])) for i in o[-5:]])

print("\n=== DV2 라벨 이상 점수 (정답만) ===")
L = TX[TX.sub_ec.notna()].copy(); L["lec"] = np.log(L.sub_ec.clip(lower=.01))
rows = []
for (f, d), g in L.groupby(["farm", "day"]):
    g = g.sort_values("hour"); le = g["lec"].values; st = g.sub_temp.values
    s1 = np.nan
    if len(g) >= 12 and np.nanstd(st) > .3:
        b = np.polyfit(st[~np.isnan(st)], le[~np.isnan(st)], 1)[0]; s1 = b
    dl = np.abs(np.diff(le)); s2 = np.nanmax(dl) if len(dl) else np.nan
    s3 = abs(le[-1] - le[0]) / (np.nanstd(le) + 1e-6)
    rows.append((f, d, s1, s2, s3, np.nanmean(le), g.sub_ec.mean()))
Q = pd.DataFrame(rows, columns=["farm", "day", "slope", "S2", "S3", "lem", "ym"])
med = Q.slope.median(); Q["S1"] = (Q.slope - med).abs()
def s4(r):
    nb = Q[(Q.farm == r.farm) & (Q.day != r.day) & ((Q.day - r.day).abs() <= 2)]
    return (nb.lem - r.lem).abs().min() if len(nb) else np.nan
Q["S4"] = Q.apply(s4, axis=1); Q["p2"] = Q.day >= 179
thr = {s: Q.loc[~Q.p2, s].quantile(.975) for s in ["S1", "S2", "S3", "S4"]}
for s in thr: Q["f" + s] = (Q[s] > thr[s]).astype(int)
Q["nflag"] = Q[["fS1", "fS2", "fS3", "fS4"]].sum(axis=1); Q["susp"] = Q["nflag"] >= 2
print(f"기울기 중앙값 {med:.4f}/℃; 문턱(1차 97.5%) " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()))
print(f"날 {len(Q)} (1차 {(~Q.p2).sum()}, 2차 {Q.p2.sum()}); 의심 날 1차 {Q[~Q.p2].susp.sum()}, 2차 {Q[Q.p2].susp.sum()}")
print("의심 날:", [(f"{r.farm}_{r.day}", round(r.ym, 2), int(r.nflag)) for r in Q[Q.susp].itertuples()])
hi = Q.ym >= 1.2; print(f"고EC(≥1.2) 날 {hi.sum()} 중 의심 {int((hi & Q.susp).sum())} ({(hi & Q.susp).sum()/max(hi.sum(),1):.2f}) vs 나머지 {(~hi & Q.susp).sum()/(~hi).sum():.2f}")
print("2차 플래그별 개수:", {s: int(Q.loc[Q.p2, "f" + s].sum()) for s in thr})
q2 = Q.set_index(["farm", "day"]).reindex(days); sp = S["P2LOO"].values; su = q2.susp.fillna(False).values.astype(bool)
if su.sum():
    obs = sp[su].mean() / sp[~su].mean(); null = []
    for _ in range(NB // 4):
        perm = RNG.permutation(ug); sus_g = pd.Series(su).groupby(G).max(); mp = dict(zip(ug, perm))
        m = np.array([sus_g[mp[g]] for g in G]).astype(bool)
        if m.sum() and (~m).sum(): null.append(sp[m].mean() / sp[~m].mean())
    print(f"2차 46일: 의심 {su.sum()}일 평균 P2LOO SSE {sp[su].mean():.2f} vs 나머지 {sp[~su].mean():.2f} (비 {obs:.2f}), 묶음 순열 단측 p {(np.array(null) >= obs).mean():.4f}")
print("2차 46일 플래그 합 ~ P2LOO SSE 스피어만:", round(spearmanr(q2["nflag"].fillna(0), sp).correlation, 3))

print("\n=== DV3 날 영향 서술 (P2LOO, 시드 평균) ===")
# dm 은 정답 하루 평균(RV2 74행)이므로 SG2 이전 CUR 은 WT1 에서 다시 계산 (시드 평균)
d = P[P.validator == "P2LOO"].copy(); wt = WT[WT.validator == "P2LOO"].set_index("row_id")
cur = np.mean([np.clip(.8 * (.6 * wt["et_%d" % s] + .3 * wt["lgb_%d" % s] + .1 * wt["mlp_%d" % s]) + .2 * wt.pfn, wt.lo, wt.hi) for s in SEEDS], axis=0)
d["cur"] = d.row_id.map(pd.Series(cur, index=wt.index))
for nm, a, b in (("SG2 (cur→sg)", "cur", "sg"), ("HG3 (sg→hg)", "sg", "hg")):
    if d[a].isna().any() or d[b].isna().any():
        print(nm, "열 결측 — 생략"); continue
    ch = (day_sse(d, b) - day_sse(d, a)).sort_values(); tot = ch.sum()
    top = ch.iloc[:5]
    print(f"{nm}: 총 ΔSSE {tot:+.3f}; 이득 상위 5일 {[(f'{f}_{dd}', round(x,2), 'S' if q2.loc[(f,dd),'susp'] else '') for (f,dd),x in top.items()]} (몫 {top.sum()/tot:.2f}); 손해 상위 3 {[(f'{f}_{dd}', round(x,2)) for (f,dd),x in ch.iloc[::-1][:3].items()]}")

out = pd.DataFrame({"farm": [k[0] for k in days], "day": [k[1] for k in days], "grp": G, "sse_p2loo": S["P2LOO"].values,
                    "sse_diag10": S["DIAG10"].reindex(days).values, "sse_el1": S["EL1"].reindex(days).values, "comask": CM, "ymean": yd})
out = out.merge(Q[["farm", "day", "slope", "S1", "S2", "S3", "S4", "nflag", "susp"]], on=["farm", "day"], how="left")
out.to_csv(os.path.join(R, "..", "results", "ec_dv_days_v1.csv"), index=False)
Q.to_csv(os.path.join(R, "..", "results", "ec_dv_label_scores_v1.csv"), index=False)
print("저장 완료")
