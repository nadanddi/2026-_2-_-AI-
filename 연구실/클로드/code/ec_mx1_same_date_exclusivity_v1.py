# -*- coding: utf-8 -*-
"""MX1 — 같은 날짜 묶음 고EC 배타성 보정 (계획: 연구실/클로드/문서/MX1_같은날짜_고EC배타_보정_계획_v1.md, 실행 전 고정) — 2026-10-07 연구실 클로드
h시 행: pm_h = mean(p_0..h) ≥ .9 이고, 외기 0..h 쌍둥이(z-RMSE ≤ .05, h ≥ 5)인 '보이는 정답 짝' 중
  ≥1.2 있으면 F1(내리기), 짝 있으나 없으면 F0(올리기). δ_F = 교차적합 평균 하루 잔차(축소, |δ|≤.5).
안 D(F1만), U(F0만), B(둘 다). 검증기 DIAG10(360일)·EL1·P2LOO(2차), 시드 47/1414/6464.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; SEEDS = (47, 1414, 6464); GATE, HI, TW = .9, 1.2, .05
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])]
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int); YM = Y.groupby(["farm", "day"]).sub_ec.mean()
isT = X.groupby(["farm", "day"]).t.first()
# 시각별 z (학습 기록 기준), 배열 [기록, 시각, 변수]
Z = []
for v in W:
    P = X.pivot_table(index=["farm", "day"], columns="hour", values=v)
    mu, sd = P[isT[P.index] == 0].mean(), P[isT[P.index] == 0].std()
    Z.append(((P - mu) / sd).values)
KEYS = list(X.pivot_table(index=["farm", "day"], columns="hour", values=W[0]).index); KI = {k: i for i, k in enumerate(KEYS)}
Z = np.stack(Z, axis=2)  # n×24×4
LAB = [k for k in KEYS if k in YM.index]; LI = np.array([KI[k] for k in LAB]); LY = YM[LAB].values
# 시각 인과 쌍둥이 거리: D[h][i, j] = sqrt(mean_{0..h, v} (z_i - z_lab_j)^2)
sq = (Z[:, None, :, :] - Z[None, LI, :, :]) ** 2  # n × nl × 24 × 4
cs = np.nancumsum(np.nanmean(sq, axis=3), axis=2); cnt = np.cumsum(np.isfinite(np.nanmean(sq, axis=3)), axis=2)
DIST = np.sqrt(cs / np.maximum(cnt, 1))  # n × nl × 24
del sq
# 기준 예측
O = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv"))
RV = pd.read_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv")); RV = RV[RV.tag == "CUR"]
def base(validator):
    """반환: DataFrame(row_id, farm, day, hour, y, fold, p_47, p_1414, p_6464)"""
    if validator == "DIAG10":
        o = O[O.validator == "DIAG10"].copy()
        for s in SEEDS:
            cur = np.clip(.8 * (.6 * o["et_%d" % s] + .3 * o["lgb_%d" % s] + .1 * o["mlp_%d" % s]) + .2 * o.pfn, o.lo, o.hi)
            sg = RV[(RV.validator == "DIAG10") & (RV.seed == s)].set_index("row_id").sg
            o["p_%d" % s] = o.row_id.map(sg).fillna(cur).values
        o["fold"] = o.validation_fold
    else:
        r = RV[RV.validator == validator]
        o = r[r.seed == SEEDS[0]][["row_id", "fold"]].copy()
        for s in SEEDS:
            o["p_%d" % s] = o.row_id.map(r[r.seed == s].set_index("row_id").sg).values
        o["sub_ec"] = np.nan
    o["farm"], o["day"], o["hour"] = o.row_id.str[:3], o.row_id.str[4:7].astype(int), o.row_id.str[8:10].astype(int)
    o["y"] = Y.set_index("row_id").sub_ec.reindex(o.row_id).values
    return o.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
D10 = base("DIAG10"); F10 = D10.groupby(["farm", "day"]).fold.first().to_dict()
def flags(df, s, hidden_of):
    """행별 표지: 0=짝 없음/게이트 밖, 1=F1, 2=F0. hidden_of(key) → 그 기록 기준 가림 집합(정답 짝 제외용)"""
    df = df.copy(); df["pm"] = df.groupby(["farm", "day"])["p_%d" % s].transform(lambda x: x.expanding().mean())
    fl = np.zeros(len(df), dtype=int)
    for (f, d), g in df.groupby(["farm", "day"]):
        k = (f, d); i = KI[k]; hid = hidden_of(k)
        vis = np.array([(m != k) and (m not in hid) for m in LAB])
        for ri, h, pm in zip(g.index, g.hour, g.pm):
            if h < 5 or pm < GATE:
                continue
            mt = vis & (DIST[i, :, h] <= TW)
            if not mt.any():
                continue
            fl[ri] = 1 if (LY[mt] >= HI).any() else 2
    return fl
# 학습용(δ 적합) 표지: DIAG10 OOF, 각 기록 자기 DIAG10 폴드 가림
TRF = {s: flags(D10, s, lambda k: {m for m in LAB if F10.get(m) == F10.get(k)}) for s in SEEDS}
D10["yd"] = D10.groupby(["farm", "day"]).y.transform("mean")
def delta(s, excl_keys):
    out = {}
    pdm = D10.groupby(["farm", "day"])["p_%d" % s].transform("mean")
    r = (D10.yd - pdm).values; ok = ~np.array([(f, d) in excl_keys for f, d in zip(D10.farm, D10.day)])
    for F in (1, 2):
        m = ok & (TRF[s] == F)
        out[F] = float(np.clip(r[m].sum() / (m.sum() + 24 * 5), -.5, .5))
    return out
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = {}; keep = {}
for V in ("DIAG10", "EL1", "P2LOO"):
    df = D10 if V == "DIAG10" else base(V)
    fmap = df.groupby(["farm", "day"]).fold.first().to_dict()
    hidden_of = lambda k: {m for m in LAB if fmap.get(m, -99) == fmap.get(k)}
    for s in SEEDS:
        fl = flags(df, s, hidden_of)
        corr = {c: np.zeros(len(df)) for c in "DUB"}
        for fo in sorted(set(df.fold)):
            te = (df.fold == fo).values
            hk = set(zip(df.farm[te], df.day[te]))
            # 계획 비평 반영: 가림 기록의 같은 날짜 쌍둥이(다른 ID 포함)도 제외, 그리고 모두의 ±1 기록일
            tw = {LAB[j] for k in hk for j in np.where(DIST[KI[k], :, 23] <= TW)[0]}
            ex = {(f, d + j) for f, d in hk | tw for j in (-1, 0, 1)}
            dl = delta(s, ex)
            corr["D"][te] = np.where(fl[te] == 1, dl[1], 0); corr["U"][te] = np.where(fl[te] == 2, dl[2], 0)
            corr["B"][te] = corr["D"][te] + corr["U"][te]
        p = df["p_%d" % s].values; y = df.y.values; yd = df.groupby(["farm", "day"]).y.transform("mean").values
        nm = yd < 1; late = df.day.values >= 179
        rec = {"base": (rm(p - y), rm((p - y)[nm]), rm((p - y)[late]))}
        for c in "DUB":
            q = p + corr[c]; rec[c] = (rm(q - y), rm((q - y)[nm]), rm((q - y)[late]))
        rec["fires"] = (int((fl == 1).sum()), int((fl == 2).sum()))
        res[(V, s)] = rec; keep[(V, s)] = (df[["farm", "day", "hour", "y"]].assign(p=p, **{c: p + corr[c] for c in "DUB"}, fl=fl))
        print("%-6s 시드%4d | 기준 %.4f | D %.4f | U %.4f | B %.4f || 일반 기준 %.4f D %.4f U %.4f B %.4f || 2차 기준 %.4f B %.4f | 발동 행 F1 %d F0 %d" % (
            V, s, rec["base"][0], rec["D"][0], rec["U"][0], rec["B"][0], rec["base"][1], rec["D"][1], rec["U"][1], rec["B"][1], rec["base"][2], rec["B"][2], *rec["fires"]), flush=True)
# 판정
for c in "DUB":
    a = all(res[k][c][0] <= res[k]["base"][0] + 1e-12 for k in res) and all(res[("DIAG10", s)][c][0] < res[("DIAG10", s)]["base"][0] for s in SEEDS)
    b = all(res[k][c][1] <= res[k]["base"][1] + 1e-12 for k in res)
    K = [keep[("DIAG10", s)] for s in SEEDS]; Zm = K[0][["farm", "day", "y"]].copy()
    Zm["a"] = np.mean([k.p for k in K], 0); Zm["b"] = np.mean([k[c] for k in K], 0)
    Zm["cl"] = Zm.farm + "_" + (Zm.day // 5).astype(str)
    dd = ((Zm.b - Zm.y) ** 2 - (Zm.a - Zm.y) ** 2).groupby(Zm.cl).sum().values
    idx = np.random.default_rng(20261007).integers(0, len(dd), (20000, len(dd))); pw = float((dd[idx].sum(1) >= 0).mean())
    floor = (1 - (np.abs(dd) > 1e-12).mean()) ** len(dd)
    dday = ((Zm.b - Zm.y) ** 2 - (Zm.a - Zm.y) ** 2).groupby([Zm.farm, Zm.day]).sum(); ydm = Zm.groupby(["farm", "day"]).y.mean()
    ch = dday[np.abs(dday) > 1e-9]
    print("\n[%s] (a) %s (b) %s (c) P(worse) %.4f (묶음 %d, 하한 %.4f) → %s" % (c, a, b, pw, len(dd), floor, "통과" if a and b and pw < .025 / 3 else "미통과"))
    print("   바뀐 날 %d (개선 %d·악화 %d): %s" % (len(ch), (ch < 0).sum(), (ch > 0).sum(), ", ".join("%s_%d(y%.2f)%+.2f" % (f, d, ydm[(f, d)], v) for (f, d), v in ch.sort_values().items())))
# 평가 60 발동(입력만): 제출 14 예측을 p 로, 짝 = 정답 기록 전부
SUB = pd.read_csv(os.path.join(env.ROOT, u"제출", [d for d in os.listdir(os.path.join(env.ROOT, u"제출")) if d.startswith("09")][0], "submission_14.csv"))
SUB = SUB[SUB.row_id.str[:3].isin(["F13", "F47"])].copy(); SUB["farm"], SUB["day"], SUB["hour"] = SUB.row_id.str[:3], SUB.row_id.str[4:7].astype(int), SUB.row_id.str[8:10].astype(int)
SUB = SUB.sort_values(["farm", "day", "hour"]).reset_index(drop=True); SUB["p_0"] = SUB.sub_ec
fl = flags(SUB, 0, lambda k: set()); SUB["fl"] = fl
dl = {s: delta(s, set()) for s in SEEDS}
print("\n평가 60 발동 기록(행 수 F1/F0):", SUB[SUB.fl > 0].groupby(["farm", "day"]).fl.agg(lambda x: "%d/%d" % ((x == 1).sum(), (x == 2).sum())).to_dict())
print("전체 학습 δ(시드별):", dl)
