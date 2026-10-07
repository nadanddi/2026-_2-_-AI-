# -*- coding: utf-8 -*-
"""RF2 — 운영 지문 기반 시각별 인과 하루 잔차 보정기, 9회차 근사 구성 위 (계획 v1 + 계획 비평 반영, 실행 전 고정) — 2026-10-07 연구실 클로드
계획: 연구실/클로드/문서/RF2_하루잔차보정기_계획_v1.md. 비평 반영:
 - 목표 = SG2 이전 잔차 y_day − CUR_day (CUR = 0.8 R3_DP1 + 0.2 PFN, WT1 DIAG10 OOF, 360일) → SG2 의 이웃 정답 누수 차단.
   (R3 OOF 자체가 다른 폴드 정답으로 학습된 점의 약한 누수는 남음 — 한계로 명시)
 - 적용은 2차 행만(1차는 이웃 효과·악화). 두 안: A = CUR + 보정기(SG2 대신), B = CUR + SG2 + 보정기.
 - 대조: C0 = 상수 이동(학습 폴드 2차 평균 잔차), ACT = 구동기 누적 특징만 쓴 보정기.
특징(h 시 행, 0..h 인과): 구동기 7 누적평균·0비율, 실내 3·외기 4 누적평균, h. 보정기 = ExtraTrees(300, leaf 20, max_features .5, seed 0).
보정기 학습 행 = 보정기 폴드 밖의 정답 날(1·2차 모두) 중 ±1 기록일 제외.
폴드: (i) DIAG10 폴드(10), (ii) 연속 블록 — 같은 ID 기록 번호 10개 블록을 분할 시드 s∈{1,2,3}로 5묶음 배정, 같은 날짜 묶음의 다른 ID 기록도 같은 폴드에서 제외.
기준 시드: R3 47/1414/6464 (SG2 출력은 RV2 preds 의 CUR sg).
[고정 기준] A 또는 B 각각 통과 iff
 (a) 기준 시드 3 × 폴드 4종(DIAG10, 블록 s1/s2/s3) = 12칸 모두 2차 시간 RMSE 가 CUR+SG2 보다 낮음
 (b) 같은 12칸 2차 일반 날 악화 ≤ 1%
 (c) 12칸 모두 상수 이동 대조(C0, 같은 적용 방식)보다 낮음
 (d) DIAG10 폴드판 시드평균 2차 온실×5기록 묶음 부트스트랩 P(worse) < .025/2 (A·B 두 안) — 구조적 하한 함께 보고
 함께: 시각대별, 고EC, 바뀐 고EC 날 vs HG3 4일(F13 241·243, F47 229·231), ACT 판, 중요도 상위.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy"); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TR = TR[TR.row_id.str[:3].isin(["F13", "F47"])].copy()
TR["farm"], TR["day"], TR["hour"] = TR.row_id.str[:3], TR.row_id.str[4:7].astype(int), TR.row_id.str[8:10].astype(int)
TR = TR.sort_values(["farm", "day", "hour"]).reset_index(drop=True); gg = TR.groupby(["farm", "day"])
feat = pd.DataFrame({"row_id": TR.row_id})
for v in ACTS:
    feat[v + "_tdm"] = gg[v].transform(lambda s: s.expanding().mean()).values
    feat[v + "_tdz"] = gg[v].transform(lambda s: (s == 0).astype(float).where(s.notna()).expanding().mean()).values
for v in INDOOR + W:
    feat[v + "_tdm"] = gg[v].transform(lambda s: s.expanding().mean()).values
FACT = [c for c in feat.columns if c.startswith("act_")]; FALL = [c for c in feat.columns if c != "row_id"] + ["hour"]
O = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv")); O = O[O.validator == "DIAG10"].copy()
SEEDS = (47, 1414, 6464)
for s in SEEDS:
    r3 = .6 * O["et_%d" % s] + .3 * O["lgb_%d" % s] + .1 * O["mlp_%d" % s]
    O["cur_%d" % s] = np.clip(.8 * r3 + .2 * O.pfn, O.lo, O.hi)
RV2 = pd.read_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv")); RV2 = RV2[(RV2.validator == "DIAG10") & (RV2.tag == "CUR")]
for s in SEEDS:
    m = RV2[RV2.seed == s].set_index("row_id").sg
    O["sg_%d" % s] = O.row_id.map(m).fillna(O["cur_%d" % s])
D = O.merge(feat, on="row_id"); D["yd"] = D.groupby(["farm", "day"]).sub_ec.transform("mean"); D["late"] = D.day >= 179
# 날짜 묶음(다른 ID 같은 날짜 함께 제외용)
WV = TR.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV.mean()) / WV.std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = {k: fd(i) for i, k in enumerate(keys)}
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); f10 = DP[DP.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first().to_dict()
def block_folds(seed):
    rng = np.random.default_rng(seed); out = {}
    for f in ("F13", "F47"):
        ds = sorted(d for ff, d in keys if ff == f); blocks = [ds[i:i + 10] for i in range(0, len(ds), 10)]
        lab = rng.permutation(len(blocks)) % 5
        for b, l in zip(blocks, lab):
            for d in b:
                out[(f, d)] = int(l)
    return out
SPLITS = {"DIAG10": f10, "블록s1": block_folds(1), "블록s2": block_folds(2), "블록s3": block_folds(3)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
res = {}; preds_keep = {}
for sname, fmap in SPLITS.items():
    D["fold"] = [fmap.get((f, d), -1) for f, d in zip(D.farm, D.day)]
    for s in SEEDS:
        cur = D["cur_%d" % s]; sg = D["sg_%d" % s]
        target = D.yd - cur.groupby([D.farm, D.day]).transform("mean")
        rh_all = np.zeros(len(D)); rh_act = np.zeros(len(D)); c0 = np.zeros(len(D))
        for k in sorted(set(D.fold) - {-1}):
            te = (D.fold == k).values
            hid = set(zip(D.farm[te], D.day[te]))
            hid_g = {grp[x] for x in hid}
            near = {(f, d + j) for f, d in hid for j in (-1, 0, 1)}
            tr = np.array([(f, d) not in near and grp[(f, d)] not in hid_g for f, d in zip(D.farm, D.day)])
            for cols, store in ((FALL, rh_all), (FACT, rh_act)):
                mdl = ExtraTreesRegressor(300, min_samples_leaf=20, max_features=.5, random_state=0, n_jobs=4).fit(D.loc[tr, cols].fillna(-9), target[tr])
                store[te] = mdl.predict(D.loc[te, cols].fillna(-9))
            c0[te] = target[tr & D.late.values].mean()
        L = D.late.values
        A_ = np.where(L, cur + rh_all, cur); B_ = np.where(L, sg + rh_all, sg); C0 = np.where(L, sg + c0, sg); ACTb = np.where(L, sg + rh_act, sg)
        q = L; nq = L & (D.yd < 1).values; hq = L & (D.yd >= 1).values
        rec = {}
        for nm, pr in (("SG2", sg), ("A", A_), ("B", B_), ("C0", C0), ("ACT", ACTb)):
            pr = np.asarray(pr); rec[nm] = (rm((pr - D.sub_ec)[q]), rm((pr - D.sub_ec)[nq]), rm((pr - D.sub_ec)[hq]))
        res[(sname, s)] = rec
        if sname == "DIAG10":
            preds_keep[s] = (np.asarray(sg), np.asarray(A_), np.asarray(B_))
        print("%-7s 시드%4d | 2차 SG2 %.4f | A %.4f | B %.4f | 상수 %.4f | 구동기만 %.4f || 일반 SG2 %.4f A %.4f B %.4f || 고EC SG2 %.3f A %.3f B %.3f" % (
            sname, s, rec["SG2"][0], rec["A"][0], rec["B"][0], rec["C0"][0], rec["ACT"][0], rec["SG2"][1], rec["A"][1], rec["B"][1], rec["SG2"][2], rec["A"][2], rec["B"][2]), flush=True)
L = D.late.values
for cand in ("A", "B"):
    a_ok = all(res[k][cand][0] < res[k]["SG2"][0] for k in res)
    b_ok = all(res[k][cand][1] <= 1.01 * res[k]["SG2"][1] for k in res)
    c_ok = all(res[k][cand][0] < res[k]["C0"][0] for k in res)
    sgm = np.mean([preds_keep[s][0] for s in SEEDS], 0); cm = np.mean([preds_keep[s][1 if cand == "A" else 2] for s in SEEDS], 0)
    Z = pd.DataFrame(dict(farm=D.farm, day=D.day, y=D.sub_ec, a=sgm, b=cm))[L]
    Z["cl"] = Z.farm + "_" + (Z.day // 5).astype(str)
    dd = ((Z.b - Z.y) ** 2 - (Z.a - Z.y) ** 2).groupby(Z.cl).agg(["sum", "count"]); sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm))); pw = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    floor = (1 - (np.abs(sm) > 1e-12).mean()) ** len(sm)
    dday = Z.groupby(["farm", "day"]).apply(lambda g_: ((g_.b - g_.y) ** 2).sum() - ((g_.a - g_.y) ** 2).sum())
    hi = Z.groupby(["farm", "day"]).y.mean(); hd = dday[hi[dday.index] >= 1]
    print("\n[%s] (a) 12칸 개선 %s, (b) 일반 날 ≤1%% %s, (c) 상수 대조보다 나음 %s, (d) P(worse) %.4f (묶음 %d, 하한 %.4f) → %s" % (
        cand, a_ok, b_ok, c_ok, pw, len(sm), floor, "통과" if a_ok and b_ok and c_ok and pw < .0125 else "미통과"))
    print("   고EC 날 변화(ΔSSE): %s | 날 단위 개선 %d·악화 %d" % (", ".join("%s_%d %+.2f" % (f, d, v) for (f, d), v in hd.items()), int((dday < 0).sum()), int((dday > 0).sum())))
