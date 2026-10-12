# -*- coding: utf-8 -*-
"""EXC1 (2026-10-13 집 클로드; 사용자: 'F13_237 규칙화해봐').  사전 고정(이 파일 커밋 후 실행).  EC 만.
근거: HX8(6.778) F13_237 = 같은 날 고EC 동은 학습 F47_231 → F13_237 은 고EC 동 아님; 6.434 MX1(연구실) 은 같은 원리 'D 내리기'를 1차에서 기각,
  F13_235·237 은 '2차 검증 사례 0건' 으로 남김. 서술(이 파일 작성 전 확인): 2차 학습 기록 중 고EC 쌍둥이 있는 9개 모두 정답 보통(≤ .84)이나 모델도 모두 낮게 예측;
  모델 높음(≥1.0) + 보이는 고EC 쌍둥이 = DIAG10 1차 6기록(모두 정답 < 예측), P2LOO·EL1 0.  → 아래 판정은 발동 없는 칸 때문에 FAIL 이 예상됨(밝혀 둠).
규칙(h 시 행, 규정: 대상 자기 0..h시 입력 + 학습 기록): 쌍둥이 = 외기 4열 0..h시(소수 1자리)가 대상과 같은 학습 기록(폴드 숨김 아님).
  고EC 쌍둥이 = 하루 평균 정답 ≥ 1.2.  발동 = sg 0..h 누적평균 ≥ 1.0 & 고EC 쌍둥이 ≥ 1.
  안 EXA: A = 보이는 보통 쌍둥이(하루 평균 < 1.0) 하루 평균의 평균(없으면 발동 안 함).  안 EXM: A = 학습 기록 중 하루 평균 < 1.0 인 기록들의 하루 평균 중앙값(고정 상수).
  발동 행: pred_h = clip(sg_h + (A − sg 0..h 누적평균)).  v7 앵커 기록은 v7 그대로(기준선 V7 일 때), NONE 기준선에서는 sg 위에만.
판정(SRJ1 의 작은 표본 경로, 사용자 10-13 결정): 기준선 NONE·V7 각각, rv1 DIAG10·P2LOO·EL1 × 시드 47/1414/6464 + EV × 시드 7/101/2024 = 12칸
  (1) 12칸 모두 2차 행 RMSE 개선(변화 없음 = 개선 아님); (2) 어느 칸에서도 2차 기록 ΔSSE > 1e-6 없음; (3) 바뀐 2차 기록 ≥ 1.  모두 → PASS.
  참고 출력: 전체 행(1차 포함) 칸별 변화·발동 기록·손해 기록; 평가 60 에 적용했을 때 발동 기록과 12·13회차 대비 변화(서술).
"""
import os, sys, glob
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1)
NORM_MED = float(np.median(DM[TRAIN & (DM < 1.0)])); print(f"EXM constant A = {NORM_MED:.3f}")
def twins(i, h, vis):
    m = vis.copy(); m[i] = False; c = np.where(m)[0]
    return c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]
def fire_A(i, h, cmh, vis, arm):
    if cmh < 1.0: return np.nan
    t = twins(i, h, vis); t = t[np.isfinite(DM[t])]
    if not (DM[t] >= 1.2).any(): return np.nan
    if arm == "EXM": return NORM_MED
    lo = t[DM[t] < 1.0]; return float(DM[lo].mean()) if len(lo) else np.nan
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
ARMS = ["EXA", "EXM"]; res = []; rows = []; fired = []
def run(Gv, y, sg, p2, hours, cm, recrows, v7val, hsf, tag, sd):
    base_v7 = sg.copy(); P = {(a, b): sg.copy() for a in ARMS for b in ("NONE", "V7")}
    for r, ix in recrows.items():
        if r in v7val: base_v7[ix] = np.clip(sg[ix] + (v7val[r] - cm[ix]), lo_, hi_)
        vis = TRAIN & ~np.array([rid(k) in hsf(r) for k in range(N)]) if hsf(r) else TRAIN.copy()
        for a in ARMS:
            Ah = np.array([fire_A(idx[r], int(hours[k]), cm[k], vis, a) for k in ix]); m = np.isfinite(Ah)
            P[(a, "V7")][ix] = base_v7[ix]
            if m.any():
                P[(a, "NONE")][ix[m]] = np.clip(sg[ix[m]] + (Ah[m] - cm[ix[m]]), lo_, hi_)
                if r not in v7val: P[(a, "V7")][ix[m]] = P[(a, "NONE")][ix[m]]
                fired.append(dict(val=tag, seed=str(sd), arm=a, rec=r, nrows=int(m.sum()), A=float(np.nanmean(Ah)), y=float(np.mean(y[ix])), sg=float(np.mean(sg[ix])), v7=r in v7val))
    B = {"NONE": sg, "V7": base_v7}
    for a in ARMS:
        for bn in B:
            b_, n_ = B[bn], P[(a, bn)]
            res.append(dict(val=tag, seed=str(sd), arm=a, base=bn, all_pct=100 * (rm(n_ - y) / rm(b_ - y) - 1), p2_pct=100 * (rm((n_ - y)[p2]) / rm((b_ - y)[p2]) - 1)))
            ds = pd.Series(((n_ - y) ** 2 - (b_ - y) ** 2)).groupby(Gv.rec.values).sum()
            for r_, d in ds.items(): rows.append(dict(val=tag, seed=str(sd), arm=a, base=bn, rec=r_, p2=NUM[idx[r_]] >= 179, dsse=d))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.sub_ec.values; sg = Gv.sg.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fold_of = Gv.groupby("rec").fold.first().to_dict(); hsf = lambda r: hidden.get(fold_of[r], set())
    recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    v7val = {r: EC[idx[p], 23] for r, p in V7.items() if r in recrows and p not in hsf(r)}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, hsf, v, sd); print("done", v, sd, flush=True)
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); y = Gv.y.values; sg = Gv.SG2.values; ea = Gv.EA2.values; p2 = Gv.day.values >= 179
    cm = Gv.groupby("rec").SG2.transform(lambda s: s.expanding().mean()).values; recrows = {r: g.index.values for r, g in Gv.groupby("rec")}
    fo = {r: int(Gv.fold.values[ix[0]]) for r, ix in recrows.items()}; hsf = lambda r: hid[fo[r]]
    v7val = {r: float(ea[recrows[r][0]]) for r in set(Gv.rec[Gv.anchored.astype(bool)])}
    run(Gv, y, sg, p2, Gv.hour.values, cm, recrows, v7val, hsf, "EV", sd); print("done EV", sd, flush=True)
S = pd.DataFrame(res); D = pd.DataFrame(rows); Fd = pd.DataFrame(fired); pd.set_option("display.width", 260)
print("pass-2 RMSE % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string())
print("all-rows % change:"); print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="all_pct").round(2).to_string())
if len(Fd): print("\nfired records (seed-collapsed):"); print(Fd.groupby(["arm", "val", "rec"]).agg(n=("seed", "size"), rows=("nrows", "mean"), A=("A", "mean"), sg=("sg", "mean"), y=("y", "first"), v7=("v7", "first")).round(3).to_string())
for a in ARMS:
    verdict = True
    for bn in ("NONE", "V7"):
        cells = S[(S.arm == a) & (S.base == bn)]; dd = D[(D.arm == a) & (D.base == bn) & D.p2]
        c1 = bool((cells.p2_pct < 0).all()); harm = dd[dd.dsse > 1e-6]; ch = dd[dd.dsse.abs() > 1e-6].rec.unique()
        ok = c1 and len(harm) == 0 and len(ch) >= 1; verdict &= ok
        a1 = D[(D.arm == a) & (D.base == bn) & ~D.p2]; h1 = a1[a1.dsse > 1e-6].rec.unique(); g1 = a1[a1.dsse < -1e-6].rec.unique()
        print(f"\n{a} vs {bn}: p2 cells better {int((cells.p2_pct < 0).sum())}/{len(cells)} | p2 harmed {harm.rec.nunique()} | p2 changed {len(ch)} -> {'pass' if ok else 'fail'}  || (ref) pass-1 records better {len(g1)} worse {len(h1)} {list(h1)[:6]}")
    print(f"EXC1 {a}:", "PASS" if verdict else "FAIL")
# ---- descriptive: eval application ----
Xt = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); Xt["rec"] = Xt.row_id.str[:7]
for lab, d in (("12th", "12회차_2026-10-11(나)"), ("13th", "13회차_2026-10-12(나)")):
    f = [p for p in glob.glob(os.path.join(env.ROOT, "제출", d, "*.csv")) if "submission" in os.path.basename(p)][0]
    P = pd.read_csv(f); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int); P = P[P.rec.str[:3].isin(["F13", "F47"])].sort_values(["rec", "h"])
    P["cm"] = P.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean())
    print(f"\n== eval application on {lab} ({os.path.basename(f)})")
    for a in ARMS:
        out = []
        for r, g in P.groupby("rec"):
            i = idx[r]; Ah = np.array([fire_A(i, int(h), c, TRAIN.copy(), a) for h, c in zip(g.h, g.cm)]); m = np.isfinite(Ah)
            if m.any():
                new = np.where(m, np.clip(g.sub_ec.values + (np.nan_to_num(Ah) - g.cm.values), lo_, hi_), g.sub_ec.values)
                out.append(dict(rec=r, rows=int(m.sum()), A=round(float(np.nanmean(Ah)), 3), old_mean=round(g.sub_ec.mean(), 3), new_mean=round(new.mean(), 3), rms_change=round(rm(new - g.sub_ec.values), 3)))
        print(a, pd.DataFrame(out).to_string() if out else "no fire")
