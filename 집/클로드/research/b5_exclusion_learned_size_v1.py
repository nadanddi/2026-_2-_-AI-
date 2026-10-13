# -*- coding: utf-8 -*-
"""B5 (2026-10-13 집 클로드; 사용자: '리스트 전부 다 해봐' — B5: 같은 날 고EC 배타를 '크기까지' 데이터로).  사전 고정(커밋 후 실행).  EC.  과거 정보만(같은 날 학습 정답·대상 0..h 입력).
EXC1(6.780) 은 방향 6/6 맞았으나 대체값(보통 쌍둥이 .4~.5)이 너무 낮아 실패.  여기선 내림 폭 δ 를 '같은 조건 기록들의 잔차'로 기록 하나 빼기 교차 적합.
발동: 대상 h 시 행에서 sg 0..h 누적평균 ≥ 1.0 & 외기 0..h 쌍둥이(학습, 숨김 아님) 중 하루 평균 정답 ≥ 1.2 가 있음(EXC1 과 같음).
보정: pred_h = sg_h + δ_(−r),  δ_(−r) = 같은 검증 칸에서 발동한 다른 기록들의 (실제 하루 평균 − sg 하루 평균) 평균(대상 제외), 발동 기록이 2 미만이면 보정 없음.
판정 한계(밝혀 둠): 2차 구간에는 발동 사례가 없음(EXC1) → 현재 규칙(2차 12칸)으로는 판정 불가.  서술로 1차 발동 기록의 교차 적합 결과와 평가 적용값만 보고.
"""
import os, sys, glob
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "exc1_same_date_high_exclusion_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "exc1_same_date_high_exclusion_v1.py")
exec(src[:src.index("L7 = pd.read_csv")])  # EC, X, TRAIN, DM, twins, idx, rid, rm, lo_, hi_
def fired_rows(r, hours, cm, vis):
    i = idx[r]; m = np.zeros(len(hours), bool)
    for k, (h, c) in enumerate(zip(hours, cm)):
        if c < 1.0: continue
        t = twins(i, int(h), vis); t = t[np.isfinite(DM[t])]
        m[k] = bool((DM[t] >= 1.2).any())
    return m
out = []
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
cellsets = []
for (v, sd), Gv in O.groupby(["validator", "seed"]):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True); hidden = Gv.groupby("fold").rec.apply(set).to_dict(); fo = Gv.groupby("rec").fold.first().to_dict()
    cellsets.append((v, sd, Gv.rename(columns={"sub_ec": "y"}), lambda r, hidden=hidden, fo=fo: hidden.get(fo[r], set())))
for sd, Gv in R.groupby(R.seed.astype(str)):
    Gv = Gv.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"SG2": "sg"}); fo = {r: int(g.fold.iloc[0]) for r, g in Gv.groupby("rec")}
    cellsets.append(("EV", sd, Gv, lambda r, fo=fo: hid[fo[r]]))
for v, sd, Gv, hsf in cellsets:
    Gv["cm"] = Gv.groupby("rec").sg.transform(lambda s: s.expanding().mean()); F = {}
    for r, g in Gv.groupby("rec"):
        vis = TRAIN & ~np.array([rid(k) in hsf(r) for k in range(N)])
        m = fired_rows(r, g.hour.values, g.cm.values, vis)
        if m.any(): F[r] = (g.index.values, m, float(g.y.mean() - g.sg.mean()))
    for r, (ix, m, res_r) in F.items():
        others = [F[q][2] for q in F if q != r]
        if len(others) < 2: continue
        d = float(np.mean(others)); y = Gv.y.values[ix]; s = Gv.sg.values[ix]; new = np.where(m, np.clip(s + d, lo_, hi_), s)
        out.append(dict(val=v, seed=str(sd), rec=r, p2=NUM[idx[r]] >= 179, rows=int(m.sum()), delta=d, y=y.mean(), sg=s.mean(), dsse=float(np.sum((new - y) ** 2) - np.sum((s - y) ** 2))))
T = pd.DataFrame(out); pd.set_option("display.width", 200)
print(T.groupby(["val", "rec"]).agg(n=("seed", "size"), p2=("p2", "first"), rows=("rows", "mean"), delta=("delta", "mean"), y=("y", "first"), sg=("sg", "mean"), dsse=("dsse", "mean")).round(3).to_string())
print("\nby validator: records fired, improved, total dSSE:", T.groupby("val").apply(lambda g: (g.rec.nunique(), int((g.groupby("rec").dsse.mean() < 0).sum()), round(float(g.groupby("rec").dsse.mean().sum()), 3))).to_dict())
# eval application with delta from ALL training-geometry fired records (DIAG10 seed-mean residuals, descriptive)
dall = float(T[T.val == "DIAG10"].groupby("rec").apply(lambda g: g.y.iloc[0] - g.sg.mean()).mean()) if (T.val == "DIAG10").any() else np.nan
print(f"\nδ from DIAG10 fired records (mean residual): {dall:+.3f}")
f = glob.glob(os.path.join(env.ROOT, "제출", "12회차_2026-10-11(나)", "submission_19.csv"))[0]
P = pd.read_csv(f); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int); P = P.sort_values(["rec", "h"]); P["cm"] = P.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean())
for r, g in P.groupby("rec"):
    m = fired_rows(r, g.h.values, g.cm.values, TRAIN.copy())
    if m.any(): print(f"eval {r}: rows {int(m.sum())} | 12th mean {g.sub_ec.mean():.3f} -> {np.where(m, np.clip(g.sub_ec + dall, lo_, hi_), g.sub_ec).mean():.3f}")
