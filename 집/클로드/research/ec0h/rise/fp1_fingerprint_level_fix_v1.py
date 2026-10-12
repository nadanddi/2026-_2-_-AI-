# -*- coding: utf-8 -*-
"""FP1 (2026-10-12 집 클로드; 사용자: '지문 기반 크기 보정 실험').  사전 고정(이 파일을 커밋한 뒤 실행).
배경: HS1~HS3(6.690) — 고EC 동은 운영 일정 지문(커튼 9~16시, CO2 9·11·13시 짧게, 유동팬·환기 꺼짐, 실내 CO2 높음)으로 거의 완벽히 구별되나
모델은 그 크기를 낮게 냄(예 1.69→1.19).  6.177 은 '예측이 높은 기록 전체'를 보정해 실패 — 여기서는 지문으로 대상만 골라 보정.

인과 지문 G(rec, h) — 자기 기록의 0..h 입력만 (뒤 시각·다른 기록 입력 안 씀):
  g1 유동팬: 0..h 중 9시를 뺀 시각의 act_circfan 최대 ≤ 5
  g2 환기:   0..h 의 act_vent 최대 ≤ 5
  g3 (h ≥ 8)  act_shade[7] == 0 그리고 act_shade[8] == 0          (보통 일정은 7~8시에 닫힘)
  g4 (h ≥ 10) act_shade[10] ≥ 95 그리고 0..h 중 8~14시 act_co2 최대 < 30
  g5 (h ≥ 17) act_shade[17] == 0
  g6 (h ≥ 10) 0..h 중 8~h시 in_co2 평균 ≥ 390                       (고EC 438 vs 보통 346 의 중간)
  G = 모두 참.  (h < 8 에서는 g1·g2 만 — 넓은 문, 보정 크기를 시간대별로 따로 학습해 자연히 약해지게 함)
보정 크기 (블록 밖 학습): 달력 날짜 DM11 을 5일 블록으로 나눔; 대상 기록의 블록을 뺀 학습 기록들 중 G 가 참인 행으로 시간대 묶음별 계수 학습.
  시간대 묶음: 0–7 / 8–9 / 10–16 / 17–23.  달력 없는 기록은 보정 안 함.
  A1 곱셈: p' = p · k,  k = Σ(y·p)/Σ(p²) (G 참인 학습 행, 같은 검증기·시드의 예측 p).
  A2 덧셈: p' = p + c,  c = mean(y − p).
검증기 x 시드 (모두 OOF 예측 그대로 사용, 보정만 덧붙임): DIAG10·EL1·P2LOO (CUR tag, sg 열, 시드 47/1414/6464),
  EV BASE SG2·EA2 (시드 7/101/2024).  = 5 x 3 = 15 칸.
채택(각 안 따로, 본페로니 2안): 15칸 모두 전체 RMSE 개선 + DIAG10 기록 단위 부트스트랩(2000) P(악화) < .0125 (시드 3개 모두) + 개선 상위 1기록 비중 ≤ .5 (DIAG10 시드 평균).
서술 보고: 지문 문 G 의 학습 정밀도(G 참 행 중 고EC 기록 비율, 시간대별), 고EC/보통별 RMSE 변화, 평가 기록에 적용했을 때 바뀌는 기록(12회차 대비, 계수는 전체 학습으로 — 서술만, 제출 아님).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
piv = {v: X.pivot(index="rec", columns="h", values=v).reindex(columns=range(24)) for v in ("act_circfan", "act_vent", "act_shade", "act_co2", "in_co2")}
recs = piv["act_shade"].index
cf, ve, sh, co, ic = (piv[v].values for v in ("act_circfan", "act_vent", "act_shade", "act_co2", "in_co2"))
G = np.zeros((len(recs), 24), bool)
for h in range(24):
    hrs = np.arange(h + 1)
    g = np.nan_to_num(np.nanmax(np.where(hrs[None, :] == 9, 0, cf[:, :h + 1]), 1), nan=0) <= 5
    g &= np.nan_to_num(np.nanmax(ve[:, :h + 1], 1), nan=0) <= 5
    if h >= 8: g &= (sh[:, 7] == 0) & (sh[:, 8] == 0)
    if h >= 10:
        g &= (sh[:, 10] >= 95) & (np.nanmax(co[:, 8:min(h, 14) + 1], 1) < 30)
        g &= np.nanmean(ic[:, 8:h + 1], 1) >= 390
    if h >= 17: g &= sh[:, 17] == 0
    G[:, h] = g
GD = pd.DataFrame(G, index=recs)
CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["rec"] = Y.row_id.str[:7]; ymean = Y.groupby("rec").sub_ec.mean()
BAND = np.array([0] * 8 + [1] * 2 + [2] * 7 + [3] * 7)

def frame(df, pcol):
    d = df[["rec", "hour", "y", pcol]].rename(columns={pcol: "p"}).copy()
    d["G"] = GD.reindex(d.rec).values[np.arange(len(d)), d.hour.values] if len(d) else []
    d["day"] = d.rec.map(DAY); d["blk"] = (d.day // 5); d["band"] = BAND[d.hour.values]
    return d.dropna(subset=["y", "p"])

def correct(d):
    out = {"A1": d.p.values.copy(), "A2": d.p.values.copy()}
    for (b, bd), idx in d[d.G & d.day.notna()].groupby(["blk", "band"]).groups.items():
        tr = d[d.G & d.day.notna() & (d.blk != b) & (d.band == bd)]
        if len(tr) < 24: continue
        k = float((tr.y * tr.p).sum() / (tr.p ** 2).sum()); c = float((tr.y - tr.p).mean())
        pos = d.index.get_indexer(idx); out["A1"][pos] = d.p.values[pos] * k; out["A2"][pos] = d.p.values[pos] + c
    return out

def load():
    cells = {}
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[O.tag == "CUR"]; O["rec"] = O.row_id.str[:7]; O = O.rename(columns={"sub_ec": "y"})
    for (v, s), g in O.groupby(["validator", "seed"]): cells[(v, s)] = frame(g.reset_index(drop=True), "sg")
    R = pd.read_csv(os.path.join(env.ROOT, u"집", u"클로드", "research", "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv"))
    R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
    for s, g in R.groupby("seed"):
        for col in ("SG2", "EA2"): cells[("EV-" + col, s)] = frame(g.reset_index(drop=True), col)
    return cells

if __name__ == "__main__":
    cells = load(); rows = []; rng = np.random.default_rng(12)
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    for key, d in cells.items():
        d = d.reset_index(drop=True); C = correct(d); hi = d.rec.map(ymean).values >= 1
        row = dict(validator=key[0], seed=key[1], base=rm(d.p - d.y), G_rows=int(d.G.sum()), G_prec=float(np.mean(hi[d.G.values])) if d.G.any() else np.nan)
        for a in ("A1", "A2"):
            row[a] = rm(C[a] - d.y); row[a + "_hi"] = rm(C[a][hi] - d.y[hi]) - rm(d.p[hi] - d.y[hi]); row[a + "_lo"] = rm(C[a][~hi] - d.y[~hi]) - rm(d.p[~hi] - d.y[~hi])
            if key[0] == "DIAG10":
                e0 = pd.Series((d.p - d.y) ** 2).groupby(d.rec).sum(); e1 = pd.Series((C[a] - d.y) ** 2).groupby(d.rec).sum(); dlt = (e1 - e0).values
                B = rng.integers(0, len(dlt), (2000, len(dlt))); row[a + "_pworse"] = float((dlt[B].sum(1) >= 0).mean())
                gain = -dlt; row[a + "_topshare"] = float(gain.max() / gain.sum()) if gain.sum() > 0 else np.nan
        rows.append(row)
    T = pd.DataFrame(rows); pd.set_option("display.width", 250)
    for a in ("A1", "A2"): T[a + "_chg%"] = 100 * (T[a] / T.base - 1)
    print(T.round(4).to_string(index=False))
    for a in ("A1", "A2"):
        D = T[T.validator == "DIAG10"]
        ok = (T[a] < T.base).all() and (D[a + "_pworse"] < .0125).all() and D[a + "_topshare"].mean() <= .5
        print(f"\n{a}: all 15 cells better {bool((T[a] < T.base).all())} | DIAG10 pworse {D[a + '_pworse'].round(4).tolist()} | top share {D[a + '_topshare'].mean():.2f} -> {'ADOPT' if ok else 'FAIL'}")
    # descriptive: gate precision by hour band (train, all records with labels) and eval records gated
    tr = [r for r in recs if r in ymean.index]
    for bd, hs in ((0, range(0, 8)), (1, range(8, 10)), (2, range(10, 17)), (3, range(17, 24))):
        m = GD.loc[tr, list(hs)].any(axis=1); print(f"band {bd}: train records gated {int(m.sum())}, of which high-EC {int((ymean[tr][m] >= 1).sum())}")
    ev = [r for r in recs if r not in ymean.index]
    print("eval records gated at hour 23:", sorted(GD.loc[ev][GD.loc[ev, 23]].index.tolist()))
    print("eval records gated at hour 7:", sorted(GD.loc[ev][GD.loc[ev, 7]].index.tolist()))
