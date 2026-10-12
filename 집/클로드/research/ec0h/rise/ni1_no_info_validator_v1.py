# -*- coding: utf-8 -*-
"""NI1 (2026-10-12 집 클로드; 사용자: '그 23개로 정보 없는 기록 전용 검증').  진단(표본 23 — 채택 판정 아님).
대상: 학습 기록 중 진단용 연결상 같은 동 전날·다음날 모두 정답 기록이 아닌 23개(NF1 'neither').  비교군: 같은 동 앞뒤 정답 있는 기록(261).
기본 q = DIAG10 CUR sg(시드 3).  방안(모두 평가에서 쓸 수 있는 정보: 대상 0..h 입력 + 학습 정답):
  S0 기본 / S1 같은 날짜(DM11) 다른 학습 기록들의 하루 평균 EC 로 수준 맞춤(w=1) / S2 S1 을 w=.5
  S3 d−1·d+1 날짜 학습 기록들(동 무관)의 평균 EC(전날은 23시, 다음날은 0시) w=1 / S4 같은 농장 ±7일 학습 기록 하루 평균의 중앙값 w=.5
  적용: q_h + w·(T − cm_h), cm_h = q 의 0..h 누적평균(뒤 시각 예측 안 씀).
보고: 두 집단의 방안별 RMSE(시드별), 대상 23 기록별 기본 오차.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
here = os.path.dirname(os.path.abspath(__file__))
NF = pd.read_csv(os.path.join(here, "nf1_v1.csv")); CAL = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
NFt = NF[NF.split == "train"].copy(); NFt["hp"] = NFt.true_prev_lab.fillna(False).astype(bool); NFt["hn"] = NFt.true_next_lab.fillna(False).astype(bool)
G_none = list(NFt[~NFt.hp & ~NFt.hn].rec); G_both = list(NFt[NFt.hp & NFt.hn].rec)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
lab = {rid(i): i for i in np.where(TRAIN)[0]}
by_day = CAL[CAL.split == "train"].dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
def targets(r):
    d = DAY.get(r); f = r[:3]; out = {}
    sib = [x for x in by_day.get(d, []) if x != r]; out["S1"] = np.mean([EC[lab[x]].mean() for x in sib]) if sib else np.nan
    nb = [EC[lab[x], 23] for x in by_day.get(d - 1, [])] + [EC[lab[x], 0] for x in by_day.get(d + 1, [])]; out["S3"] = np.mean(nb) if nb else np.nan
    sea = [EC[lab[x]].mean() for dd in range(int(d) - 7, int(d) + 8) for x in by_day.get(dd, []) if x != r and x[:3] == f]; out["S4"] = np.median(sea) if sea else np.nan
    return out
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
for name, grp in (("NO same-house info (23)", G_none), ("same-house prev & next labelled (261)", G_both)):
    recs = [r for r in grp if r in Y.index and np.isfinite(DAY.get(r, np.nan))]
    TG = {r: targets(r) for r in recs}; res = []
    for sd, G in O.groupby("seed"):
        Q = G.pivot_table(index="rec", columns="hour", values="sg").loc[recs].values; cm = np.cumsum(Q, 1) / np.arange(1, 25); Yv = Y.loc[recs].values
        row = dict(seed=sd, S0=rm(Q - Yv))
        for k, (src, w) in {"S1": ("S1", 1), "S2": ("S1", .5), "S3": ("S3", 1), "S4": ("S4", .5)}.items():
            T = np.array([TG[r][src] for r in recs]); P = Q.copy(); ok = np.isfinite(T)
            P[ok] = Q[ok] + w * (T[ok][:, None] - cm[ok]); row[k] = rm(P - Yv)
        res.append(row)
    R = pd.DataFrame(res); print(f"\n=== {name}: n {len(recs)}"); print(R.round(4).to_string(index=False))
    if name.startswith("NO"):
        Q = O[O.seed == O.seed.iloc[0]].pivot_table(index="rec", columns="hour", values="sg").loc[recs]
        e = (Y.loc[recs] - Q); print("  per record: day-mean error, truth day mean, sibling mean (S1):")
        print("  " + ", ".join(f"{r} {e.loc[r].mean():+.2f} (y {Y.loc[r].mean():.2f}, sib {TG[r]['S1']:.2f})" for r in e.mean(1).abs().sort_values(ascending=False).index))
