# -*- coding: utf-8 -*-
"""DGX4 (2026-10-16 집 클로드; 사용자 'ec 도 이런 식으로 개선 부분 찾아봐' — 온도 DGX1~3 처럼 EC 부품 분해).  서술 진단(판정 없음), 실행 전 커밋.  EC.
저장 예측 nh1_rows_v1.csv.gz (SZ2 새 시드 1357·2468·3579, 검증기 P2LOO·EL1·DIAG10)로 27번 EC 부품의 몫을 같은 잣대로 분해:
  MT1   : A1 (= .9 NH + .1 MT1)  vs  NH (= (A1 − .1 MT1)/.9)            — 10% 섞는 GRU 다중 과제
  K8    : NH + FL K8 EC 앵커(fl3, p >= .8, |dT| <= 1.5, 폴드 밖 정답 전날; pred = x + (A23 − cm))  vs  NH
  HR5   : SH (구조판 S + HR5)  vs  S                                     — 하루 수준 TabPFN 보정 (구조판 위에서 잰 몫)
  SEASON: A1 (계절 지표 포함)  vs  A2 (계절 지표 뺌)
  GEN   : A1 (일반판)  vs  S (구조판, SG2 포함 — W2 는 없음)              — 참고
지표: 3시드 평균 SSE 변화(음수 = 좋음), 상대 %, 1차/2차, 보통/고EC, 시각대, 실내온도 구간, 상위 3기록 몫·빼고 남는 이득, 좋아진/나빠진 기록 비율,
      하루 수준 몫 vs 하루 안 모양 몫, 고침 상관(−기준오차 vs 변화), 검증기 간 기록 이득 상관.  마지막: 부품 x 조각(1·2차 x 보통·고EC) 방향 일치 표.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dgx4_ec_component_dissection_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "dgx4_ec_component_dissection_v1.py")
y = R.y.values; hr = R.hour.values
R["it"] = [TIN.at[r, h] if r in TIN.index else np.nan for r, h in zip(R.rec, R.hour)]
def anchor(x):
    out = x.copy()
    for (v, k, r), ii in GRP:
        if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
        xx = x[ii]; out[ii] = xx + (EC_[idx_[FL3[r][0]], 23] - np.cumsum(xx) / np.arange(1, len(xx) + 1))
    return out
PAIRS = {}
for s in SEEDS:
    nh = (R[f"A1_{s}"].values - .1 * mt) / .9
    PAIRS[("MT1", s)] = (R[f"A1_{s}"].values, nh); PAIRS[("K8", s)] = (anchor(nh), nh); PAIRS[("HR5", s)] = (R[f"SH_{s}"].values, R[f"S_{s}"].values)
    PAIRS[("SEASON", s)] = (R[f"A1_{s}"].values, R[f"A2_{s}"].values); PAIRS[("GEN", s)] = (R[f"A1_{s}"].values, R[f"S_{s}"].values)
COMP = ["MT1", "K8", "HR5", "SEASON", "GEN"]; VALS = ["P2LOO", "EL1", "DIAG10"]
for c in COMP:
    R["g_" + c] = np.mean([(PAIRS[(c, s)][0] - y) ** 2 - (PAIRS[(c, s)][1] - y) ** 2 for s in SEEDS], 0)
    R["d_" + c] = np.mean([PAIRS[(c, s)][0] - PAIRS[(c, s)][1] for s in SEEDS], 0); R["e_" + c] = np.mean([PAIRS[(c, s)][1] - y for s in SEEDS], 0)
R["seg"] = np.where(R.p2, "p2", "p1") + "_" + np.where(R.hi, "hi", "norm")
R["hgrp"] = pd.cut(R.hour, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"]); R["band"] = pd.cut(R.it, [-99, 10, 15, 20, 99], labels=["<=10", "10-15", "15-20", ">20"])
pd.set_option("display.width", 260)
print("\n[1] totals: rel % (3 seeds, cell-wise mean), sum gain, top3 record share, without top3, records better/worse, corr_fix")
rows = []
for c in COMP:
    for v in VALS:
        g = (R.validator == v).values; S = R[g]; Rr = S.groupby("rec")["g_" + c].sum(); top = Rr.abs().nlargest(3).index
        rel = np.mean([rm((PAIRS[(c, s)][0] - y)[g]) / rm((PAIRS[(c, s)][1] - y)[g]) - 1 for s in SEEDS]) * 100
        rel2 = np.mean([rm((PAIRS[(c, s)][0] - y)[g & R.p2.values]) / rm((PAIRS[(c, s)][1] - y)[g & R.p2.values]) - 1 for s in SEEDS]) * 100
        cf = np.corrcoef(-S["e_" + c], S["d_" + c])[0, 1] if S["d_" + c].std() > 0 else np.nan
        rows.append(dict(comp=c, val=v, rel=rel, rel_p2=rel2, gain=S["g_" + c].sum(), top3=Rr.abs().nlargest(3).sum() / max(Rr.abs().sum(), 1e-9), wo_top3=Rr.drop(top).sum(),
                         better=(Rr < -1e-9).mean(), worse=(Rr > 1e-9).mean(), corr_fix=cf))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for col, title in (("seg", "[2] gain per row x1000 by pass x high-EC"), ("hgrp", "[3] by hour group"), ("band", "[4] by in_temp band"), ("farm", "[5] by farm")):
    T = R.groupby(["validator", col], observed=True)[["g_" + c for c in COMP]].mean() * 1000; T.columns = COMP; n = R.groupby(["validator", col], observed=True).size().rename("rows")
    print(f"\n{title}"); print(pd.concat([n, T.round(3)], axis=1).to_string())
print("\n[6] level (record mean) vs within-day shape part")
rows = []
for c in COMP:
    for v in VALS:
        S = R[R.validator == v]; n = S.groupby("rec").size(); e = S.groupby("rec")["e_" + c].mean(); dm = S.groupby("rec")["d_" + c].mean()
        lev = float((n * ((e + dm) ** 2 - e ** 2)).sum()); tot = float(((S["e_" + c] + S["d_" + c]) ** 2 - S["e_" + c] ** 2).sum()); rows.append(dict(comp=c, val=v, total=tot, level=lev, shape=tot - lev))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n[7] record-gain agreement across validators (corr; same-sign share, records |gain| > .01)")
for c in COMP:
    Pv = R.groupby(["validator", "rec"])["g_" + c].sum().unstack(0); out = []
    for a in VALS:
        for b in VALS:
            if a < b:
                Q = Pv[[a, b]].dropna(); Q = Q[Q.abs().max(axis=1) > .01]
                if len(Q) >= 5: out.append(f"{a}-{b} n{len(Q)} same {np.mean(np.sign(Q[a]) == np.sign(Q[b])):.2f} r {Q[a].corr(Q[b]):.2f}")
    print(f"  {c}: " + " | ".join(out))
print("\n[8] slice table: sign of summed gain per component x segment x validator (better / cells)")
rows = []
for c in COMP:
    r_ = dict(comp=c)
    for sg in sorted(R.seg.unique()):
        sgn = [np.sign(R[(R.validator == v) & (R.seg == sg)]["g_" + c].sum()) for v in VALS if ((R.validator == v) & (R.seg == sg)).sum() >= 24]
        r_[sg] = f"{sum(x < 0 for x in sgn)}/{len(sgn)}"
    rows.append(r_)
print(pd.DataFrame(rows).to_string(index=False))
