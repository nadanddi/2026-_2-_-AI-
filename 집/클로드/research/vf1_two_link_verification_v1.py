# -*- coding: utf-8 -*-
"""VF1 (2026-10-14 집 클로드; 사용자: '좀 더 검증해봐').  서술(판정 아님, 리더보드 안 씀).  평가 순수 창 두 덩어리의 첫 짝 검증:
  F47_221 ← F47_145 (GP-RS+LK3 q .67, 조상 EC 2.14),  F13_221 ← F13_218 (q .38, 조상 EC .39).  비교 후보: 각 대상의 다른 링커 후보들.
근거(모두 입력·학습 정답만; 대상 기록의 하루 전체 입력은 '검증 근거' 로만 씀 — 예측에는 안 씀):
  (1) 링커별 1순위(v4·v5·v7·v10·v11·v12·v13·v14·v15·v15r15), (2) 번호 구조: 사이 번호들이 학습/평가/없음, 후보의 cert 다음날,
  (3) 학습 달력 DMT3e 날짜(대상 날짜는 0시 외기 일치 정답 기록의 최빈 날짜), (4) 23시→0시 실내·외기온 차,
  (5) 동 불변값(하루 전체: 유동팬 가동률·환기 시작 온도·난방 설정·CO₂ 첫 시각·차광 시각) 거리, (6) 고EC 운영 지문(커튼 9시 닫힘·16시 열림, CO₂ 9·11·13시, 0~5시 유동팬 꺼짐).
"""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERTS = {}
for a, b, c in zip(TL.a1, TL.b, TL.cert):
    if c: CERTS.setdefault(rid(int(a)), []).append(rid(int(b)))
TARGETS = ["F47_221", "F13_221"]
# (1) linkers
rows = []
for f in sorted(glob.glob(os.path.join(env.LOCAL, "linker_v*", "*eval*.csv"))):
    d = pd.read_csv(f, encoding="utf-8-sig"); key = "row" if "row" in d.columns else "rec"
    pc = next((c for c in ("prev0", "prev_h0", "L_prev0", "prev", "v11") if c in d.columns), None); pp = next((c for c in ("p0", "p_h0", "L_p0", "p") if c in d.columns), None)
    for t in TARGETS:
        s = d[d[key] == t]
        if len(s): rows.append(dict(file=os.path.relpath(f, env.LOCAL), target=t, pick=s[pc].iloc[0], p=round(float(s[pp].iloc[0]), 3) if pp else np.nan))
L = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("(1) linker picks:"); print(L.pivot_table(index="file", columns="target", values="pick", aggfunc="first").to_string())
print(L.pivot_table(index="file", columns="target", values="p", aggfunc="first").to_string())
# (2) number structure
TESTSET = {rid(i) for i in range(N) if not TRAIN[i]}
for t, cand in (("F47_221", "F47_145"), ("F13_221", "F13_218")):
    f = t[:3]; lo, hi = int(cand[4:]), int(t[4:])
    status = {n: ("train" if f"{f}_{n:03d}" in idx and TRAIN[idx[f"{f}_{n:03d}"]] else "eval" if f"{f}_{n:03d}" in TESTSET else "absent") for n in range(lo, hi + 1)}
    print(f"\n(2) {cand} -> {t}: numbers between: ", {k: v for k, v in status.items() if k not in (lo, hi)} if hi - lo < 12 else f"{sum(v == 'train' for v in status.values())} train / {sum(v == 'eval' for v in status.values())} eval / {sum(v == 'absent' for v in status.values())} absent over {hi - lo - 1}")
    print(f"    cert successors of {cand}: {CERTS.get(cand)} | {cand} EC day-mean {DM[idx[cand]]:.3f}, EC 23h {EC[idx[cand], 23]:.3f}")
# (3) training calendar
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
def tdate(r):
    i = idx[r]; vi = np.where(TRAIN)[0]; vi = vi[vi != i]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[i, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return (pd.Series(ds).value_counts().head(3).to_dict() if ds else None)
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); G = {r: g.sort_values("h").set_index("h") for r, g in _X.groupby("rec")}
def inv(r):
    g = G[r]; heat = g.act_heating > 0; vent = np.where(g.act_vent.values > 0)[0]; co2 = np.where(g.act_co2.values > 0)[0]; sh = np.where(g.act_shade.values > 50)[0]; th = g.act_thermal.values
    return dict(fan_duty=round(float((g.act_circfan > 0).mean()), 2), fan_0_5=round(float((g.act_circfan.values[:6] > 0).mean()), 2), vent_temp=round(float(g.in_temp.values[vent[0]]), 1) if len(vent) else np.nan,
                heat_set=round(float(g.in_temp[heat].min()), 1) if heat.any() else np.nan, co2_hours=list(co2), shade_first=int(sh[0]) if len(sh) else None,
                thermal_closed_hours=list(np.where(th >= 50)[0]), in_temp_0=g.in_temp.values[0], in_temp_23=g.in_temp.values[23], out_0=g.out_temp.values[0], out_23=g.out_temp.values[23])
for t in TARGETS:
    cands = sorted(set(L[L.target == t].pick.dropna()) | ({"F47_145"} if t == "F47_221" else {"F13_218", "F13_123"}))
    print(f"\n===== {t}  (training-calendar date votes {tdate(t)})"); it = inv(t); print("target:", it)
    for c in cands:
        if c not in G: continue
        ic = inv(c); lab = TRAIN[idx[c]]
        print(f"  cand {c} [{'labelled' if lab else 'eval'}; EC {DM[idx[c]]:.2f}] date {DAYT.get(c)}/{SEGT.get(c)} | dTin(23->0) {abs(ic['in_temp_23'] - it['in_temp_0']):.1f} dTout {abs(ic['out_23'] - it['out_0']):.1f} | fan {ic['fan_duty']} (0-5h {ic['fan_0_5']}) vent {ic['vent_temp']} heat {ic['heat_set']} co2h {ic['co2_hours']} shade {ic['shade_first']} thermal {ic['thermal_closed_hours'][:3]}..{ic['thermal_closed_hours'][-3:]}")
# ---- VF1b: 같은 ID 바로 앞 번호 기록(평가/학습)과 F47_145 이후 F47 고EC 지문 기록 ----
print("\n[VF1b] immediate same-ID predecessors and fingerprint")
for r in ["F13_218", "F13_220", "F47_218", "F47_219", "F47_220", "F13_123", "F13_124", "F47_146", "F47_147"]:
    if r not in G: print(r, "absent"); continue
    ic = inv(r); print(f"  {r} [{'train EC %.2f' % DM[idx[r]] if TRAIN[idx[r]] else 'eval'}] date {DAYT.get(r)}/{SEGT.get(r)} cal-votes {tdate(r) if not TRAIN[idx[r]] else '-'} | fan {ic['fan_duty']} co2h {ic['co2_hours']} shade {ic['shade_first']} heat {ic['heat_set']} vent {ic['vent_temp']} | in 0h {ic['in_temp_0']} 23h {ic['in_temp_23']} out 0h {ic['out_0']} 23h {ic['out_23']}")
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; pm = P14.groupby("rec").sub_ec.mean()
print("  14th EC day-mean for F13_220 / F47_220:", {r: round(float(pm[r]), 3) for r in ("F13_220", "F47_220") if r in pm.index})
